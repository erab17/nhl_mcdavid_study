"""
End-to-end modern xG pipeline.

Steps:
  1. Load MoneyPuck shots (2015-2023), build features.
  2. Train LightGBM with GroupKFold(game_id) -> out-of-fold xG for every shot.
  3. Evaluate (log loss / Brier / AUC) and compare to MoneyPuck's xGoal.
  4. Plot reliability curve + feature importance.
  5. Redo the McDavid career xG study with the modern, leak-free xG.

Run:  python src/run_pipeline.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

import modern_xg as mx

warnings.filterwarnings("ignore", category=UserWarning)
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_estimators=700,
    verbose=-1,
)


def train_oof(df, feats, n_splits=5):
    X, y, _, groups = mx.feature_matrix(df)
    X = df[feats]
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    importances = np.zeros(len(feats))
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        importances += model.booster_.feature_importance(importance_type="gain")
        print(f"  fold {fold+1}/{n_splits}  best_iter={model.best_iteration_}")
    return oof, importances / n_splits


def evaluate(name, y, p):
    return dict(
        model=name,
        log_loss=log_loss(y, p),
        brier=brier_score_loss(y, p),
        auc=roc_auc_score(y, p),
    )


def main():
    print("Loading MoneyPuck shots 2015-2023 ...")
    raw = mx.load_seasons()
    df = mx.build_features(raw)
    X, y, feats, groups = mx.feature_matrix(df)
    print(f"  modeled shots: {len(df):,}  | goal rate: {df['goal'].mean():.4f}  | features: {len(feats)}")

    print("Training LightGBM (GroupKFold OOF) ...")
    df["xg"], importances = train_oof(df, feats)

    # Persist scored shots so downstream analysis / the notebook never retrains.
    keep = feats + [
        "xg", "xGoal", "goal", "season", "game_id", "isPlayoffGame",
        "shooterPlayerId", "shooterName", "teamCode", "event",
    ]
    scored_path = os.path.join(os.path.dirname(__file__), "..", "data", "scored_shots.parquet")
    df[keep].to_parquet(scored_path, index=False)
    print(f"  saved scored shots -> {os.path.abspath(scored_path)}")

    # ---- evaluation ----
    base = np.full(len(df), df["goal"].mean())
    rows = [
        evaluate("baseline (mean rate)", df["goal"], base),
        evaluate("MoneyPuck xGoal", df["goal"], df["xGoal"].clip(1e-6, 1 - 1e-6)),
        evaluate("our LightGBM (OOF)", df["goal"], df["xg"]),
    ]
    metrics = pd.DataFrame(rows)
    print("\n=== Model comparison (whole population) ===")
    print(metrics.to_string(index=False))
    metrics.to_csv(os.path.join(FIG_DIR, "metrics.csv"), index=False)

    # ---- reliability curve ----
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, p in [("Our xG", df["xg"]), ("MoneyPuck xG", df["xGoal"])]:
        frac, mean_pred = calibration_curve(df["goal"], p, n_bins=20, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=3, label=label)
    ax.plot([0, 0.6], [0, 0.6], "k--", lw=1, label="perfect")
    ax.set_xlabel("Predicted xG"); ax.set_ylabel("Observed goal rate")
    ax.set_title("Reliability (calibration) curve"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "reliability.png"), dpi=120)

    # ---- feature importance ----
    imp = pd.Series(importances, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    imp.plot.barh(ax=ax)
    ax.set_title("LightGBM feature importance (gain)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "importance.png"), dpi=120)
    print("\n=== Top features by gain ===")
    print(imp.sort_values(ascending=False).head(12).to_string())

    # ---- McDavid career study (leak-free OOF xG, regular season only) ----
    mcd = df[(df["shooterPlayerId"] == mx.MCDAVID_ID) & (df["isPlayoffGame"] == 0)].copy()
    mcd["season_label"] = mcd["season"].astype(str) + "-" + (mcd["season"] + 1).astype(str).str[-2:]
    career = mcd.groupby("season_label").agg(
        shots=("goal", "size"),
        goals=("goal", "sum"),
        xg=("xg", "sum"),
        xg_mp=("xGoal", "sum"),
        xg_per_shot=("xg", "mean"),
        mean_dist=("shotDistance", "mean"),
    )
    career["goals_minus_xg"] = career["goals"] - career["xg"]
    career["finishing_pct"] = 100 * career["goals_minus_xg"] / career["xg"]
    pd.set_option("display.width", 200)
    print("\n=== Connor McDavid career: modern xG vs actual goals ===")
    print(career.round(2).to_string())
    career.to_csv(os.path.join(FIG_DIR, "mcdavid_career.csv"))

    # career figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.bar(career.index, career["goals"], alpha=0.6, label="Actual goals")
    ax.plot(career.index, career["xg"], "o-", color="C3", label="Our xG")
    ax.plot(career.index, career["xg_mp"], "s--", color="C2", label="MoneyPuck xG")
    ax.set_title("McDavid: goals vs expected goals"); ax.legend(); ax.tick_params(axis="x", rotation=45)
    ax = axes[1]
    ax.bar(career.index, career["goals_minus_xg"], color="C0")
    ax.axhline(0, color="k", lw=1)
    ax.set_title("Finishing: goals above expected (G - xG)"); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "mcdavid_career.png"), dpi=120)

    print(f"\nFigures written to {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
