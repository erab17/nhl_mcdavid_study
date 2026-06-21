"""
Predictive-validity harness  (next-step #1 from HANDOFF.md).

The honest scoreboard for an xG model is not in-sample log loss -- it is
*predictive validity*: does a player's xG in season N forecast their actual
goals in season N+1 better than their past goals do, and better than a rival
model's xG?  This script answers two questions:

  Q1  MODEL COMPARISON -- which season-N predictor best forecasts season-(N+1)
      goals: raw goals, our xG, or MoneyPuck's xG?
  Q2  FINISHING REPEATABILITY -- is goals-above-expected (G - xG) a repeatable
      skill year over year, and where does McDavid sit?  This is the honest
      test of whether the "McDavid is a +finisher" result is signal or noise.

Method (leak-free, strictly prospective):
  * WALK-FORWARD training.  For each test season N we train the LightGBM xG
    model on shots from seasons STRICTLY BEFORE N, then score season N's shots.
    Season N is therefore fully out of sample -- no temporal leakage, unlike the
    all-seasons GroupKFold OOF in run_pipeline.py (fine for the player study but
    optimistic for a forecast).
  * Aggregate to player-season (regular season only), pair season N with N+1,
    keep players with a meaningful sample in season N.
  * Compare predictors on pooled, out-of-sample RMSE / R^2 / rank correlation.

MoneyPuck's xGoal is used as-is; it is itself a production model trained on many
seasons, so the comparison is conservative (its model saw more data than ours).

Run:  python src/predictive_validity.py
Outputs: figures/predictive_validity_models.csv, figures/predictive_validity_finishing.csv,
         figures/predictive_validity.png, and a printed summary.
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
from scipy.stats import spearmanr

import modern_xg as mx
from run_pipeline import LGB_PARAMS

warnings.filterwarnings("ignore", category=UserWarning)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Cache of the prospective (walk-forward) per-shot xG so re-analysis / step-2 work
# never has to retrain. Like scored_shots.parquet, this is gitignored.
WF_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "scored_shots_walkforward.parquet")

MIN_SHOTS_N = 100      # season-N sample floor (≈ a regular top-9 forward; median is 99)
MIN_SHOTS_N1 = 20      # season-(N+1) floor so the target isn't pure injury noise


def walk_forward_xg(df: pd.DataFrame, feats, test_seasons):
    """Train on seasons < N, score season N. Returns df with a prospective 'xg_wf'."""
    df = df.copy()
    df["xg_wf"] = np.nan
    params = {**LGB_PARAMS}
    for N in test_seasons:
        tr = df["season"] < N
        te = df["season"] == N
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model = lgb.LGBMClassifier(**params)
        model.fit(df.loc[tr, feats], df.loc[tr, "goal"].astype(int))
        df.loc[te, "xg_wf"] = model.predict_proba(df.loc[te, feats])[:, 1]
        print(f"  season {N}: trained on {tr.sum():,} shots (< {N}), "
              f"scored {te.sum():,} shots")
    return df


def player_season_table(df: pd.DataFrame) -> pd.DataFrame:
    """Regular-season player-season aggregates of goals and the three xG flavors."""
    rs = df[(df["isPlayoffGame"] == 0) & df["xg_wf"].notna()]
    g = rs.groupby(["shooterPlayerId", "shooterName", "season"]).agg(
        shots=("goal", "size"),
        goals=("goal", "sum"),
        xg=("xg_wf", "sum"),       # our prospective xG
        xg_mp=("xGoal", "sum"),    # MoneyPuck production xG
    ).reset_index()
    return g


def make_pairs(ps: pd.DataFrame) -> pd.DataFrame:
    """Join season N with the same player's season N+1."""
    nxt = ps[["shooterPlayerId", "season", "goals", "shots"]].copy()
    nxt = nxt.rename(columns={
        "season": "season_prev", "goals": "goals_next", "shots": "shots_next"})
    nxt["season"] = nxt["season_prev"] + 1
    nxt = nxt.drop(columns="season_prev")
    pairs = ps.merge(nxt, on=["shooterPlayerId", "season"], how="inner")
    pairs = pairs[(pairs["shots"] >= MIN_SHOTS_N) & (pairs["shots_next"] >= MIN_SHOTS_N1)]
    return pairs.reset_index(drop=True)


def forecast_metrics(name, pred, actual):
    """Out-of-sample fit of a single season-N predictor against season-(N+1) goals.

    R^2 / RMSE come from a pooled 1-variable OLS (predictor -> goals_next); this
    rewards a predictor that is both correlated AND on the right scale, while
    staying agnostic to each stat's units.
    """
    pred = np.asarray(pred, float)
    actual = np.asarray(actual, float)
    b1, b0 = np.polyfit(pred, actual, 1)
    fit = b0 + b1 * pred
    sse = np.sum((actual - fit) ** 2)
    sst = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - sse / sst
    rmse = np.sqrt(sse / len(actual))
    pear = np.corrcoef(pred, actual)[0, 1]
    spear = spearmanr(pred, actual).correlation
    return dict(predictor=name, n=len(actual), pearson_r=pear,
                spearman_r=spear, r2=r2, rmse=rmse)


def main():
    print("Loading MoneyPuck shots 2015-2023 ...")
    raw = mx.load_seasons()
    df = mx.build_features(raw)
    X, y, feats, groups = mx.feature_matrix(df)
    seasons = sorted(df["season"].unique())
    # Need >=1 prior season to train and a season N+1 to score the target.
    test_seasons = [s for s in seasons if s > seasons[0] and (s + 1) in seasons]
    print(f"  shots: {len(df):,}  | seasons: {seasons}")
    print(f"  walk-forward test seasons (train on < N): {test_seasons}")

    if os.path.exists(WF_CACHE):
        print(f"Loading cached walk-forward xG -> {os.path.abspath(WF_CACHE)}")
        df["xg_wf"] = pd.read_parquet(WF_CACHE, columns=["xg_wf"])["xg_wf"].values
    else:
        print("Walk-forward training (train on seasons < N, score season N) ...")
        df = walk_forward_xg(df, feats, test_seasons)
        df[["xg_wf"]].to_parquet(WF_CACHE, index=False)
        print(f"  cached walk-forward xG -> {os.path.abspath(WF_CACHE)}")

    ps = player_season_table(df)
    pairs = make_pairs(ps)
    print(f"\nPlayer-season pairs (N -> N+1, >={MIN_SHOTS_N} shots in N): {len(pairs):,}")

    # ---- Q1: which season-N predictor best forecasts season-(N+1) goals? ----
    rows = [
        forecast_metrics("prior goals (G_N)", pairs["goals"], pairs["goals_next"]),
        forecast_metrics("our xG (xG_N)", pairs["xg"], pairs["goals_next"]),
        forecast_metrics("MoneyPuck xG (xG_N)", pairs["xg_mp"], pairs["goals_next"]),
    ]
    models = pd.DataFrame(rows)
    print("\n=== Q1. Predicting next-season goals (pooled, out-of-sample) ===")
    print(models.round(4).to_string(index=False))
    models.to_csv(os.path.join(FIG_DIR, "predictive_validity_models.csv"), index=False)

    # ---- Q2: is finishing (G - xG) per shot repeatable year over year? ----
    # Need season N+1's xG too: re-pair on the full per-shot finishing rate.
    fin = ps.copy()
    fin["fin_ours"] = (fin["goals"] - fin["xg"]) / fin["shots"]
    fin["fin_mp"] = (fin["goals"] - fin["xg_mp"]) / fin["shots"]
    nxt = fin[["shooterPlayerId", "season", "fin_ours", "fin_mp", "shots"]].copy()
    nxt = nxt.rename(columns={"season": "s", "fin_ours": "fin_ours_next",
                              "fin_mp": "fin_mp_next", "shots": "shots_next"})
    nxt["season"] = nxt["s"] - 1
    fin_pairs = fin.merge(nxt.drop(columns="s"), on=["shooterPlayerId", "season"], how="inner")
    fin_pairs = fin_pairs[(fin_pairs["shots"] >= MIN_SHOTS_N) &
                          (fin_pairs["shots_next"] >= MIN_SHOTS_N)]

    fin_rows = []
    for label, a, b in [("our xG", "fin_ours", "fin_ours_next"),
                        ("MoneyPuck xG", "fin_mp", "fin_mp_next")]:
        r = np.corrcoef(fin_pairs[a], fin_pairs[b])[0, 1]
        rho = spearmanr(fin_pairs[a], fin_pairs[b]).correlation
        fin_rows.append(dict(xg_model=label, n_pairs=len(fin_pairs),
                             finishing_r=r, finishing_spearman=rho))
    fin_tbl = pd.DataFrame(fin_rows)
    print(f"\n=== Q2. Finishing (G-xG per shot) repeatability, N -> N+1 "
          f"(>={MIN_SHOTS_N} shots both seasons, n={len(fin_pairs)}) ===")
    print(fin_tbl.round(4).to_string(index=False))
    fin_tbl.to_csv(os.path.join(FIG_DIR, "predictive_validity_finishing.csv"), index=False)

    # Where does McDavid land on per-season finishing (our xG)?
    mcd = fin[fin["shooterPlayerId"] == mx.MCDAVID_ID].copy()
    if len(mcd):
        mcd["fin_per100"] = 100 * mcd["fin_ours"]
        # league percentile of his per-shot finishing among >=100-shot seasons
        pool = fin[fin["shots"] >= MIN_SHOTS_N]["fin_ours"]
        mcd["league_pctile"] = mcd["fin_ours"].apply(lambda v: 100 * (pool < v).mean())
        print("\n=== McDavid finishing by season (our xG, +goals per 100 shots / league pctile) ===")
        print(mcd[["season", "shots", "goals", "xg", "fin_per100", "league_pctile"]]
              .round(2).to_string(index=False))

    # ---- figure: forecast scatter + finishing repeatability ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(pairs["goals"], pairs["goals_next"], s=8, alpha=0.3,
               label=f"prior goals (r={models.iloc[0]['pearson_r']:.2f})")
    ax.scatter(pairs["xg"], pairs["goals_next"], s=8, alpha=0.3, color="C3",
               label=f"our xG (r={models.iloc[1]['pearson_r']:.2f})")
    ax.set_xlabel("Season-N predictor (goals)")
    ax.set_ylabel("Season-(N+1) actual goals")
    ax.set_title("Predicting next-season goals"); ax.legend()

    ax = axes[1]
    ax.scatter(100 * fin_pairs["fin_ours"], 100 * fin_pairs["fin_ours_next"],
               s=8, alpha=0.3)
    mcd_pairs = fin_pairs[fin_pairs["shooterPlayerId"] == mx.MCDAVID_ID]
    if len(mcd_pairs):
        ax.scatter(100 * mcd_pairs["fin_ours"], 100 * mcd_pairs["fin_ours_next"],
                   s=60, color="C1", edgecolor="k", zorder=5, label="McDavid")
        ax.legend()
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Finishing season N (G-xG per 100 shots)")
    ax.set_ylabel("Finishing season N+1 (per 100 shots)")
    ax.set_title(f"Finishing repeatability (r={fin_tbl.iloc[0]['finishing_r']:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "predictive_validity.png"), dpi=120)
    print(f"\nFigures + CSVs written to {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
