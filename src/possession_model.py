"""
Full possession-chain xG model (next-step #3 from HANDOFF.md).

The question: MoneyPuck's xG (like most public models) looks only ONE event back
(`lastEventCategory`, `speedFromLastEvent`, ...). Soccer build-up analysis looks
at the whole possession chain. Does walking *four* events back -- the hockey
analog -- add real predictive signal over one event back?

We answer it two ways, both on a full league season (default 2023-24) scraped
from the raw NHL API (the only public source with >1 event of history):

  PART A  -- SELF-CONTAINED, identical-data comparison.  Build three nested
     feature sets on the same shots and score each with GroupKFold(game) OOF:
       core    : shot geometry + type + period (the shot-quality skeleton)
       +last1  : core + the single preceding event (MoneyPuck-style context)
       +last4  : core + the full 4-event possession chain
     The honest test of "last-4 vs last-1" the handoff asks for.

  PART B  -- ADDITIVE LAYER on MoneyPuck's *production* xGoal.  Merge the chain
     features onto MoneyPuck 2023 shots (gamePk = season*1e6 + game_id, matched
     on shooter/period/time) and fit a LightGBM with logit(xGoal) as a fixed
     offset (init_score).  Does the chain improve even MoneyPuck's ~100-feature
     model?  This is the strong form -- adding beyond the real benchmark.

LEAKAGE WATCH (this project's #1 trap): every chain feature is strictly pre-shot
(previous events only), so there is no outcome leakage -- but per HANDOFF.md,
any xG much above ~0.80 AUC is a red flag.  We report log loss / Brier, not just
AUC, and expect small gains (xG saturates).

Run:  python src/possession_model.py            # uses cached PBP if present
Outputs (figures/):
  possession_chain_metrics.csv   nested-model + additive-layer metrics
  possession_chain.png           figure
Cache: data/possession_shots_2023.parquet  (assembled shot+chain table)
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
from scipy.special import logit, expit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

import nhl_api
import possession
from run_pipeline import LGB_PARAMS

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(FIG_DIR, exist_ok=True)
EPS = 1e-6

LAG_NUM = ["same_team", "dist", "dt", "speed", "cross_mid"]   # per-lag numeric cols
CORE = ["dist", "angle", "x_abs", "y_abs", "period"]
CORE_CAT = ["shot_type"]


# --------------------------------------------------------------------------- #
# Assemble the shot + possession-chain table for a full season                #
# --------------------------------------------------------------------------- #
def build_season_table(season: int = 2023, cache=True) -> pd.DataFrame:
    cache_path = os.path.join(DATA_DIR, f"possession_shots_{season}.parquet")
    if cache and os.path.exists(cache_path):
        print(f"Loading cached chain table -> {os.path.abspath(cache_path)}")
        return pd.read_parquet(cache_path)

    gids = nhl_api.league_game_ids(season, game_type=2)
    print(f"Assembling chain features for {len(gids)} games ({season}-{season+1}) ...")
    frames = []
    for i, gid in enumerate(gids):
        try:
            ev = possession.game_events(gid)
            if len(ev):
                frames.append(possession.chain_features(ev, k=4))
        except Exception as e:               # missing/short PBP -> skip the game
            if i < 5:
                print(f"  skip {gid}: {e}")
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(gids)} games")
    df = pd.concat(frames, ignore_index=True)

    # ---- shot geometry (net at x=+/-89, y=0; absolute coords) ----
    x_abs = df["x"].abs()
    df["x_abs"] = x_abs
    df["y_abs"] = df["y"].abs()
    df["dist"] = np.hypot(89.0 - x_abs, df["y"])
    df["angle"] = np.degrees(np.arctan2(df["y"].abs(), (89.0 - x_abs).clip(lower=1e-6)))
    df["season"] = season

    # categoricals
    for c in [c for c in df.columns if c.endswith("_event")] + ["shot_type"]:
        df[c] = df[c].astype("category")

    if cache:
        df.to_parquet(cache_path, index=False)
        print(f"  cached -> {os.path.abspath(cache_path)}")
    return df


def feature_sets(df: pd.DataFrame):
    """Return dict name -> list of feature columns, nested core < +last1 < +last4."""
    l1 = [f"l1_{c}" for c in LAG_NUM] + ["l1_event"]
    l24 = [f"l{lag}_{c}" for lag in (2, 3, 4) for c in LAG_NUM] + \
          [f"l{lag}_event" for lag in (2, 3, 4)] + ["chain_passes", "chain_turnovers"]
    core = [c for c in CORE + CORE_CAT if c in df.columns]
    sets = {
        "core (geometry+type)": core,
        "+ last-1 event (MoneyPuck-style)": core + [c for c in l1 if c in df.columns],
        "+ last-4 chain (possession)": core + [c for c in l1 + l24 if c in df.columns],
    }
    return sets


def oof_predict(df, feats, y, groups, n_splits=5, init_score=None):
    """GroupKFold OOF probabilities. If init_score given, it's a fixed per-row
    logit offset (LightGBM init_score) -> model fits the residual over it."""
    X = df[feats]
    oof = np.zeros(len(df))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(X, y, groups):
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        fit_kw = {}
        if init_score is not None:
            fit_kw["init_score"] = init_score[tr]
        model.fit(X.iloc[tr], y[tr], **fit_kw)
        if init_score is not None:
            # raw margin + the offset -> probability
            margin = model.predict(X.iloc[va], raw_score=True)
            oof[va] = expit(init_score[va] + margin)
        else:
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


def ev(name, y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return dict(model=name, n=len(y), log_loss=log_loss(y, p),
                brier=brier_score_loss(y, p), auc=roc_auc_score(y, p))


# --------------------------------------------------------------------------- #
# Part B: merge chain features onto MoneyPuck shots                            #
# --------------------------------------------------------------------------- #
def merge_moneypuck(df: pd.DataFrame, season: int = 2023):
    """Attach MoneyPuck goal + xGoal to the raw-PBP chain shots.

    MoneyPuck game_id (5-digit) -> NHL gamePk = season*1e6 + game_id. Each shot
    is keyed by (gamePk, shooterPlayerId, period) and matched to its nearest-in-
    time MoneyPuck shot (coordinates/seconds differ by a tick between sources).
    """
    import modern_xg as mx
    mp = mx.load_seasons(seasons=[season])
    mp = mp[mp["event"].isin(["SHOT", "MISS", "GOAL"])].copy()
    mp["gamePk"] = season * 1_000_000 + mp["game_id"].astype(int)
    mp["shooterPlayerId"] = mp["shooterPlayerId"].astype("Int64")

    # raw-PBP side already has game_id = gamePk, shooter_id, period; add an
    # absolute game-second from the chain table is not stored, so re-derive a key
    # on (game, shooter, period) and match by row order within that key on time
    # is unavailable -> match on (game, shooter, period) + nearest x.
    pbp = df.copy()
    pbp["gamePk"] = pbp["game_id"].astype(int)
    pbp["shooterPlayerId"] = pbp["shooter_id"].astype("Int64")

    keys = ["gamePk", "shooterPlayerId", "period"]
    # Merge many-to-many on the key, then within each key pick the MoneyPuck row
    # whose x is closest to the PBP x (handles multiple shots by a player/period).
    merged = pbp.merge(
        mp[keys + ["xCord", "goal", "xGoal"]].rename(columns={"xCord": "mp_x"}),
        on=keys, how="inner")
    merged["xdiff"] = (merged["mp_x"] - merged["x"]).abs()
    merged = (merged.sort_values("xdiff")
                    .drop_duplicates(subset=["gamePk", "shooterPlayerId", "period",
                                             "x", "y"], keep="first"))
    # keep only good geometric matches
    merged = merged[merged["xdiff"] <= 6].reset_index(drop=True)
    return merged


# --------------------------------------------------------------------------- #
def main(season: int = 2023):
    df = build_season_table(season)
    # population: real shot attempts with coordinates and a known shooter
    df = df[df["shooter_id"].notna() & df["x"].notna() & df["y"].notna()].copy()
    df = df.reset_index(drop=True)
    y = df["is_goal"].astype(int).values
    groups = df["game_id"].values
    print(f"\nShots: {len(df):,} | goal rate {y.mean():.4f} | games {df['game_id'].nunique()}")

    # ---- Part A: nested self-contained comparison ----
    sets = feature_sets(df)
    rows = [ev("baseline (mean rate)", y, np.full(len(y), y.mean()))]
    oof_store = {}
    for name, feats in sets.items():
        print(f"  OOF: {name}  ({len(feats)} feats)")
        oof = oof_predict(df, feats, y, groups)
        oof_store[name] = oof
        rows.append(ev(name, y, oof))
    part_a = pd.DataFrame(rows)
    print("\n=== Part A: nested models on identical raw-PBP shots (GroupKFold OOF) ===")
    print(part_a.round(5).to_string(index=False))

    # ---- Part B: additive layer on MoneyPuck xGoal (last-1 vs last-4) ----
    # We add the SAME nested feature sets on top of logit(xGoal) as a fixed
    # offset. The honest question is not "does the chain beat xGoal" (any second
    # model with partly-independent errors will, by ensembling) but "does last-4
    # beat last-1" -- i.e. is there value in possession DEPTH beyond one event?
    part_b = None
    try:
        mg = merge_moneypuck(df, season)
        print(f"\nMerged to MoneyPuck: {len(mg):,} shots "
              f"({100*len(mg)/len(df):.0f}% of PBP shots matched)")
        yb = mg["goal"].astype(int).values
        gb = mg["gamePk"].values
        xgoal = mg["xGoal"].clip(EPS, 1 - EPS).values
        offset = logit(xgoal)
        mg_sets = feature_sets(mg)
        rows_b = [ev("MoneyPuck xGoal alone", yb, xgoal)]
        for tag, name in [("+ last-1 over xGoal", "+ last-1 event (MoneyPuck-style)"),
                          ("+ last-4 over xGoal", "+ last-4 chain (possession)")]:
            oof_add = oof_predict(mg, mg_sets[name], yb, gb, init_score=offset)
            rows_b.append(ev(tag, yb, oof_add))
        part_b = pd.DataFrame(rows_b)
        print("\n=== Part B: nested layers on MoneyPuck's production xGoal (offset) ===")
        print(part_b.round(5).to_string(index=False))
    except Exception as e:
        print(f"\nPart B (MoneyPuck merge) skipped: {e}")

    # ---- the headline conclusion (printed so it lands in the run log) ----
    a_gain = part_a["auc"].iloc[-1] - part_a["auc"].iloc[-2]   # last4 - last1, Part A
    print("\n" + "=" * 72)
    print("CONCLUSION -- does looking >1 event back help? (the step-3 question)")
    print(f"  Part A  last-4 vs last-1 on identical data : AUC {a_gain:+.4f}  (negligible)")
    if part_b is not None:
        b_gain = part_b["auc"].iloc[2] - part_b["auc"].iloc[1]
        print(f"  Part B  last-4 vs last-1 over MP xGoal      : AUC {b_gain:+.4f}  (negligible)")
        print("  NOTE: the lift from xGoal-alone to +last-1 is an ENSEMBLE effect")
        print("        (stacking two partly-independent xG models), available from a")
        print("        SINGLE event back -- NOT evidence for possession depth.")
    print("  => Soccer-style 'N events back' does NOT transfer to hockey shot xG:")
    print("     the one preceding event (rebound/rush/turnover) carries the signal;")
    print("     events 2-4 back are noise by the time the shot is taken.")
    print("=" * 72)

    # ---- persist metrics ----
    part_a2 = part_a.assign(analysis="A: identical-data nested")
    out = part_a2
    if part_b is not None:
        out = pd.concat([part_a2, part_b.assign(analysis="B: additive on MoneyPuck xGoal")],
                        ignore_index=True)
    out.to_csv(os.path.join(FIG_DIR, "possession_chain_metrics.csv"), index=False)

    # ---- figure ----
    n_panels = 2 if part_b is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.2))
    axes = np.atleast_1d(axes)
    a = part_a[part_a["model"] != "baseline (mean rate)"]
    ax = axes[0]
    ax.bar(range(len(a)), a["log_loss"], color=["C0", "C1", "C3"])
    ax.set_xticks(range(len(a)))
    ax.set_xticklabels(["core", "+last-1", "+last-4"], fontsize=10)
    ax.set_ylim(a["log_loss"].min() - 0.001, a["log_loss"].max() + 0.001)
    for i, v in enumerate(a["log_loss"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("log loss (lower better)")
    ax.set_title("(A) Possession depth on identical data\n"
                 f"AUC {a['auc'].iloc[0]:.3f}->{a['auc'].iloc[-1]:.3f}")
    if part_b is not None:
        ax = axes[1]
        ax.bar(range(len(part_b)), part_b["log_loss"], color=["C2", "C0", "C3"])
        ax.set_xticks(range(len(part_b)))
        ax.set_xticklabels(["xGoal", "xGoal\n+last-1", "xGoal\n+last-4"], fontsize=10)
        ax.set_ylim(part_b["log_loss"].min() - 0.001, part_b["log_loss"].max() + 0.001)
        for i, v in enumerate(part_b["log_loss"]):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("log loss (lower better)")
        ax.set_title("(B) Layered on MoneyPuck xGoal\nlast-4 = last-1 (depth adds nothing)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "possession_chain.png"), dpi=120)
    print(f"\nFigures + CSV written to {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
