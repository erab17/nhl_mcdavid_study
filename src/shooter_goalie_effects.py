"""
Hierarchical shooter + goalie empirical-Bayes effects on top of the base xG
(next-step #2 from HANDOFF.md).

The base LightGBM xG is *shooter-blind by design* (so is MoneyPuck's xGoal) --
that is what makes goals-above-expected (G - xG) a clean finishing metric. This
script asks the complementary question: if we are willing to give up that purity,
how much does knowing *who shot* and *who is in net* improve the per-shot goal
probability, and does the resulting shooter-adjusted xG forecast next-season
goals better than the shooter-blind models?

Model -- a logistic random-intercept layered on the base xG:

    logit(p_adj_i) = logit(xg_base_i) + u_{shooter(i)} + v_{goalie(i)}

    u_s ~ N(0, tau_u^2)      shooter finishing offset (log-odds)
    v_g ~ N(0, tau_v^2)      goalie suppression offset (log-odds, <0 = good goalie)

The base xG enters as a fixed per-shot OFFSET (we do not refit it), so u and v
capture only what the shot-quality model leaves on the table. u_s, v_g and the
variance components tau^2 are fit by an EM-style empirical-Bayes scheme
(penalised/MAP Newton for the offsets, EM update for the variances). Shrinkage is
automatic and per-entity: a shooter with few prior shots is pulled hard toward 0;
McDavid, with thousands, barely. A shooter never seen in the prior window gets
exactly 0 (no adjustment), which falls out of the math for free.

LEAK-FREE, walk-forward (mirrors src/predictive_validity.py):
  * The base xG is the cached *prospective* walk-forward xG (trained on seasons
    < N, scoring season N) from data/scored_shots_walkforward.parquet -- NOT the
    all-seasons OOF, which would leak future seasons into a "season < N" prior.
  * For each test season N, u/v are estimated ONLY from shots in seasons < N
    (that have a walk-forward base xG, i.e. 2016..N-1), then applied to season N.
    A shot's own season never informs its own shooter/goalie effect.

INTERPRETATION WARNING (printed in the output, repeated here on purpose):
  Adding a shooter effect changes what the metric *is*. Base xG / MoneyPuck xG
  estimate CHANCE QUALITY (shooter-blind); the shooter-adjusted xG estimates
  EXPECTED GOALS FOR THIS SHOOTER. The latter is better for predicting goals, but
  G - xg_adj is no longer "finishing" -- finishing has been folded into the xG.
  Keep the base xG for finishing studies; use the adjusted xG for projection.

Run:  python src/shooter_goalie_effects.py
Outputs (figures/):
  shooter_goalie_shotlevel.csv     shot-level log loss / Brier / AUC
  shooter_goalie_predval.csv       next-season-goal forecast metrics
  shooter_effect_ranking.csv       EB shooter offsets, ranked, McDavid flagged
  goalie_effect_ranking.csv        EB goalie offsets, ranked
  shooter_goalie_effects.png       figure
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import expit, logit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

import modern_xg as mx
from predictive_validity import (
    WF_CACHE, MIN_SHOTS_N, MIN_SHOTS_N1, forecast_metrics,
)

warnings.filterwarnings("ignore", category=UserWarning)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

EPS = 1e-6
# Shot-count floor for *ranking/reporting* shooter & goalie effects (the EB fit
# itself uses every shot; this only filters who we list as a stable estimate).
RANK_MIN_SHOTS = 500


# --------------------------------------------------------------------------- #
# Empirical-Bayes two-way logistic random-intercept fit (offset = base logit)  #
# --------------------------------------------------------------------------- #
def fit_eb_two_way(z, goal, s_idx, n_s, g_idx, n_g,
                   n_em=30, n_newton=4, tau2_init=0.25, verbose=False):
    """Fit logit(p) = z + u[s_idx] + v[g_idx] with u~N(0,tau_u^2), v~N(0,tau_v^2).

    z      : per-shot base-xG log-odds (fixed offset)
    goal   : 0/1 outcomes
    s_idx  : contiguous shooter index per shot in [0, n_s)
    g_idx  : contiguous goalie index per shot in [0, n_g)
    Returns (u, info_u, tau_u2, v, info_v, tau_v2). info_* is the Fisher
    information sum_i p_i(1-p_i) per entity (precision of the raw estimate).
    """
    z = np.asarray(z, float)
    goal = np.asarray(goal, float)
    u = np.zeros(n_s)
    v = np.zeros(n_g)
    Gs = np.bincount(s_idx, weights=goal, minlength=n_s)   # goals per shooter
    Gg = np.bincount(g_idx, weights=goal, minlength=n_g)   # goals against goalie
    tau_u2 = tau_v2 = tau2_init

    for em in range(n_em):
        # --- update shooter offsets u given v (penalised Newton) ---
        for _ in range(n_newton):
            p = expit(z + u[s_idx] + v[g_idx])
            w = p * (1.0 - p)
            grad = np.bincount(s_idx, weights=p, minlength=n_s) - Gs + u / tau_u2
            hess = np.bincount(s_idx, weights=w, minlength=n_s) + 1.0 / tau_u2
            u -= grad / hess
        # --- update goalie offsets v given u ---
        for _ in range(n_newton):
            p = expit(z + u[s_idx] + v[g_idx])
            w = p * (1.0 - p)
            grad = np.bincount(g_idx, weights=p, minlength=n_g) - Gg + v / tau_v2
            hess = np.bincount(g_idx, weights=w, minlength=n_g) + 1.0 / tau_v2
            v -= grad / hess
        # --- EM update of the variance components ---
        p = expit(z + u[s_idx] + v[g_idx])
        w = p * (1.0 - p)
        Sw_u = np.bincount(s_idx, weights=w, minlength=n_s)
        Sw_v = np.bincount(g_idx, weights=w, minlength=n_g)
        post_var_u = 1.0 / (Sw_u + 1.0 / tau_u2)
        post_var_v = 1.0 / (Sw_v + 1.0 / tau_v2)
        tau_u2 = float(np.mean(u ** 2 + post_var_u))
        tau_v2 = float(np.mean(v ** 2 + post_var_v))
        if verbose and (em % 10 == 0 or em == n_em - 1):
            print(f"    EM {em:2d}: tau_u={np.sqrt(tau_u2):.4f} "
                  f"tau_v={np.sqrt(tau_v2):.4f}")

    p = expit(z + u[s_idx] + v[g_idx])
    w = p * (1.0 - p)
    info_u = np.bincount(s_idx, weights=w, minlength=n_s)
    info_v = np.bincount(g_idx, weights=w, minlength=n_g)
    return u, info_u, tau_u2, v, info_v, tau_v2


def _idx_and_offsets(df_pool, df_apply, id_col):
    """Factorise pool ids; return (idx_pool, mapper) where mapper(series)->idx for
    apply rows, with unseen ids -> -1 (sentinel for 'no effect')."""
    codes, uniques = pd.factorize(df_pool[id_col].values)
    lookup = {v: i for i, v in enumerate(uniques)}
    apply_idx = df_apply[id_col].map(lookup).fillna(-1).astype(int).values
    return codes, len(uniques), apply_idx, uniques


# --------------------------------------------------------------------------- #
# Data plumbing                                                                #
# --------------------------------------------------------------------------- #
def load_scored():
    """Rebuild the feature frame and attach the cached walk-forward base xG."""
    if not os.path.exists(WF_CACHE):
        raise FileNotFoundError(
            f"{WF_CACHE} not found -- run src/predictive_validity.py first to "
            "build the walk-forward base-xG cache.")
    raw = mx.load_seasons()
    df = mx.build_features(raw)
    df["xg_wf"] = pd.read_parquet(WF_CACHE, columns=["xg_wf"])["xg_wf"].values
    # Goalie id present for all non-empty-net shots; coerce to a clean int key.
    df["goalieId"] = df["goalieIdForShot"]
    return df


def apply_offsets(z, u, s_apply_idx, v=None, g_apply_idx=None):
    """logit base + shooter (and optionally goalie) offset -> adjusted prob.
    Unseen entities (idx == -1) contribute 0."""
    off = np.where(s_apply_idx >= 0, u[np.clip(s_apply_idx, 0, None)], 0.0)
    if v is not None:
        off = off + np.where(g_apply_idx >= 0, v[np.clip(g_apply_idx, 0, None)], 0.0)
    return expit(z + off)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("Loading shots + cached walk-forward base xG ...")
    df = load_scored()
    seasons = sorted(df["season"].unique())
    wf_seasons = sorted(df.loc[df["xg_wf"].notna(), "season"].unique())
    # Test seasons we can ADJUST: need >=1 prior season that itself has a
    # walk-forward base xG (2016..2022), so the prior pool is non-empty.
    test_seasons = [N for N in wf_seasons if any(s < N for s in wf_seasons)]
    print(f"  shots: {len(df):,} | seasons: {seasons}")
    print(f"  walk-forward base-xG seasons: {wf_seasons}")
    print(f"  adjustable test seasons (prior pool non-empty): {test_seasons}")

    # Per-shot result holder for the adjusted population (test seasons only).
    parts = []
    for N in test_seasons:
        pool = df[(df["season"] < N) & df["xg_wf"].notna() &
                  df["goalieId"].notna() & df["shooterPlayerId"].notna()].copy()
        appl = df[(df["season"] == N) & df["xg_wf"].notna()].copy()
        if len(pool) == 0 or len(appl) == 0:
            continue

        s_codes, n_s, s_apply, _ = _idx_and_offsets(pool, appl, "shooterPlayerId")
        g_codes, n_g, g_apply, _ = _idx_and_offsets(pool, appl, "goalieId")
        z_pool = logit(pool["xg_wf"].clip(EPS, 1 - EPS).values)
        u, iu, tu2, v, iv, tv2 = fit_eb_two_way(
            z_pool, pool["goal"].values, s_codes, n_s, g_codes, n_g)

        z_appl = logit(appl["xg_wf"].clip(EPS, 1 - EPS).values)
        appl["xg_shooter"] = apply_offsets(z_appl, u, s_apply)
        appl["xg_both"] = apply_offsets(z_appl, u, s_apply, v, g_apply)
        parts.append(appl[[
            "season", "isPlayoffGame", "shooterPlayerId", "shooterName",
            "goal", "xg_wf", "xGoal", "xg_shooter", "xg_both"]])
        print(f"  season {N}: pool {len(pool):,} shots / {n_s} shooters / {n_g} "
              f"goalies  ->  scored {len(appl):,}  "
              f"(tau_u={np.sqrt(tu2):.3f}, tau_v={np.sqrt(tv2):.3f})")

    adj = pd.concat(parts, ignore_index=True)

    # ---------------------------------------------------------------------- #
    # (a) SHOT-LEVEL: does the adjustment improve per-shot probabilities?     #
    # ---------------------------------------------------------------------- #
    y = adj["goal"].astype(int).values
    def ev(name, p):
        p = np.clip(p, EPS, 1 - EPS)
        return dict(model=name, n=len(y), log_loss=log_loss(y, p),
                    brier=brier_score_loss(y, p), auc=roc_auc_score(y, p))
    shot_rows = [
        ev("base xG (walk-forward)", adj["xg_wf"]),
        ev("+ shooter EB", adj["xg_shooter"]),
        ev("+ shooter & goalie EB", adj["xg_both"]),
        ev("MoneyPuck xGoal (benchmark)", adj["xGoal"]),
    ]
    shot_tbl = pd.DataFrame(shot_rows)
    print("\n=== (a) Shot-level metrics, adjustable test seasons "
          f"({adj['season'].min()}-{adj['season'].max()}, n={len(adj):,}) ===")
    print(shot_tbl.round(5).to_string(index=False))
    shot_tbl.to_csv(os.path.join(FIG_DIR, "shooter_goalie_shotlevel.csv"), index=False)

    # ---------------------------------------------------------------------- #
    # (b) PREDICTIVE VALIDITY: forecast next-season goals (reuse harness)     #
    # ---------------------------------------------------------------------- #
    rs = adj[adj["isPlayoffGame"] == 0]
    ps = rs.groupby(["shooterPlayerId", "shooterName", "season"]).agg(
        shots=("goal", "size"), goals=("goal", "sum"),
        xg=("xg_wf", "sum"), xg_mp=("xGoal", "sum"),
        xg_adj=("xg_shooter", "sum"),
    ).reset_index()
    nxt = ps[["shooterPlayerId", "season", "goals", "shots"]].rename(
        columns={"season": "sp", "goals": "goals_next", "shots": "shots_next"})
    nxt["season"] = nxt["sp"] - 1
    pairs = ps.merge(nxt.drop(columns="sp"), on=["shooterPlayerId", "season"], how="inner")
    pairs = pairs[(pairs["shots"] >= MIN_SHOTS_N) &
                  (pairs["shots_next"] >= MIN_SHOTS_N1)].reset_index(drop=True)

    pv_rows = [
        forecast_metrics("prior goals (G_N)", pairs["goals"], pairs["goals_next"]),
        forecast_metrics("base xG", pairs["xg"], pairs["goals_next"]),
        forecast_metrics("MoneyPuck xG", pairs["xg_mp"], pairs["goals_next"]),
        forecast_metrics("shooter-adjusted xG", pairs["xg_adj"], pairs["goals_next"]),
    ]
    pv_tbl = pd.DataFrame(pv_rows)
    print(f"\n=== (b) Predicting next-season goals (pooled OOS, n={len(pairs):,} "
          "player-season pairs) ===")
    print(pv_tbl.round(4).to_string(index=False))
    pv_tbl.to_csv(os.path.join(FIG_DIR, "shooter_goalie_predval.csv"), index=False)

    # ---------------------------------------------------------------------- #
    # McDavid's estimated shooter effect + league ranking (descriptive fit)   #
    # ---------------------------------------------------------------------- #
    # One stable fit over ALL walk-forward-scored seasons (2016-2022) -- this is
    # a descriptive estimate of each player's finishing offset, NOT used for the
    # leak-free prediction above.
    full = df[df["xg_wf"].notna() & df["goalieId"].notna() &
              df["shooterPlayerId"].notna()].copy()
    s_codes, n_s, _, s_uniq = _idx_and_offsets(full, full, "shooterPlayerId")
    g_codes, n_g, _, g_uniq = _idx_and_offsets(full, full, "goalieId")
    z_full = logit(full["xg_wf"].clip(EPS, 1 - EPS).values)
    print("\nFitting descriptive full-sample EB effects (2016-2022) ...")
    u, iu, tu2, v, iv, tv2 = fit_eb_two_way(
        z_full, full["goal"].values, s_codes, n_s, g_codes, n_g, verbose=True)

    # interpretable units: extra goals per 100 shots at the league-mean base xG.
    p0 = float(full["xg_wf"].mean())
    z0 = logit(p0)
    def per100(off):
        return 100.0 * (expit(z0 + off) - p0)

    sh = full.groupby("shooterPlayerId").agg(
        name=("shooterName", "first"), shots=("goal", "size"),
        goals=("goal", "sum"), xg=("xg_wf", "sum")).reset_index()
    sh = sh.merge(pd.DataFrame({"shooterPlayerId": s_uniq, "u": u, "info": iu}),
                  on="shooterPlayerId", how="left")
    sh["goals_per100_vs_base"] = per100(sh["u"])
    rank_pool = sh[sh["shots"] >= RANK_MIN_SHOTS].copy()
    rank_pool["rank"] = rank_pool["u"].rank(ascending=False).astype(int)
    rank_pool["pctile"] = 100 * rank_pool["u"].rank(pct=True)
    rank_pool = rank_pool.sort_values("u", ascending=False).reset_index(drop=True)
    rank_pool.to_csv(os.path.join(FIG_DIR, "shooter_effect_ranking.csv"), index=False)

    gl = full.groupby("goalieId").agg(
        name=("goalieNameForShot", "first"), shots=("goal", "size"),
        goals=("goal", "sum"), xg=("xg_wf", "sum")).reset_index()
    gl = gl.merge(pd.DataFrame({"goalieId": g_uniq, "v": v, "info": iv}),
                  on="goalieId", how="left")
    gl["goals_per100_vs_base"] = per100(gl["v"])
    gl_rank = gl[gl["shots"] >= RANK_MIN_SHOTS].sort_values("v").reset_index(drop=True)
    gl_rank.to_csv(os.path.join(FIG_DIR, "goalie_effect_ranking.csv"), index=False)

    n_ranked = len(rank_pool)
    mcd = rank_pool[rank_pool["shooterPlayerId"] == mx.MCDAVID_ID]
    print(f"\n=== Shooter effect: top 10 (of {n_ranked} shooters >= {RANK_MIN_SHOTS} "
          "shots, 2016-2022) ===")
    print(rank_pool.head(10)[["rank", "name", "shots", "goals", "xg", "u",
                              "goals_per100_vs_base", "pctile"]]
          .round(3).to_string(index=False))
    if len(mcd):
        m = mcd.iloc[0]
        print(f"\n--- Connor McDavid ---")
        print(f"  shooter offset u = {m['u']:+.4f} log-odds  "
              f"(+{m['goals_per100_vs_base']:.2f} goals per 100 shots vs base xG)")
        print(f"  rank {int(m['rank'])} / {n_ranked}  "
              f"(league {m['pctile']:.1f}th percentile of finishing skill)")
    print(f"\n  tau_u={np.sqrt(tu2):.3f} (shooter) vs tau_v={np.sqrt(tv2):.3f} "
          "(goalie) -- shooter spread is larger than goalie spread.")
    print(f"\n=== Goalie effect: best 5 (most suppression, of {len(gl_rank)}) ===")
    print(gl_rank.head(5)[["name", "shots", "goals", "xg", "v",
                           "goals_per100_vs_base"]].round(3).to_string(index=False))

    print("\nINTERPRETATION: the shooter-adjusted xG is no longer a pure "
          "chance-quality metric -- it folds the shooter's finishing skill INTO "
          "the expectation. Use base xG for finishing studies (G - xG), the "
          "adjusted xG for goal projection.")

    # ---------------------------------------------------------------------- #
    # Figure                                                                  #
    # ---------------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (1) shot-level log loss bars
    ax = axes[0]
    m = shot_tbl.set_index("model")["log_loss"]
    colors = ["C0", "C3", "C1", "C2"]
    ax.bar(range(len(m)), m.values, color=colors)
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(["base\nxG", "+shooter", "+shooter\n&goalie", "MoneyPuck"],
                       fontsize=9)
    ax.set_ylabel("log loss (lower better)")
    ax.set_ylim(m.min() - 0.0006, m.max() + 0.0006)
    for i, val in enumerate(m.values):
        ax.text(i, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) Shot-level fit\n(gains are real but tiny)")

    # (2) shooter EB effect distribution with McDavid flagged
    ax = axes[1]
    ax.hist(rank_pool["goals_per100_vs_base"], bins=40, color="C0", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8)
    if len(mcd):
        xv = mcd.iloc[0]["goals_per100_vs_base"]
        ax.axvline(xv, color="C1", lw=2,
                   label=f"McDavid +{xv:.2f}/100\n(rank {int(mcd.iloc[0]['rank'])}/{n_ranked})")
        ax.legend(fontsize=9)
    ax.set_xlabel("shooter EB effect (extra goals / 100 shots vs base xG)")
    ax.set_ylabel(f"# shooters (>= {RANK_MIN_SHOTS} shots)")
    ax.set_title("(b) Finishing skill is real and spread out")

    # (3) predictive validity: r by predictor
    ax = axes[2]
    order = pv_tbl.sort_values("pearson_r")
    cmap = {"prior goals (G_N)": "C7", "base xG": "C0",
            "MoneyPuck xG": "C2", "shooter-adjusted xG": "C3"}
    ax.barh(range(len(order)), order["pearson_r"],
            color=[cmap[k] for k in order["predictor"]])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["predictor"], fontsize=9)
    ax.set_xlim(0.66, order["pearson_r"].max() + 0.01)
    for i, val in enumerate(order["pearson_r"]):
        ax.text(val, i, f" {val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Pearson r vs next-season goals")
    ax.set_title("(c) Predictive validity\n(metric that matters)")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shooter_goalie_effects.png"), dpi=120)
    print(f"\nFigures + CSVs written to {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
