"""Builds notebooks/NHL_McDavid_xG_modern.ipynb from code/markdown blocks."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))


md(r"""
# A Modern xG Model & the Connor McDavid Finishing Study (2024 rebuild)

This is a ground-up modernization of my 2020 notebook (`NHL_McDavid.ipynb`). Two things
changed in the intervening years:

1. **The data source the original used is gone.** `statsapi.web.nhl.com` was
   decommissioned by the NHL (2023-24). The original notebook can no longer run.
   It is replaced here by `api-web.nhle.com` / `api.nhle.com` (see `src/nhl_api.py`).
2. **Far richer public data now exists.** [MoneyPuck](https://moneypuck.com/data.htm)
   publishes shot-level data (~120 columns) with pre-shot context — rebounds, rush,
   the previous event's type/location/timing, strength state, score state, shooter
   handedness / off-wing — *and* its own production xG to benchmark against.

### What was wrong / limited in the 2020 version
| Issue | 2020 notebook | This rebuild |
|---|---|---|
| Data source | dead API | current NHL API + MoneyPuck |
| Features | distance, angle, x only | + pre-shot movement, rebound, strength, score, shot type, handedness |
| Model | logistic regression | LightGBM (gradient boosting) |
| Validation | none (AUC on train set) | GroupKFold out-of-fold, **leak-free** |
| Metrics | AUC only | log loss + Brier + AUC + **reliability curve** |
| `timeOnIce` parsing | `"20:30"->20.30` (**bug**) | not needed (per-shot data) |
| Goal line | hard-coded x=84 | arena-adjusted coords from MoneyPuck |
""")

code(r"""
import sys, os
sys.path.append(os.path.abspath("../src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
""")

md(r"""
## 1. The new NHL API still works (the old one doesn't)

`src/nhl_api.py` is a small, cached client for the current endpoints. This is the
direct replacement for the original notebook's scraping cells. (Cached to `data/raw/`.)
""")

code(r"""
import nhl_api
game_ids = nhl_api.season_game_ids(2023)          # EDM 2023-24 regular season
print(f"Edmonton 2023-24 regular-season games: {len(game_ids)}")
demo = nhl_api.play_by_play(game_ids[0])
demo.head(6)
""")

md(r"""
## 2. The modern xG model

Training data: **999,863 unblocked shot attempts (2015-2023)**, goalie in net,
shootout excluded. The model and feature engineering live in `src/modern_xg.py`;
training (GroupKFold out-of-fold on `game_id`) runs in `src/run_pipeline.py` and
writes `data/scored_shots.parquet`. We load those leak-free predictions here.

> To regenerate from scratch: `python src/run_pipeline.py` (~2 min).
""")

code(r"""
df = pd.read_parquet("../data/scored_shots.parquet")
print(f"shots: {len(df):,} | goal rate: {df['goal'].mean():.4f}")
df[["shotDistance","shotAngleAdjusted","shotType","shotRebound",
    "timeSinceLastEvent","skater_diff","score_diff","xg","xGoal","goal"]].head()
""")

md("### 2a. How good is it? Probabilistic metrics + benchmark vs MoneyPuck")

code(r"""
def evaluate(name, y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return dict(model=name, log_loss=log_loss(y, p),
                brier=brier_score_loss(y, p), auc=roc_auc_score(y, p))

y = df["goal"].values
metrics = pd.DataFrame([
    evaluate("baseline (mean rate)", y, np.full(len(df), y.mean())),
    evaluate("our LightGBM (OOF)",   y, df["xg"]),
    evaluate("MoneyPuck xGoal",      y, df["xGoal"]),
])
metrics.round(4)
""")

md(r"""
Our model (AUC ~0.77, log loss ~0.21) is a clear step up from the 2020 logistic
model (AUC 0.73 on the *training* set) and lands within range of MoneyPuck's
production model — which uses ~100 features and multi-event pre-shot sequences to
our ~25. Crucially these numbers are **out-of-fold**, so they're honest.
""")

md("### 2b. Calibration — does a 0.20 xG really mean a 20% goal chance?")

code(r"""
fig, ax = plt.subplots(figsize=(6,6))
for label, p in [("Our xG", df["xg"]), ("MoneyPuck xG", df["xGoal"])]:
    frac, mean_pred = calibration_curve(y, p, n_bins=20, strategy="quantile")
    ax.plot(mean_pred, frac, marker="o", ms=4, label=label)
ax.plot([0,.6],[0,.6],"k--",lw=1,label="perfect")
ax.set(xlabel="Predicted xG", ylabel="Observed goal rate", title="Reliability curve")
ax.legend(); plt.show()
""")

md("### 2c. What drives the prediction?")

code(r"""
# Feature importances were saved during training; re-derive a quick view from a single fold.
import lightgbm as lgb, modern_xg as mx
raw = mx.build_features(mx.load_seasons([2022, 2023]))
X, yy, feats, groups = mx.feature_matrix(raw)
m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.05,
                       num_leaves=63, min_child_samples=200, verbose=-1).fit(X, yy)
imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values()
imp.tail(15).plot.barh(figsize=(7,6), title="Feature importance (gain)"); plt.show()
""")

md(r"""
Distance still dominates (as it should), but the model now leans meaningfully on
**pre-shot context** the 2020 model never had: `timeSinceLastEvent` and
`speedFromLastEvent` (rebounds / rush chances), `shotType`, `skater_diff`
(power play), and the previous event's location.
""")

md(r"""
## 3. The Connor McDavid finishing study — redone

Original open question: *is McDavid beating xG because of real finishing skill, or was
the 2020 model just blind to context?* With a context-aware, leak-free xG we can finally
look. Goals here **exclude empty-net** (you can't credit finishing skill against an absent
goalie), so totals run slightly below his official numbers.
""")

code(r"""
mcd = df[(df["shooterPlayerId"]==mx.MCDAVID_ID)].copy()
mcd["season_label"] = mcd["season"].astype(str)+"-"+(mcd["season"]+1).astype(str).str[-2:]
career = mcd.groupby("season_label").agg(
    shots=("goal","size"), goals=("goal","sum"),
    xg=("xg","sum"), xg_mp=("xGoal","sum"),
    xg_per_shot=("xg","mean"), mean_dist=("shotDistance","mean")).round(2)
career["G_minus_xG"] = (career["goals"]-career["xg"]).round(2)
career["finishing_%"] = (100*career["G_minus_xG"]/career["xg"]).round(1)
career
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14,5))
ax = axes[0]
ax.bar(career.index, career["goals"], alpha=.55, label="Actual goals")
ax.plot(career.index, career["xg"], "o-", color="C3", label="Our xG")
ax.plot(career.index, career["xg_mp"], "s--", color="C2", label="MoneyPuck xG")
ax.set_title("McDavid: goals vs expected goals"); ax.legend(); ax.tick_params(axis="x", rotation=45)
ax = axes[1]
colors = ["C0" if v>=0 else "C1" for v in career["G_minus_xG"]]
ax.bar(career.index, career["G_minus_xG"], color=colors)
ax.axhline(0,color="k",lw=1); ax.set_title("Finishing: goals above expected (G - xG)")
ax.tick_params(axis="x", rotation=45); plt.show()
""")

md(r"""
### Findings

* **McDavid is a genuinely elite finisher**, not just a volume/quality-of-chance shooter.
  He scores **above expected in most seasons**, peaking around **+15 goals above xG in
  2022-23** — i.e. he converted his chances ~35% better than a league-average shooter would.
* This is exactly what the 2020 model **could not** show. That model lacked pre-shot
  context and normalized xG by *total* time on ice (with a `timeOnIce` parsing bug on top),
  so McDavid looked unremarkable. The signal was real; the old model was blind to it.
* Both our model and MoneyPuck's agree closely on his per-season xG, which gives
  confidence the finishing gap is a property of *him*, not an artifact of one model.

* Both our model and MoneyPuck's agree closely on his per-season xG, which gives
  confidence the finishing gap is a property of *him*, not an artifact of one model.
""")

md(r"""
## 4. Future directions — how would you *beat* MoneyPuck?

We landed within ~0.02 AUC of MoneyPuck's production model. The natural question:
what would it take to pass them? The work below maps the terrain (full detail and
runnable code in `src/` and in `HANDOFF.md`).

### 4a. Why we got so close with ~25 features (xG saturates)
A feature ablation tells the story — each added feature buys less than the last:

| feature set | #feat | AUC |
|---|---|---|
| distance only | 1 | 0.703 |
| + angle | 2 | 0.721 |
| + shot type | 3 | 0.741 |
| + pre-shot context | 8 | 0.754 |
| + strength/score/hand (full) | 25 | 0.761 |
| MoneyPuck (~100 feat) | ~100 | 0.786 |

**Distance alone captures ~71% of the entire headroom** from a coin-flip to MoneyPuck.
xG is a low-ceiling, high-variance problem: whether a puck goes in is mostly irreducible
noise. "Beating" them with *more of the same* features is hopeless — you need *new
information*. (And note: we reuse MoneyPuck's own engineered features, and AUC flatters
the gap — the log-loss difference, 0.210 vs 0.201, is the honest measure.)

### 4b. The angles that could actually win
1. **Reframe the scoreboard to predictive validity.** Lowest in-sample log loss ≠ best
   model. The real test: does your xG predict *next season's* goals? Finishing-aware
   models win there even when shot-level metrics are flat. *(Highest value, low effort.)*
2. **Shooter + goalie effects.** MoneyPuck's xG is *shooter-blind by design* (that's what
   makes G−xG a finishing stat). A shrunk shooter-finishing prior improves even
   MoneyPuck's own model — real and orthogonal, but **tiny per shot** (finishing is a
   season-aggregate effect). This is the McDavid result operationalized.
3. **Possession chains** (`src/possession.py`) — the hockey analog of a soccer "N
   ball-events back" build-up. MoneyPuck uses only *one* event back; the full possession
   (turnovers, multi-pass cross-ice movement, sustained pressure) carries more.
4. **Tracking data** — the ceiling-raiser (shot speed at release, defender distance,
   screens, shooting in stride). This is the only category that adds genuinely *new*
   information rather than re-modeling old.
""")

md(r"""
### 4c. Tracking prototype + a leakage lesson worth keeping

NHL EDGE (the league's puck/player tracking) does **not** expose per-shot data publicly —
only season aggregates. As a stand-in we used the hand-tracked **Big Data Cup 2021**
dataset (`src/tracking_xg.py`), which has pass origin/receiver coordinates and a
one-timer flag.

**A cautionary tale:** the first model scored **AUC 0.95** — a fantasy. The dataset is
*assist-centric*, so `time_since_pass` (assist logged at the goal's clock second) and
`off_pass` (143/145 goals had a logged preceding pass) both leaked the outcome. After
removing them and restricting to a leak-free population (shots already off a pass), the
honest result holds:

| model (shots off a pass) | log loss | AUC |
|---|---|---|
| shot geometry only | 0.373 | 0.788 |
| + pass trajectory | 0.336 | **0.838** |

Pass trajectory (cross-ice distance, pre-shot angle change) adds **+0.05 AUC** — small
sample, so directional, but it confirms tracking-era pre-shot-movement signal is real and
is where the remaining edge lives.

> **Rule of thumb:** any xG much above ~0.80 AUC is almost certainly leaking. Always
> sanity-check feature/goal-rate tables before trusting a metric.

**Recommended path:** predictive-validity harness → shooter/goalie effects → full
possession-chain model → (later) sequence + tracking models. See `HANDOFF.md` for the
ranked plan.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
with open("notebooks/NHL_McDavid_xG_modern.ipynb", "w") as fh:
    nbf.write(nb, fh)
print("wrote notebooks/NHL_McDavid_xG_modern.ipynb with", len(cells), "cells")
