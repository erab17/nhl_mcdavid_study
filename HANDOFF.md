# Project handoff — NHL xG & the Connor McDavid finishing study

> Written for a future agent (or human) picking this up cold. Read this first, then
> `README.md`, then the code. It covers **what** was done, **why**, **how**, **when**,
> the **learnings** (especially the traps), and **ranked next steps**.

---

## 0. TL;DR

The repo started as a 2020-era notebook (`NHL_McDavid.ipynb`) that built a logistic
Expected-Goals (xG) model from the old NHL API and studied Connor McDavid. That notebook
**no longer runs** (its data API was decommissioned). In this rebuild (2026-06) we:

1. Migrated the data layer to the **current** NHL API + **MoneyPuck** shot data.
2. Built a **modern LightGBM xG** model (leak-free, calibrated): **AUC 0.771, log loss
   0.210**, vs MoneyPuck's production **0.794 / 0.201** — close, with ~25 features vs ~100.
3. Re-did the **McDavid study**: with context-aware xG he is a genuine **above-expected
   finisher**, peaking at **+15.3 goals over xG in 2022-23**.
4. Explored **how to beat MoneyPuck**: shooter/goalie effects, possession chains,
   tracking data — and quantified each.
5. Built a **possession-chain** extractor (your soccer "N events back" idea, in hockey).
6. Investigated **NHL EDGE** tracking (not freely available per-shot) and prototyped
   **tracking-based xG** on the Big Data Cup dataset — where we caught and fixed two
   layers of **target leakage** before landing a clean **+0.05 AUC** from pass trajectory.

---

## 1. Timeline / context (the "when")

- **~2020-2021 (original):** `NHL_McDavid.ipynb` — logistic regression on shot
  (distance, angle, x) from `statsapi.web.nhl.com`; AUC ~0.73 on the training set.
  Author's own README flagged it as context-poor. Kept for reference; **does not run**.
- **2026-06 (this rebuild):** everything in `src/`, `notebooks/`, `figures/`, this doc.

## 2. What was done, in order, and why

### 2.1 Diagnosed the original notebook
- **The data API is dead.** `curl https://statsapi.web.nhl.com/...` → no response (host
  decommissioned ~2023-24). The new endpoints return 200:
  - `https://api-web.nhle.com/v1/...` (play-by-play, schedules, rosters)
  - `https://api.nhle.com/stats/rest/en/...` (aggregated season stats)
- **Bugs found in the original** (fix these if anyone revives that notebook):
  - `timeOnIce`: `"20:30".replace(":",".")→20.30` treats 30s as 0.30 min. Wrong; should be
    `min + sec/60`. Corrupted every per-time metric.
  - Goal line hard-coded at `x=84`; the old NHL coords put the net at `x≈±89`.
  - No train/test split; AUC computed on training data; xG normalized by *total* TOI.

### 2.2 Migrated the data layer
- **MoneyPuck shot data** (`https://moneypuck.com/data.htm`), seasons **2015-2023**,
  ~**1,000,000 unblocked shots**, ~120 columns incl. pre-shot context (rebound, rush,
  last-event type/loc/time, strength, score, handedness) **and their production `xGoal`**.
  - Download note: canonical host needs a browser User-Agent; older seasons (2015-17)
    came from the mirror `peter-tanner.com/moneypuck/downloads/`.
- **New NHL API client** `src/nhl_api.py` — cached (`data/raw/*.json`) replacement for the
  dead scraper. Verified pulling all 82 Edmonton games + shot events with full context.

### 2.3 Built the modern xG model  (`src/modern_xg.py`, `src/run_pipeline.py`)
- **Population:** unblocked attempts (SHOT/MISS/GOAL), goalie in net, no shootout.
- **Features (~25):** geometry (arena-adjusted distance/angle/coords), shot type,
  pre-shot movement (`timeSinceLastEvent`, `speedFromLastEvent`, `distanceFromLastEvent`,
  `lastEventCategory`), rebound, strength (`skater_diff`), score diff, home, handedness.
- **Leakage guards:** `LEAK_COLS` in `modern_xg.py` drops outcome / MoneyPuck-model
  columns (`xGoal`, `shotGeneratedRebound`, `shotWasOnGoal`, `timeUntilNextEvent`, …).
- **Validation:** `GroupKFold(game_id)` → out-of-fold predictions for every shot →
  honest metrics **and** a leak-free xG for the player study. Saves
  `data/scored_shots.parquet` so downstream never retrains.
- **Result (OOF):**

  | model | log loss | Brier | AUC |
  |---|---|---|---|
  | baseline (mean rate) | 0.241 | 0.061 | 0.500 |
  | **our LightGBM** | **0.210** | **0.056** | **0.771** |
  | MoneyPuck `xGoal` | 0.201 | 0.054 | 0.794 |

  Figures: `figures/reliability.png` (well calibrated), `figures/importance.png`.

### 2.4 Re-did the McDavid study  (regular season, empty-net excluded)
- With leak-free OOF xG, McDavid scores **above expected in 7 of 9 seasons**:
  - 2022-23: **+15.3 G over xG** (~35% better finishing than league avg); 2019-20 +7.6.
  - Down goal-year 2023-24 ≈ neutral. Our xG ≈ MoneyPuck's per season (cross-check).
- **Answers the original open question:** the +finishing is real; the 2020 model was
  structurally blind to it (no context + the TOI-normalization bug).
- Output: `figures/mcdavid_career.{png,csv}`.

### 2.5 "How could we beat MoneyPuck?" — investigated and quantified
- **Ablation (the key insight):** distance *alone* → AUC 0.703 (= **71% of the entire
  headroom** from coin-flip to MoneyPuck). Adding angle/type/context shrinks fast;
  8→25 features only +0.007 AUC. **xG saturates; it's a low-ceiling, high-variance
  problem.** "Close to MoneyPuck" mostly means "close to the practical ceiling."
- **Why we got close with less:** (a) saturation, (b) we *reuse MoneyPuck's feature
  engineering* (their `shotRebound`, `speedFromLastEvent`, … are pre-computed), (c) AUC
  is forgiving — the log-loss gap (0.210 vs 0.201) is the more honest measure.
- **Shooter/goalie effects:** MoneyPuck's xG is **shooter-blind by design** (so G−xG
  measures finishing). Adding an empirical-Bayes shooter finishing prior **improves even
  MoneyPuck's own model** (logloss 0.2112→0.2107) — real, orthogonal, but **tiny per
  shot** (finishing skill is a season-aggregate effect, not a single-shot one).
- **Reframe "beat":** the meaningful scoreboard is **predictive validity** (does your xG
  predict *next* season's goals?), not in-sample log loss. Finishing-aware models win
  there even when shot-level metrics are flat.

### 2.6 Possession chains  (`src/possession.py`) — the soccer "N events back" idea
- Built from **raw NHL API** play-by-play (MoneyPuck only exposes ONE event back).
- For each shot, walks back up to `k=4` events: event type, team (turnover detection),
  x/y, Δtime, derived **puck-transport speed** (dist/Δt), and **royal-road crossing**.
- Demonstrated on a real McDavid goal: `takeaway → blocked-shot (EDM) → GOAL`, puck
  crossing the midline pre-shot. Most non-administrative events carry coordinates.
- **Missing vs soccer:** distance-to-other-players (needs tracking; see 2.7).

### 2.7 NHL EDGE + tracking-xG prototype  (`src/tracking_xg.py`)
- **NHL EDGE** = the league's puck/player tracking (since 2021-22): shot speed, skating
  speed, zone time, heat maps. Public site `nhl.com/nhl-edge`.
- **Finding:** the EDGE web API is **gated/obfuscated** (RTK-Query on `api-web.nhle.com`,
  paths not statically recoverable) and only surfaces **season aggregates**, not
  **per-shot tracking**. The per-shot layer (this-shot puck speed, defender distance) is
  **not freely available** — it's commercial (Sportlogiq/Stathletes).
- **Free substitute used:** **Big Data Cup 2021** hand-tracked dataset
  (`github.com/bigdatacup/Big-Data-Cup-2021`, `data/bigdatacup/bdc2021.csv`): 26,882
  events incl. shots with **pass origin/receiver coordinates** + **one-timer** flag.
- **Prototype result (after fixing leakage — see §3):** within shots that came off a
  pass, adding **pass trajectory** (pass distance, pre-shot angle change, cross-ice)
  improves xG from **AUC 0.788 → 0.838** (log loss −10%). Small sample → directional,
  but confirms tracking-era pre-shot-movement signal is real and worth chasing.

---

## 3. Hard-won learnings (read before touching the models)

1. **The old NHL API is gone.** Use `api-web.nhle.com` / `api.nhle.com`. Schema changed
   (`plays[].typeDescKey`, `details.xCoord/yCoord`, `situationCode`, …).
2. **xG saturates.** Distance does most of the work; expect diminishing returns. Don't
   chase marginal features — chase *new information* (sequences, tracking).
3. **AUC lies by omission.** Always also report **log loss / Brier** and a **reliability
   curve**. An xG that isn't calibrated is not an xG.
4. **Leakage is the main danger, and it's sneaky.** In the tracking prototype, a naive
   model hit **AUC 0.95** — a fantasy. Two leaks, both from the dataset being
   *assist-centric*: `time_since_pass` (assist logged at the goal's clock second) and
   `off_pass` (143/145 goals had a logged preceding pass). **Rule of thumb: any xG much
   above ~0.80 AUC is leaking.** Sanity-check feature/goal-rate tables before trusting a
   metric. The clean test restricts to a population where the suspect variable is constant.
5. **MoneyPuck being shooter-blind is a *feature*, not a bug** — it's what makes G−xG a
   finishing metric. Don't "fix" it without separating the two objectives (chance quality
   vs. goal prediction).
6. **Per-shot tracking isn't public.** Plan around MoneyPuck + raw-PBP sequences; treat
   tracking as commercial or small research datasets (Big Data Cup).
7. **Reproducibility:** large data is gitignored; `run_pipeline.py` caches to parquet;
   `nhl_api.py` caches JSON. Re-running is cheap except the one ~2-min training pass.

---

## 4. File map

```
NHL_McDavid.ipynb                       original 2020 notebook (reference; does NOT run)
README.md                               project overview + reproduce steps
HANDOFF.md                              this document
requirements.txt / .gitignore
build_notebook.py                       regenerates the analysis notebook
notebooks/NHL_McDavid_xG_modern.ipynb   the executed write-up (metrics, figures, McDavid)
src/nhl_api.py                          CURRENT NHL API client (cached) — replaces dead scraper
src/modern_xg.py                        MoneyPuck loading + feature engineering + LEAK_COLS
src/run_pipeline.py                     train (GroupKFold OOF) -> metrics, figures, parquet
src/predictive_validity.py              STEP 1: walk-forward next-season-goal forecast harness
src/shooter_goalie_effects.py           STEP 2: empirical-Bayes shooter+goalie effects on base xG
src/possession.py                       N-events-back possession chains from raw PBP
src/tracking_xg.py                      tracking-xG prototype on Big Data Cup (+leakage notes)
figures/                               reliability, importance, McDavid career, predictive_validity*, shooter_goalie*
data/  (gitignored)                     moneypuck/ (shots 2015-23), bigdatacup/, raw/, scored_shots*.parquet
```

How to run: `pip install -r requirements.txt`; download MoneyPuck shot CSVs into
`data/moneypuck/`; `python src/run_pipeline.py`; `python build_notebook.py`.

---

## 4b. Steps done since the original rebuild

- **Step 1 — predictive-validity harness — DONE** (`src/predictive_validity.py`, 2026-06-21).
  Walk-forward (train xG on seasons < N, score season N prospectively), aggregate to
  player-season, forecast season N+1 goals. Our walk-forward xG forecasts next-season goals
  **better than MoneyPuck and better than prior goals** — that's the honest "beat." Caches
  prospective per-shot xG to `data/scored_shots_walkforward.parquet` (gitignored). See memory
  for exact numbers.

- **Step 2 — hierarchical shooter + goalie empirical-Bayes effects — DONE**
  (`src/shooter_goalie_effects.py`, 2026-06-21). A logistic random-intercept layered on the
  base xG as a fixed offset: `logit(p_adj) = logit(xg_base) + u_shooter + v_goalie`, with
  `u~N(0,τ_u²)`, `v~N(0,τ_v²)` fit by an EM/empirical-Bayes scheme (penalised-MAP Newton for
  the offsets, EM for the variance components). Per-entity shrinkage falls out automatically;
  an unseen shooter gets exactly 0. **Leak-free / walk-forward:** `u,v` for season N are
  estimated ONLY from seasons < N (those with a walk-forward base xG, 2016..N-1), reusing the
  step-1 cache as the base xG (NOT the all-seasons OOF, which would leak future seasons into a
  "season<N" prior). Results:
  - *Shot-level* (n=659k, test seasons 2017-22): base logloss **0.21353 → 0.21329** (+shooter)
    → **0.21326** (+goalie). Real but tiny, exactly as expected (xG saturates; finishing is a
    season-aggregate effect, not a single-shot one). AUC 0.7669→0.7679 — well under the ~0.80
    "you're leaking" line. MoneyPuck still wins shot-level (0.2034) — fine, it's a richer
    shot-quality model; that's not what the shooter layer is for.
  - *Predictive validity* (the metric that matters, n=2080 player-season pairs): shooter-
    adjusted xG forecasts next-season goals **best of all**: Pearson r **0.706** vs base xG
    0.692, prior goals 0.690, MoneyPuck 0.680; R² 0.499 vs 0.479; RMSE 6.80 vs 6.93.
  - *McDavid:* shooter offset **u=+0.194 log-odds (+1.27 goals/100 shots vs base xG)**, rank
    **43 / 543** shooters (≥500 shots), **92nd league percentile** — corroborates step 1's
    top-decile finisher result via a totally different method. Draisaitl is #2 overall (nice
    Edmonton sanity check). `τ_u=0.163 > τ_v=0.072`: shooter finishing spread >> goalie spread.
    Goalie leaderboard (best suppression): Shesterkin, Sorokin, Saros, Vasilevskiy, Hellebuyck
    — exactly the league's elite, a strong face-validity check.
  - **INTERPRETATION (important):** the shooter-adjusted xG is **no longer a pure chance-quality
    metric** — it folds the shooter's finishing skill INTO the expectation. Keep **base xG for
    finishing studies** (G − xG), use the **adjusted xG for goal projection**. Don't compute
    "finishing above adjusted xG" — finishing is already baked in.
  - Outputs: `figures/shooter_goalie_shotlevel.csv`, `shooter_goalie_predval.csv`,
    `shooter_effect_ranking.csv`, `goalie_effect_ranking.csv`, `shooter_goalie_effects.png`.

---

## 5. Ranked next steps (for whoever picks this up)

1. ~~**Predictive-validity harness.**~~ **DONE — see §4b.**
2. ~~**Hierarchical shooter + goalie effects.**~~ **DONE — see §4b.**
3. **Full possession-chain model** *(medium).* Scrape a full league season of raw PBP via
   `nhl_api.py`, build `possession.chain_features` for every shot, and test last-4 vs
   last-1 (MoneyPuck-style) on identical data + as an additive layer on `xGoal`.
4. **Sequence model** *(medium-high).* GRU/Transformer over the possession event stream
   instead of hand-crafted chain features — likely where genuinely new shot-level gains are.
5. **Tracking features at scale** *(high, needs data access).* The Big Data Cup prototype
   shows the signal (+0.05 AUC). Productionizing needs Sportlogiq/Stathletes or whatever
   NHL EDGE eventually exposes per-shot. Pre-shot pass velocity, defender distance,
   screens, shooting-in-stride are the ceiling-raisers.
6. **Modeling polish** *(low, fractional).* Optuna tuning, monotonic constraints on
   distance/angle, isotonic recalibration, multi-task (goal/rebound/SOG jointly).

**Steps 1 & 2 are done** (§4b) — we now beat MoneyPuck *on the metric that matters*
(predictive validity) both with the walk-forward base xG and, more so, with the shooter-
adjusted xG. **Next up is #3 (possession chains),** then the harder sequence/tracking work.
