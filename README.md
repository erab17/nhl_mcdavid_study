# NHL Expected Goals (xG) & the Connor McDavid finishing study

A modern rebuild of my original 2020 exploration of NHL shot data. The goal is the
same — build an Expected Goals (xG) model and use it to study Connor McDavid — but
both the **data** and the **modelling** have been brought up to date.

## Why the rebuild?

1. **The original data source no longer exists.** The 2020 notebook scraped
   `statsapi.web.nhl.com`, which the NHL decommissioned in 2023-24. That notebook
   can no longer run. It's preserved as `NHL_McDavid.ipynb` for reference.
2. **Much richer public data exists now.** [MoneyPuck](https://moneypuck.com/data.htm)
   publishes shot-level data (~120 columns) including pre-shot context (rebound,
   rush, previous-event type/location/timing), strength state, score state, shooter
   handedness / off-wing — plus its own production xG to benchmark against.

## What's new vs. the 2020 notebook

| | 2020 | This rebuild |
|---|---|---|
| Data | dead `statsapi.web.nhl.com` | `api-web.nhle.com` + MoneyPuck shot data |
| Features | distance, angle, x | + pre-shot movement, rebound, strength, score, shot type, handedness |
| Model | logistic regression | LightGBM gradient boosting |
| Validation | none (AUC on training set) | GroupKFold out-of-fold (leak-free) |
| Metrics | AUC only | log loss + Brier + AUC + reliability curve |
| Fixed bugs | — | `timeOnIce "20:30"->20.30`, hard-coded goal line x=84 |

## Results

* **Modern xG:** AUC ~= 0.77, log loss ~= 0.21 (out-of-fold) — up from the original
  0.73 (measured on the training set), and within range of MoneyPuck's ~100-feature
  production model.
* **McDavid:** with context-aware, leak-free xG he scores **above expected in most
  seasons**, peaking at **~+15 goals above xG in 2022-23**. He is a genuinely elite
  *finisher* — a signal the 2020 model was structurally blind to.

That finishing result is **triangulated three independent ways** (all leak-free,
walk-forward):

1. **Predictive validity** — our xG forecasts *next-season* goals better (Pearson r
   **0.708**) than both MoneyPuck (0.696) and a player's own prior goals (0.686). The
   honest "beat" over MoneyPuck is here, on the metric that matters for projection — not
   on shot-level log loss.
2. **Hierarchical shooter effect** — a shrunk empirical-Bayes finishing prior ranks
   McDavid **43 / 543 qualified shooters (92nd pctile)**, +1.27 goals per 100 shots
   above expected. Linemate Draisaitl ranks #2; the goalie leaderboard
   (Shesterkin/Sorokin/Saros/…) is face-valid. The shooter-adjusted xG also forecasts
   next-season goals best of all.
3. **Possession chains** — a negative control: looking >1 event back adds ~nothing to
   shot xG (last-4 ≈ last-1), so the McDavid signal isn't an under-modeling artifact. The
   event-stream sequence is exhausted; the one remaining frontier is player/puck tracking.

## Layout

```
src/nhl_api.py               cached client for the CURRENT NHL APIs (replaces the dead scraper)
src/modern_xg.py             data loading + feature engineering for the xG model
src/run_pipeline.py          train (GroupKFold OOF) -> metrics, figures, scored_shots.parquet
src/predictive_validity.py   walk-forward harness: does xG forecast next-season goals?
src/shooter_goalie_effects.py  empirical-Bayes shooter/goalie finishing effects + rankings
src/possession_model.py      last-N-event possession chains (negative result)
src/tracking_xg.py           Big Data Cup tracking prototype + leakage lesson
build_notebook.py            generates the analysis notebook
notebooks/NHL_McDavid_xG_modern.ipynb   the write-up (run end-to-end)
figures/              metrics, reliability, importance, McDavid career, step 1-3 tables
NHL_McDavid.ipynb     the original 2020 notebook (kept for reference; no longer runs)
NEXT_STEPS.md         ranked forward roadmap; HANDOFF.md has the full history
```

## Reproduce

```bash
pip install -r requirements.txt

# 1. Download MoneyPuck shot data (2015-2023) into data/moneypuck/
#    from https://moneypuck.com/data.htm  (shots_YYYY.zip)

# 2. Train + score (~2 min) -> data/scored_shots.parquet + figures/
python src/run_pipeline.py

# 3. (optional) regenerate the triangulation tables in figures/
python src/predictive_validity.py      # step 1
python src/shooter_goalie_effects.py   # step 2
python src/possession_model.py         # step 3 (scrapes a full season of PBP)

# 4. Build & run the notebook (loads the figures/ tables above)
python build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/NHL_McDavid_xG_modern.ipynb
```

## Data note (the true tracking frontier)

The remaining frontier is **NHL EDGE puck-and-player tracking** (shot speed, skating
speed, separation), which would let us separate shot *quality we still can't see* from
pure finishing talent. It isn't published as a bulk per-shot feed yet, so it's noted
as future work rather than used here.
