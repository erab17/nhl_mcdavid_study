# Next steps — NHL xG / McDavid study

> Living roadmap. Updated after **steps 1–3** (predictive-validity harness, shooter+goalie
> empirical-Bayes effects, possession chains). For the full history and the "why," see
> `HANDOFF.md` (§4b has the step-by-step results, §3 the hard-won learnings).

## Where we are

The modeling-experiment arc is **complete and coherent**:

- **Base xG** (LightGBM, leak-free): logloss 0.210 / AUC 0.771, just behind MoneyPuck (0.201 / 0.794) with ~25 features vs ~100.
- **Step 1 — predictive validity:** our walk-forward xG forecasts *next-season* goals better than MoneyPuck and better than prior goals. That's the honest "beat."
- **Step 2 — shooter+goalie EB effects:** a shooter-adjusted xG forecasts next-season goals best of all (r 0.706). McDavid's finishing offset ranks **43/543, 92nd pctile** — a second, independent confirmation of the headline result.
- **Step 3 — possession chains:** **negative result.** Looking >1 event back adds ~nothing to shot xG (last-4 ≈ last-1). Hockey's sequence signal is tapped out at one event back.

Net: **the McDavid finishing result is real and triangulated three ways**, and we know where the
remaining modeling headroom is *not* (more features, deeper sequences) and where it might be
(spatial/tracking).

## Ranked next steps

### 1. Consolidate the study into one write-up  ·  *high value, low effort*  ·  **recommended first**
The repo's value is the McDavid story, but the analysis now lives across `run_pipeline.py`,
`predictive_validity.py`, `shooter_goalie_effects.py`, `possession_model.py`, while the executed
notebook (`notebooks/NHL_McDavid_xG_modern.ipynb`) predates steps 1–3.
- **First action:** extend `build_notebook.py` to fold in the step 1–3 figures/tables and tell
  one narrative: base xG → predictive validity vs MoneyPuck → shooter EB effect & McDavid's rank
  → possession-depth negative result. Refresh `README.md`'s results section to match.
- **Done when:** a reader can go notebook-only from "is McDavid a real finisher?" to the
  triangulated yes, with the leakage caveats inline.

### 2. Spatial / tracking features  ·  *high value, gated on data*  ·  **best lead for new signal**
Step 3 shows event-stream *sequence* is exhausted; the only untapped shot-level headroom is
**where the other players are** (defender distance, screens, shooting-in-stride).
- **First actions:** (a) re-probe whether NHL EDGE exposes anything per-shot now (last checked
  2026-06, season-aggregates only); (b) turn the Big Data Cup tracking prototype (`tracking_xg.py`,
  +0.05 AUC, small sample) into a full, leakage-audited quantification and write it up as the
  "ceiling estimate"; (c) cheap proxy worth a look — derive a screen/traffic or defender-count
  proxy from on-ice player lists in the raw PBP (we already scrape full-season PBP).
- **Risk:** per-shot tracking is commercial (Sportlogiq/Stathletes). Treat Big Data Cup as the
  evidence and EDGE as the watch-this-space.

### 3. Honest modeling polish  ·  *low value, low effort*  ·  closes the gap to MoneyPuck
Won't change any conclusion, but tightens the base model and removes "they just have more
features" caveats.
- Isotonic/Platt recalibration, monotonic constraints on distance/angle, Optuna tuning,
  multi-task (goal/rebound/SOG jointly).
- **Loose end to close:** step 2 only scored the *shooter*-adjusted xG in the predictive-validity
  harness — add the *goalie*-adjusted (and joint) variant for completeness.

### 4. Sequence model against a *different* target  ·  *medium, speculative*
A GRU/Transformer over the possession event stream is **deprioritised for shot xG** (step 3 showed
the depth signal isn't there). Only worth it pointed at a target where sequence *should* matter:
zone-entry value, chance creation, or next-shot timing — a different study, not this one.

---

**Recommended order: 1 → 2 → 3.** Ship the story first (it's the point of the repo and it's ready),
then chase the one remaining source of new shot-level signal (spatial), then polish. #4 is a
separate project.
