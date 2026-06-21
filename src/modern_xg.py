"""
Modern Expected Goals (xG) model built on MoneyPuck shot-level data.

This replaces the 2020 logistic-regression-on-(distance, angle) model with a
gradient-boosted model that uses pre-shot context (rebound, rush, last-event
type/location/time, strength state, score state, shooter handedness / off-wing).

Key design choices vs. the original notebook:
  * Population: unblocked shot attempts (fenwick = SHOT + MISS + GOAL),
    excluding empty-net and shootout (period 5) events.
  * Out-of-fold (OOF) predictions via GroupKFold on game_id so every shot is
    scored by a model that never saw its game -> honest evaluation *and* a
    leak-free xG for the player/season study.
  * Evaluation is probabilistic: log loss, Brier score, AUC, plus a reliability
    (calibration) curve -- not AUC alone.
  * MoneyPuck's production `xGoal` is kept aside purely as a benchmark.

Data source: https://moneypuck.com/data.htm  (shot files, 2015-2023 downloaded).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "moneypuck")
MCDAVID_ID = 8478402

# ---- columns that would leak the outcome or are post-shot / model outputs ----
LEAK_COLS = {
    "goal", "xGoal", "xRebound", "xFroze", "xPlayContinuedInZone",
    "xPlayContinuedOutsideZone", "xPlayStopped", "xShotWasOnGoal",
    "shotGeneratedRebound", "shotGoalieFroze", "shotPlayContinuedInZone",
    "shotPlayContinuedOutsideZone", "shotPlayStopped", "shotWasOnGoal",
    "shotOnEmptyNet", "timeUntilNextEvent", "homeTeamWon",
}

NUMERIC_FEATURES = [
    "arenaAdjustedShotDistance", "shotDistance", "shotAngle", "shotAngleAdjusted",
    "arenaAdjustedXCordABS", "arenaAdjustedYCordAbs",
    "timeSinceLastEvent", "distanceFromLastEvent", "speedFromLastEvent",
    "lastEventShotAngle", "lastEventShotDistance",
    "lastEventxCord_adjusted", "lastEventyCord_adjusted",
    "shotRush", "shotRebound", "offWing",
    "timeSinceFaceoff", "period", "shooterTimeOnIceSinceFaceoff",
    # engineered below:
    "skater_diff", "score_diff", "is_home",
]

CATEGORICAL_FEATURES = ["shotType", "lastEventCategory", "shooterLeftRight"]


def load_seasons(seasons=None) -> pd.DataFrame:
    """Load and concatenate MoneyPuck shot CSVs."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "shots_*.csv")))
    if seasons is not None:
        files = [f for f in files if int(f.split("_")[-1].split(".")[0]) in seasons]
    if not files:
        raise FileNotFoundError(f"No shot CSVs found in {DATA_DIR}")
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to modeled population and engineer strength/score/home features."""
    df = df.copy()

    # Model population: unblocked attempts with a goalie in net, regulation/OT only.
    df = df[df["shotOnEmptyNet"] == 0]
    df = df[df["period"] <= 4]              # drop shootout (period 5)
    df = df[df["event"].isin(["SHOT", "MISS", "GOAL"])]

    # Strength state: skater differential from the shooter's perspective.
    home = df["isHomeTeam"].fillna(0).astype(int)
    shooting_skaters = np.where(home == 1, df["homeSkatersOnIce"], df["awaySkatersOnIce"])
    defending_skaters = np.where(home == 1, df["awaySkatersOnIce"], df["homeSkatersOnIce"])
    df["skater_diff"] = shooting_skaters - defending_skaters

    # Score differential from the shooter's perspective.
    shooting_goals = np.where(home == 1, df["homeTeamGoals"], df["awayTeamGoals"])
    defending_goals = np.where(home == 1, df["awayTeamGoals"], df["homeTeamGoals"])
    df["score_diff"] = shooting_goals - defending_goals
    df["is_home"] = home

    # Categoricals -> pandas 'category' dtype (LightGBM handles natively).
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].astype("category")

    df = df.reset_index(drop=True)
    return df


def feature_matrix(df: pd.DataFrame):
    """Return (X, y, groups) for modeling."""
    feats = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    feats = [c for c in feats if c not in LEAK_COLS]
    X = df[feats].copy()
    y = df["goal"].astype(int).values
    groups = df["game_id"].values
    return X, y, feats, groups
