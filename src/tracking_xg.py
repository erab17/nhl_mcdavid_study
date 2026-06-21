"""
Tracking-augmented xG prototype on the Big Data Cup 2021 hand-tracked dataset.

Public play-by-play (and therefore MoneyPuck) gives shot geometry + ONE prior event.
This dataset additionally encodes, per shot, the *pre-shot pass* (passer and receiver
coordinates) and a *one-timer* flag -- the "shot quality" signal that normally only
puck/player tracking (NHL EDGE) provides.

We hold the model class fixed and compare two feature sets on the SAME shots:
  A) geometry-only  : distance, angle, shot type            (what public PBP supports)
  B) + tracking     : one-timer, off-the-pass, pass distance / lateral movement /
                      royal-road crossing / pre-shot angle change / time since pass

Evaluation: leave-one-game-out (7 games) out-of-fold log loss / AUC.
Small sample -> a prototype to quantify the *signal*, not a production model.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import log_loss, roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bigdatacup", "bdc2021.csv")
NET_X_RIGHT, NET_X_LEFT, MID_Y = 189.0, 11.0, 42.5


def _clock_s(c):
    m, s = str(c).split(":")
    return int(m) * 60 + int(s)


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["clock_s"] = df["Clock"].map(_clock_s)
    rows = []
    for i, r in df.iterrows():
        if r["Event"] != "Shot" or r["Detail 2"] == "Blocked":
            continue  # model unblocked attempts (fenwick), goalie-relevant
        x, y = r["X Coordinate"], r["Y Coordinate"]
        net_x = NET_X_RIGHT if x > 100 else NET_X_LEFT
        dist = float(np.hypot(net_x - x, MID_Y - y))
        angle = float(np.degrees(np.arctan2(abs(y - MID_Y), abs(net_x - x))))

        rec = {
            "game": r["game_date"] + r["Home Team"],
            "goal": int(r["Detail 4"] == "t"),
            # ---- geometry (public-PBP-equivalent) ----
            "distance": dist,
            "angle": angle,
            "shot_type": r["Detail 1"],
            # ---- tracking-only ----
            "one_timer": int(r["Detail 3"] == "t"),
        }

        # find the immediately preceding same-team pass (Play / Incomplete Play)
        off_pass = 0
        pass_dist = pass_lat = pass_angle_chg = royal_road = 0.0
        time_since_pass = 5.0
        for j in range(i - 1, max(i - 4, -1), -1):
            e = df.loc[j]
            if e["Period"] != r["Period"]:
                break
            if e["Event"] in ("Play", "Incomplete Play") and e["Team"] == r["Team"] \
                    and not np.isnan(e["X Coordinate 2"]):
                px, py = e["X Coordinate"], e["Y Coordinate"]       # passer
                rx, ry = e["X Coordinate 2"], e["Y Coordinate 2"]   # receiver
                off_pass = int(e["Event"] == "Play")
                pass_dist = float(np.hypot(rx - px, ry - py))
                pass_lat = float(abs(ry - py))                      # cross-ice component
                # royal road: pass crossed the slot centerline before the shot
                royal_road = int(np.sign(py - MID_Y) != np.sign(ry - MID_Y) and abs(py - MID_Y) > 5)
                a_pass = np.degrees(np.arctan2(abs(py - MID_Y), abs(net_x - px)))
                pass_angle_chg = float(abs(angle - a_pass))
                time_since_pass = float(max(e["clock_s"] - r["clock_s"], 0))
                break
        rec.update(off_pass=off_pass, pass_dist=pass_dist, pass_lat=pass_lat,
                   royal_road=royal_road, pass_angle_chg=pass_angle_chg,
                   time_since_pass=time_since_pass)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["shot_type"] = out["shot_type"].astype("category")
    return out


# NOTE: `time_since_pass` is intentionally EXCLUDED -- in this dataset a goal's
# assisting pass is logged at the same clock second as the goal, so the time gap
# leaks the outcome (72% goal rate at <=0.5s). We keep only the leak-free pre-shot
# *geometry* of the pass plus the one-timer mechanic.
GEO = ["distance", "angle", "shot_type"]
TRACK = ["one_timer", "off_pass", "pass_dist", "pass_lat", "royal_road",
         "pass_angle_chg"]

PARAMS = dict(objective="binary", n_estimators=120, learning_rate=0.05,
              num_leaves=15, min_child_samples=40, reg_lambda=2.0, verbose=-1)


def oof(data, feats):
    X, y, g = data[feats], data["goal"].values, data["game"].values
    p = np.zeros(len(data))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return np.clip(p, 1e-6, 1 - 1e-6)


# Pass-trajectory features that are NOT entangled with assist-tracking, evaluated
# only within the population of shots that already came off a pass.
PASS_GEO = ["one_timer", "pass_dist", "pass_lat", "royal_road", "pass_angle_chg"]


def main():
    df = pd.read_csv(DATA)
    data = build(df)
    print(f"unblocked shots: {len(data)} | goals: {data['goal'].sum()} | "
          f"goal rate: {data['goal'].mean():.3f} | games: {data['game'].nunique()}")

    # Transparency: show why the naive 'off_pass' feature is contaminated.
    print("\nLeakage check -- this dataset is assist-centric, so 'off a pass' "
          "is a proxy for 'had an assist':")
    print(data.groupby("off_pass").agg(shots=("goal", "size"),
          goals=("goal", "sum"), rate=("goal", "mean")).round(3).to_string())

    # CLEAN experiment: restrict to shots off a pass, test pass *trajectory* signal.
    sub = data[data["off_pass"] == 1].reset_index(drop=True)
    y = sub["goal"].values
    pa = oof(sub, GEO)
    pb = oof(sub, GEO + PASS_GEO)
    base = np.full(len(sub), y.mean())
    print(f"\n=== CLEAN test: shots off a pass only (n={len(sub)}, goals={y.sum()}) ===")
    print("    does the pass's trajectory improve xG beyond shot location?")
    print(f"{'model':<34}{'log loss':>10}{'AUC':>8}")
    for name, p in [("baseline (mean rate)", base),
                    ("A: shot geometry only", pa),
                    ("B: + pass trajectory", pb)]:
        print(f"{name:<34}{log_loss(y, p):>10.4f}{roc_auc_score(y, p):>8.4f}")
    lift = (log_loss(y, pa) - log_loss(y, pb)) / log_loss(y, pa) * 100
    print(f"\npass-trajectory features change log loss by {lift:+.1f}% and AUC "
          f"{roc_auc_score(y, pa):.3f} -> {roc_auc_score(y, pb):.3f}")

    m = lgb.LGBMClassifier(**PARAMS).fit(sub[GEO + PASS_GEO], y)
    imp = pd.Series(m.booster_.feature_importance("gain"),
                    index=GEO + PASS_GEO).sort_values(ascending=False)
    print("\n=== feature importance (gain), off-a-pass shots ===")
    print(imp.round(0).to_string())


if __name__ == "__main__":
    main()
