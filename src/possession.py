"""
Possession-chain features for hockey shots -- the hockey analog of a soccer
"last-N ball events" build-up feature list.

For every shot we walk BACK up to `k` events through the play-by-play stream and,
for each lag, record: event type, which team had it (turnover detection), location,
the time/distance/derived puck-speed between consecutive events, and lateral
("royal road") movement. This is exactly the structure of a soccer possession chain;
the only thing missing vs. soccer 360 / tracking data is distance-to-other-players,
which hockey public PBP does not provide.

Built on the raw NHL API stream via nhl_api.py (one event back is all MoneyPuck gives).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import nhl_api

XY_EVENTS = {  # events that carry coordinates and represent puck location
    "faceoff", "hit", "shot-on-goal", "blocked-shot", "missed-shot",
    "giveaway", "takeaway", "goal", "penalty",
}
SHOT_EVENTS = {"shot-on-goal", "missed-shot", "goal"}


def _secs(period, mmss):
    m, s = mmss.split(":")
    return (period - 1) * 1200 + int(m) * 60 + int(s)


def game_events(game_id: int) -> pd.DataFrame:
    """Ordered, coordinate-bearing events for a game with absolute game-seconds."""
    data = nhl_api._get_json(f"{nhl_api.WEB}/gamecenter/{game_id}/play-by-play",
                             cache_key=f"pbp_{game_id}")
    rows = []
    for p in sorted(data.get("plays", []), key=lambda x: x.get("sortOrder", 0)):
        t = p.get("typeDescKey")
        d = p.get("details", {})
        if t not in XY_EVENTS or "xCoord" not in d:
            continue
        rows.append({
            "game_id": game_id,
            "event": t,
            "team": d.get("eventOwnerTeamId"),
            "x": d.get("xCoord"), "y": d.get("yCoord"), "zone": d.get("zoneCode"),
            "shooter_id": d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            "shot_type": d.get("shotType"),
            "t": _secs(p["periodDescriptor"]["number"], p["timeInPeriod"]),
            "period": p["periodDescriptor"]["number"],
            "is_goal": int(t == "goal"),
        })
    return pd.DataFrame(rows)


def chain_features(ev: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """For each shot in `ev`, flatten the previous `k` events into chain features."""
    ev = ev.reset_index(drop=True)
    out = []
    for i, row in ev.iterrows():
        if row["event"] not in SHOT_EVENTS:
            continue
        rec = {"game_id": row["game_id"], "shooter_id": row["shooter_id"],
               "x": row["x"], "y": row["y"], "is_goal": row["is_goal"],
               "shot_type": row["shot_type"], "period": row["period"]}
        shooting_team = row["team"]
        prev_x, prev_y, prev_t = row["x"], row["y"], row["t"]
        possession_changes, passes = 0, 0
        for lag in range(1, k + 1):
            j = i - lag
            if j < 0 or ev.loc[j, "period"] != row["period"]:
                rec[f"l{lag}_event"] = "none"
                continue
            e = ev.loc[j]
            dx, dy = prev_x - e["x"], prev_y - e["y"]
            dist = float(np.hypot(dx, dy))
            dt = max(prev_t - e["t"], 0)
            same = int(e["team"] == shooting_team)
            rec[f"l{lag}_event"] = e["event"]
            rec[f"l{lag}_same_team"] = same
            rec[f"l{lag}_x"] = e["x"]; rec[f"l{lag}_y"] = e["y"]
            rec[f"l{lag}_dist"] = dist
            rec[f"l{lag}_dt"] = dt
            rec[f"l{lag}_speed"] = dist / dt if dt > 0 else 0.0   # ft/s puck transport
            # royal-road crossing: puck changed side of the ice between these events
            rec[f"l{lag}_cross_mid"] = int(np.sign(e["y"]) != np.sign(prev_y) and abs(prev_y) > 3)
            if same and e["event"] not in ("hit",):
                passes += 1
            if not same:
                possession_changes += 1
            prev_x, prev_y, prev_t = e["x"], e["y"], e["t"]
        rec["chain_passes"] = passes
        rec["chain_turnovers"] = possession_changes
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == "__main__":
    # Demo: find a McDavid goal and print the possession chain that produced it.
    MCD = nhl_api.MCDAVID_ID
    found = None
    for gid in nhl_api.season_game_ids(2023)[:40]:
        ev = game_events(gid)
        goals = ev[(ev.event == "goal") & (ev.shooter_id == MCD)]
        if not goals.empty:
            idx = goals.index[0]
            window = ev.loc[max(0, idx - 4):idx,
                            ["event", "team", "x", "y", "zone", "t", "shooter_id"]]
            print(f"McDavid goal in game {gid} -- the 5-event build-up:\n")
            print(window.to_string(index=False))
            ch = chain_features(ev)
            row = ch[(ch.shooter_id == MCD) & (ch.is_goal == 1)].iloc[0]
            cols = [c for c in row.index if c.startswith(("l1_", "l2_", "l3_", "l4_", "chain"))]
            print("\nFlattened chain features for that shot:\n")
            print(row[cols].to_string())
            found = gid
            break
    if not found:
        print("No McDavid goal in the scanned games.")
