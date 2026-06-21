"""
Minimal client for the *current* NHL APIs (the ones that replaced the
decommissioned statsapi.web.nhl.com that the original 2020 notebook used).

Endpoints
---------
  api-web.nhle.com/v1   : game play-by-play, schedules, rosters, landing pages
  api.nhle.com/stats    : aggregated skater/goalie/team season stats

Everything is cached to data/raw/ as JSON so re-runs are instant and polite.

Quick demo:
    python src/nhl_api.py            # pulls a recent game's shots + McDavid shots
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import requests

WEB = "https://api-web.nhle.com/v1"
STATS = "https://api.nhle.com/stats/rest/en"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(CACHE_DIR, exist_ok=True)
SHOT_EVENTS = {"goal", "shot-on-goal", "missed-shot"}
MCDAVID_ID = 8478402


def _get_json(url: str, cache_key: str | None = None, ttl_days: float = 30) -> dict:
    """GET with on-disk JSON caching."""
    if cache_key:
        path = os.path.join(CACHE_DIR, cache_key + ".json")
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_days * 86400:
            with open(path) as fh:
                return json.load(fh)
    r = requests.get(url, timeout=30, headers={"User-Agent": "nhl-xg-study/1.0"})
    r.raise_for_status()
    data = r.json()
    if cache_key:
        with open(os.path.join(CACHE_DIR, cache_key + ".json"), "w") as fh:
            json.dump(data, fh)
    return data


def season_game_ids(season: int, game_type: int = 2) -> list[int]:
    """All gamePks for a season. game_type 2=regular, 3=playoffs. season=2023 -> 2023-24."""
    sched = _get_json(f"{WEB}/club-schedule-season/EDM/{season}{season+1}",
                      cache_key=f"sched_EDM_{season}")
    return sorted({g["id"] for g in sched["games"] if g["gameType"] == game_type})


def league_game_ids(season: int, game_type: int = 2) -> list[int]:
    """ALL gamePks league-wide for a season (every team, not just EDM).

    Uses the stats API game list (api.nhle.com), which returns the whole season
    in one cached call. game_type 2=regular, 3=playoffs. season=2023 -> 2023-24.
    """
    data = _get_json(
        f"{STATS}/game?cayenneExp=season={season}{season+1}",
        cache_key=f"games_{season}")
    return sorted({g["id"] for g in data["data"] if g.get("gameType") == game_type})


def play_by_play(game_id: int) -> pd.DataFrame:
    """Return shot-level rows for one game from the new play-by-play endpoint."""
    data = _get_json(f"{WEB}/gamecenter/{game_id}/play-by-play", cache_key=f"pbp_{game_id}")
    rows = []
    for p in data.get("plays", []):
        if p.get("typeDescKey") not in SHOT_EVENTS:
            continue
        d = p.get("details", {})
        rows.append({
            "game_id": game_id,
            "event": p["typeDescKey"],
            "period": p.get("periodDescriptor", {}).get("number"),
            "time": p.get("timeInPeriod"),
            "x": d.get("xCoord"),
            "y": d.get("yCoord"),
            "zone": d.get("zoneCode"),
            "shot_type": d.get("shotType"),
            "shooter_id": d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            "goalie_id": d.get("goalieInNetId"),
            "situation": p.get("situationCode"),
            "defending_side": p.get("homeTeamDefendingSide"),
            "is_goal": int(p["typeDescKey"] == "goal"),
        })
    return pd.DataFrame(rows)


def player_shots(player_id: int, seasons: list[int]) -> pd.DataFrame:
    """All shot attempts by a player across the given seasons (regular season)."""
    frames = []
    for season in seasons:
        for gid in season_game_ids(season):
            df = play_by_play(gid)
            if not df.empty:
                frames.append(df[df["shooter_id"] == player_id])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def skater_season_stats(season: int) -> pd.DataFrame:
    """Aggregated skater summary stats for a season (api.nhle.com)."""
    data = _get_json(
        f"{STATS}/skater/summary?limit=-1&cayenneExp=seasonId={season}{season+1}%20and%20gameTypeId=2",
        cache_key=f"skater_summary_{season}")
    return pd.DataFrame(data["data"])


if __name__ == "__main__":
    print("Demo: pulling McDavid's most recent completed season of shots from the NEW API")
    gids = season_game_ids(2023)
    print(f"  EDM 2023-24 regular-season games: {len(gids)}")
    df = play_by_play(gids[0])
    print(f"  sample game {gids[0]}: {len(df)} shot events")
    print(df.head(8).to_string(index=False))
    mcd = df[df["shooter_id"] == MCDAVID_ID]
    print(f"  McDavid shot events in that game: {len(mcd)}")
