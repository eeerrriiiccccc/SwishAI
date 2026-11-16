"""
Phase 0 - Download raw NBA game logs.

What this script does
---------------------
For every season in config.SEASONS it downloads two tables from stats.nba.com
(using the free `nba_api` package):

  1. PLAYER game logs: one row per player per game (points, minutes, shots...)
  2. TEAM game logs:   one row per team per game (we'll use these later to
                       compute each opponent's defensive rating and pace)
  3. Player AGES:      one row per player per season

Each table is saved to data/raw/ as a .parquet file (a fast, compact table
format - think "CSV, but better"). If a file already exists we skip it, so
you can re-run this script safely without downloading everything again.

Run it from the SwishAI folder with:
    .venv\\Scripts\\python.exe -m src.fetch_data
"""
import time

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguegamelog

from src.config import RAW_DIR, SEASONS


def download_season(season: str, kind: str) -> pd.DataFrame:
    """
    Download one season of game logs.

    season: e.g. "2023-24"
    kind:   "P" for player logs, "T" for team logs
    """
    # stats.nba.com sometimes times out, so we try up to 3 times.
    for attempt in range(1, 4):
        try:
            response = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",  # skip playoffs for now
                player_or_team_abbreviation=kind,
                timeout=60,
            )
            # The API returns a list of tables; the first one is what we want.
            return response.get_data_frames()[0]
        except Exception as error:  # network hiccup, timeout, etc.
            print(f"    attempt {attempt} failed: {error}")
            time.sleep(5 * attempt)  # wait a bit longer each retry
    raise RuntimeError(f"Could not download {kind} logs for {season}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        for kind, name in [("P", "player"), ("T", "team")]:
            out_file = RAW_DIR / f"{name}_logs_{season}.parquet"

            if out_file.exists():
                print(f"[skip] {out_file.name} already downloaded")
                continue

            print(f"[download] {name} logs for {season} ...")
            df = download_season(season, kind)
            df["SEASON"] = season  # remember which season each row is from
            df.to_parquet(out_file, index=False)
            print(f"    saved {len(df):,} rows -> {out_file.name}")

            # Be polite to the NBA's servers - don't hammer them.
            time.sleep(1.5)

        # 3. Player AGE for that season (one small table per season).
        age_file = RAW_DIR / f"player_ages_{season}.parquet"
        if age_file.exists():
            print(f"[skip] {age_file.name} already downloaded")
        else:
            print(f"[download] player ages for {season} ...")
            ages = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season, season_type_all_star="Regular Season", timeout=60
            ).get_data_frames()[0][["PLAYER_ID", "AGE"]]
            ages["SEASON"] = season
            ages.to_parquet(age_file, index=False)
            print(f"    saved {len(ages):,} rows -> {age_file.name}")
            time.sleep(1.5)

    print("\nDone! Raw data is in", RAW_DIR)


# This line means: only run main() when the file is run directly,
# not when another file imports something from it.
if __name__ == "__main__":
    main()
