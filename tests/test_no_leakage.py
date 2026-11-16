"""
Anti-cheating test: proves no feature uses information from its own game
(or from any later game).

The idea
--------
1. Build features normally.
2. Pick a date. SCRAMBLE everything that happened on that date and after
   (points, minutes, shots... for players AND teams) - i.e. rewrite "the future".
3. Build features again.
4. For games ON that date, every FEATURE must be exactly the same as before,
   because a fair feature only looks at earlier days. If a feature changed,
   it was peeking at its own game (or the future).
   (The TARGET, PTS, should change - that proves the scramble actually happened.)

We repeat this for several random dates.

Run it with:
    .venv\\Scripts\\python.exe -m tests.test_no_leakage
"""
import numpy as np
import pandas as pd

from src.features import FEATURES, SAME_GAME_STATS, SIM_COLS, build_features, load_raw


def scramble_from(df, date, rng):
    """Replace every box-score stat on/after `date` with random numbers."""
    df = df.copy()
    rows = pd.to_datetime(df["GAME_DATE"]) >= date
    for col in SAME_GAME_STATS + ["PLUS_MINUS", "FG_PCT", "FG3_PCT", "FT_PCT"]:
        if col == "MIN":
            # Keep 0-minute rows at 0: WHO plays is known before tip-off (inactive list),
            # so we don't change that. Team minutes stay a valid game length.
            low, high = (1, 48) if "PLAYER_ID" in df else (240, 241)
            played = rows & (df["MIN"] > 0)
            df.loc[played, col] = rng.integers(low, high, played.sum())
        else:
            df.loc[rows, col] = rng.integers(0, 60, rows.sum())
    return df


def main():
    rng = np.random.default_rng(42)
    player, team, ages = load_raw()
    original = build_features(player, team, ages)

    all_dates = pd.to_datetime(player["GAME_DATE"]).drop_duplicates().sort_values()
    test_dates = rng.choice(all_dates.iloc[100:].to_numpy(), size=5, replace=False)

    failed = False
    for date in sorted(test_dates):
        date = pd.Timestamp(date)
        scrambled = build_features(scramble_from(player, date, rng), scramble_from(team, date, rng), ages)

        key = ["GAME_ID", "PLAYER_ID"]
        a = original[original["GAME_DATE"] == date].set_index(key).sort_index()
        b = scrambled[scrambled["GAME_DATE"] == date].set_index(key).sort_index()

        leaks = [f for f in FEATURES + SIM_COLS if not np.allclose(a[f], b[f], equal_nan=True)]
        changed = (a["PTS"] != b["PTS"]).mean()
        status = "PASS" if not leaks else f"LEAK in {leaks}"
        print(f"{date.date()}: {len(a):3d} player-games, target changed in {changed:.0%} -> {status}")
        failed |= bool(leaks)

    if failed:
        raise SystemExit(1)
    print(f"\nAll {len(FEATURES) + len(SIM_COLS)} columns passed: none of them look at their own game or the future.")


if __name__ == "__main__":
    main()
