"""
Phase 10 - Forecast a player's next game.

Examples (from the SwishAI folder, with the venv active):
    python -m src.predict "Dylan Harper"
    python -m src.predict "Dylan Harper" --line 18.5
    python -m src.predict "Dylan Harper" --out "Victor Wembanyama"        # a teammate is ruled out
    python -m src.predict "Cooper Flagg" --opponent LAL --away --date 2026-11-12

How it works
------------
1. Download this season's games so far (once per day).
2. Find the player's next game on the NBA schedule (or use --opponent/--date).
3. Add a placeholder row for that game for everyone expected to play:
   each team's players from its last 5 games, minus anyone listed with --out / --opp-out.
   (Injury reports aren't in our free data - pass them yourself.)
4. Run the normal feature builder. The placeholder's stats are blank, which is
   fine: tests/test_no_leakage.py proved features never look at their own game.
5. Predict with the final blend (trained on all 5 seasons, saved to models/),
   and simulate the game 20,000 times for the range and probabilities.
"""
import argparse
import difflib
import pickle
import time
from datetime import date as Date

import numpy as np
import pandas as pd

from src.config import CURRENT_DIR, CURRENT_SEASON, MODELS_DIR
from src.features import SAME_GAME_STATS, build_features, load_raw

PLACEHOLDER_GAME_ID = "9999999999"
MODEL_FILE = MODELS_DIR / "swishai_blend.pkl"


# =============================================================================
# Data for the season in progress
# =============================================================================
def fetch_current(force=False):
    """Download this season's player logs, team logs, ages and schedule (at most once per day)."""
    from nba_api.stats.endpoints import leaguedashplayerstats, leaguegamelog, scheduleleaguev2

    CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = CURRENT_DIR / "downloaded_on.txt"
    if not force and stamp.exists() and stamp.read_text() == str(Date.today()):
        return
    print(f"Downloading {CURRENT_SEASON} games so far ...")
    for kind, name in [("P", "player"), ("T", "team")]:
        df = leaguegamelog.LeagueGameLog(season=CURRENT_SEASON, season_type_all_star="Regular Season",
                                         player_or_team_abbreviation=kind, timeout=60).get_data_frames()[0]
        df["SEASON"] = CURRENT_SEASON
        df.to_parquet(CURRENT_DIR / f"{name}_logs.parquet", index=False)
        time.sleep(1)
    ages = leaguedashplayerstats.LeagueDashPlayerStats(season=CURRENT_SEASON, timeout=60).get_data_frames()[0][["PLAYER_ID", "AGE"]]
    ages["SEASON"] = CURRENT_SEASON
    ages.to_parquet(CURRENT_DIR / "player_ages.parquet", index=False)
    time.sleep(1)
    sched = scheduleleaguev2.ScheduleLeagueV2(season=CURRENT_SEASON, timeout=60).get_data_frames()[0]
    sched[["gameId", "gameDateEst", "gameDateTimeUTC", "gameStatus", "homeTeam_teamTricode", "awayTeam_teamTricode"]].to_parquet(
        CURRENT_DIR / "schedule.parquet", index=False)
    stamp.write_text(str(Date.today()))


def load_all():
    player, team, ages = load_raw()
    player = pd.concat([player, pd.read_parquet(CURRENT_DIR / "player_logs.parquet")], ignore_index=True)
    team = pd.concat([team, pd.read_parquet(CURRENT_DIR / "team_logs.parquet")], ignore_index=True)
    ages = pd.concat([ages, pd.read_parquet(CURRENT_DIR / "player_ages.parquet")], ignore_index=True)
    return player, team, ages


def next_game(team_abbr):
    """The team's next not-yet-played game: (date, opponent, is_home)."""
    s = pd.read_parquet(CURRENT_DIR / "schedule.parquet")
    s["date"] = pd.to_datetime(s["gameDateEst"]).dt.date
    s = s[(s["gameStatus"] == 1) & (s["date"] >= Date.today())
          & ((s["homeTeam_teamTricode"] == team_abbr) | (s["awayTeam_teamTricode"] == team_abbr))].sort_values("date")
    if s.empty:
        raise SystemExit(f"No upcoming game found for {team_abbr}. Pass --opponent and --date.")
    g = s.iloc[0]
    home = g["homeTeam_teamTricode"] == team_abbr
    return g["date"], (g["awayTeam_teamTricode"] if home else g["homeTeam_teamTricode"]), home


# =============================================================================
# Placeholder rows for the upcoming game
# =============================================================================
def find_player(player, name):
    names = player["PLAYER_NAME"].unique()
    if name in names:
        return name
    lower = {n.lower(): n for n in names}
    if name.lower() in lower:
        return lower[name.lower()]
    close = difflib.get_close_matches(name, names, n=3, cutoff=0.6)
    raise SystemExit(f"Player '{name}' not found." + (f" Did you mean: {', '.join(close)}?" if close else ""))


def expected_roster(player, team_abbr, ruled_out):
    """Players who played for this team in its last 5 games, still on the team, not ruled out."""
    cur = player[(player["SEASON"] == CURRENT_SEASON) & (player["TEAM_ABBREVIATION"] == team_abbr) & (player["MIN"] > 0)]
    last5 = cur.drop_duplicates("GAME_ID").sort_values("GAME_DATE")["GAME_ID"].tail(5)
    recent = cur[cur["GAME_ID"].isin(last5)]
    latest_team = player.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last()
    recent = recent[recent["PLAYER_ID"].map(latest_team) == team_abbr]
    out_ids = set(player.loc[player["PLAYER_NAME"].isin(ruled_out), "PLAYER_ID"])
    return recent.drop_duplicates("PLAYER_ID")[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]].query("PLAYER_ID not in @out_ids")


def placeholder_rows(player, team, game_date, team_abbr, opp_abbr, home, out, opp_out, target_id):
    team_ids = team.drop_duplicates("TEAM_ABBREVIATION").set_index("TEAM_ABBREVIATION")["TEAM_ID"]
    rows_p, rows_t = [], []
    for abbr, other, is_home, ruled_out in [(team_abbr, opp_abbr, home, out), (opp_abbr, team_abbr, not home, opp_out)]:
        matchup = f"{abbr} vs. {other}" if is_home else f"{abbr} @ {other}"
        roster = expected_roster(player, abbr, ruled_out)
        if abbr == team_abbr and target_id not in set(roster["PLAYER_ID"]):       # always include our player
            me = player[player["PLAYER_ID"] == target_id].iloc[-1]
            roster = pd.concat([roster, pd.DataFrame([{"PLAYER_ID": target_id, "PLAYER_NAME": me["PLAYER_NAME"], "TEAM_ID": team_ids[abbr]}])])
        base = {"GAME_ID": PLACEHOLDER_GAME_ID, "GAME_DATE": str(game_date), "MATCHUP": matchup,
                "SEASON": CURRENT_SEASON, "TEAM_ABBREVIATION": abbr, **{c: 0 for c in SAME_GAME_STATS}}
        for _, r in roster.iterrows():
            rows_p.append({**base, "PLAYER_ID": r["PLAYER_ID"], "PLAYER_NAME": r["PLAYER_NAME"], "TEAM_ID": team_ids[abbr], "MIN": 1})
        rows_t.append({**base, "TEAM_ID": team_ids[abbr], "MIN": 240})
    return pd.DataFrame(rows_p), pd.DataFrame(rows_t)


# =============================================================================
# The model
# =============================================================================
def load_model(retrain=False):
    if MODEL_FILE.exists() and not retrain:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    from src.evaluate import load_data
    from src.models.blend import BLEND_WEIGHTS, Blend
    print("Training the final blend on all 5 seasons (one time, ~1-2 minutes) ...")
    model = Blend(BLEND_WEIGHTS).fit(load_data())
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="SwishAI: forecast a player's next game")
    ap.add_argument("player")
    ap.add_argument("--opponent", help="opponent abbreviation, e.g. LAL (default: next game on the schedule)")
    ap.add_argument("--date", help="YYYY-MM-DD (default: next game on the schedule)")
    side = ap.add_mutually_exclusive_group()
    side.add_argument("--home", action="store_true")
    side.add_argument("--away", action="store_true")
    ap.add_argument("--out", default="", help="teammates ruled out, comma-separated")
    ap.add_argument("--opp-out", default="", help="opponent players ruled out, comma-separated")
    ap.add_argument("--line", type=float, help="a points line, e.g. 18.5, to get P(over)")
    ap.add_argument("--retrain", action="store_true", help="retrain the saved model")
    args = ap.parse_args()

    fetch_current()
    player, team, ages = load_all()
    name = find_player(player, args.player)
    me = player[player["PLAYER_NAME"] == name].sort_values("GAME_DATE")
    target_id, team_abbr = me["PLAYER_ID"].iloc[-1], me["TEAM_ABBREVIATION"].iloc[-1]

    if args.opponent:
        if not args.date:
            raise SystemExit("With --opponent, also pass --date and --home/--away.")
        game_date, opp, home = pd.Timestamp(args.date).date(), args.opponent.upper(), not args.away
    else:
        game_date, opp, home = next_game(team_abbr)

    out = [n.strip() for n in args.out.split(",") if n.strip()]
    opp_out = [n.strip() for n in args.opp_out.split(",") if n.strip()]
    for n in out + opp_out:
        find_player(player, n)                                              # typo check
    extra_p, extra_t = placeholder_rows(player, team, game_date, team_abbr, opp, home, out, opp_out, target_id)

    feats = build_features(pd.concat([player, extra_p], ignore_index=True),
                           pd.concat([team, extra_t], ignore_index=True), ages)
    row = feats[(feats["GAME_ID"] == PLACEHOLDER_GAME_ID) & (feats["PLAYER_ID"] == target_id)]

    model = load_model(args.retrain)
    point = float(model.predict(row)[0])
    sims = model.components["sim"].simulate(row, n_sims=20000)[0]
    r = row.iloc[0]
    recent = me[me["MIN"] > 0].tail(5)

    print("\n" + "=" * 62)
    print(f" SwishAI forecast: {name} ({team_abbr}) {'vs' if home else '@'} {opp}, {game_date}")
    print("=" * 62)
    print(f" Last 5 games:        {', '.join(str(p) for p in recent['PTS'][::-1])}  (most recent first)")
    print(f" Weighted average:    {r['PTS_EWM20']:.1f} pts in {r['MIN_L10']:.1f} min (last 10)")
    print(f" Teammates out:       {r['TEAMMATES_OUT_PTS']:.1f} ppg worth" + (f"  ({', '.join(out)} ruled out)" if out else ""))
    print(f" Opponent defense:    {r['OPP_DEF_RTG']:.1f} pts allowed / 100 possessions")
    print("-" * 62)
    print(f" PREDICTION:          {point:.1f} points")
    print(f" 80% range:           {np.percentile(sims, 10):.0f} - {np.percentile(sims, 90):.0f} points")
    for t in [10, 15, 20, 25, 30]:
        print(f"   P({t}+ points)      {np.mean(sims >= t):5.1%}")
    if args.line is not None:
        print(f" P(over {args.line}):        {np.mean(sims > args.line):5.1%}")
    if r["GAMES_PLAYED_BEFORE"] < 5:
        print(" NOTE: fewer than 5 career games in our data - treat with extra caution.")
    print("=" * 62)


if __name__ == "__main__":
    main()
