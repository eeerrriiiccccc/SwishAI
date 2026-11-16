"""
Phase 2 - Feature engineering.

Turns the raw game logs into ONE table that every model will learn from:
one row per player per game, with
  - FEATURES: things we could know BEFORE tip-off (inputs to the model)
  - TARGET:   PTS, the points he actually scored (what we predict)

THE GOLDEN RULE: every feature only uses games played BEFORE the current one.
The trick that makes this work is `.shift(1)`: it moves each player's stats
down by one row, so row N only "sees" games 1..N-1.

Run it with:
    .venv\\Scripts\\python.exe -m src.features
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

# The raw box-score columns that describe WHAT HAPPENED in a game.
# None of these may ever be used directly as a feature for that same game.
SAME_GAME_STATS = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
                   "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


# =============================================================================
# Small helper functions
# =============================================================================
def past_mean(df, col, by, window=None):
    """
    Average of `col` over a player's PREVIOUS games (never the current one).

    window=None -> average of ALL previous games in the group (e.g. season so far)
    window=10   -> average of the previous 10 games
    """
    shifted = df.groupby(by)[col].shift(1)              # <- the anti-cheating step
    grouped = shifted.groupby([df[b] for b in by])
    rolled = grouped.expanding() if window is None else grouped.rolling(window, min_periods=1)
    return rolled.mean().reset_index(level=list(range(len(by))), drop=True)


def past_sum(df, col, by, window):
    """Sum of `col` over the previous `window` games (used for ratios like FT%)."""
    shifted = df.groupby(by)[col].shift(1)
    grouped = shifted.groupby([df[b] for b in by])
    return grouped.rolling(window, min_periods=1).sum().reset_index(level=list(range(len(by))), drop=True)


def safe_divide(a, b):
    """a / b, but gives NaN instead of crashing or infinity when b is 0."""
    return a / b.replace(0, np.nan)


# =============================================================================
# Step 1: load + clean
# =============================================================================
def load_raw():
    player = pd.concat([pd.read_parquet(f) for f in sorted(RAW_DIR.glob("player_logs_*.parquet"))], ignore_index=True)
    team = pd.concat([pd.read_parquet(f) for f in sorted(RAW_DIR.glob("team_logs_*.parquet"))], ignore_index=True)
    ages = pd.concat([pd.read_parquet(f) for f in sorted(RAW_DIR.glob("player_ages_*.parquet"))], ignore_index=True)
    return player, team, ages


# =============================================================================
# Step 2: team-level features (team strength, opponent defense, pace...)
# =============================================================================
def build_team_table(team):
    """
    One row per team per game, with each team's PRE-GAME ratings.
    Ratings use an exponentially weighted average of previous games:
    recent games count more, older games fade out (but still count a bit, so
    early-season games can lean on last season).
    """
    t = team.copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"])

    # Attach the opponent's box score to each team row (same GAME_ID, other team).
    opp_cols = {"TEAM_ID": "OPP_TEAM_ID", "PTS": "O_PTS", "FGA": "O_FGA", "FG3A": "O_FG3A",
                "FTA": "O_FTA", "OREB": "O_OREB", "TOV": "O_TOV"}
    opp = t[list(opp_cols)].rename(columns=opp_cols).assign(GAME_ID=t["GAME_ID"])
    t = t.merge(opp, on="GAME_ID")
    t = t[t["TEAM_ID"] != t["OPP_TEAM_ID"]].copy()

    # Possessions: standard estimate, averaged over both teams.
    own_poss = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    opp_poss = t["O_FGA"] + 0.44 * t["O_FTA"] - t["O_OREB"] + t["O_TOV"]
    t["POSS"] = (own_poss + opp_poss) / 2

    # This game's ratings (these are SAME-GAME numbers - only used after shifting).
    t["g_off_rtg"] = 100 * t["PTS"] / t["POSS"]                # points scored per 100 possessions
    t["g_def_rtg"] = 100 * t["O_PTS"] / t["POSS"]              # points allowed per 100 possessions
    t["g_pace"] = t["POSS"] * 48 / (t["MIN"] / 5)              # possessions per 48 min (fixes overtime)
    t["g_3pa_rate_allowed"] = t["O_FG3A"] / t["O_FGA"]         # share of opponent shots that are 3s
    t["g_fta_rate_allowed"] = t["O_FTA"] / t["O_FGA"]          # free throws the defense gives up

    # Needed later for usage rate.
    t["TEAM_PLAYS"] = t["FGA"] + 0.44 * t["FTA"] + t["TOV"]
    t["TEAM_MIN"] = t["MIN"]

    # PRE-GAME versions: shift(1) then exponentially weighted average.
    t = t.sort_values(["TEAM_ID", "GAME_DATE"])
    rename = {"g_off_rtg": "OFF_RTG", "g_def_rtg": "DEF_RTG", "g_pace": "PACE",
              "g_3pa_rate_allowed": "3PA_RATE_ALLOWED", "g_fta_rate_allowed": "FTA_RATE_ALLOWED"}
    for raw, name in rename.items():
        t[name] = t.groupby("TEAM_ID")[raw].transform(lambda s: s.shift(1).ewm(span=15, min_periods=3).mean())
    t["NET_RTG"] = t["OFF_RTG"] - t["DEF_RTG"]

    # Which game of the season is this for the team (0, 1, 2, ...)?
    t["TEAM_GAME_NUM"] = t.groupby(["TEAM_ID", "SEASON"]).cumcount()
    # Days off for the TEAM before this game (the schedule is known in advance).
    t["TEAM_REST_DAYS"] = (t.groupby("TEAM_ID")["GAME_DATE"].diff().dt.days - 1).clip(upper=10).fillna(10)

    keep = (["GAME_ID", "TEAM_ID", "OPP_TEAM_ID", "TEAM_GAME_NUM", "TEAM_REST_DAYS", "TEAM_PLAYS", "TEAM_MIN"]
            + list(rename.values()) + ["NET_RTG"])
    return t[keep]


# =============================================================================
# Step 3: "who is missing tonight?" (teammates and opponents)
# =============================================================================
def build_missing_players(p, team_table):
    """
    For each team-game: how many points and minutes are missing because
    regular players aren't playing?

    "Regular" = appeared for this team in at least one of its previous 5 games.
    We value each missing player by his average over his last 10 games.
    (NBA teams announce inactive players before tip-off, so this is fair to use.)
    """
    # Each player's 10-game average *after* each game (including it). We'll only
    # ever look these up for LATER games, so there's no leakage.
    p = p.copy()
    p["pts_avg_after"] = p.groupby("PLAYER_ID")["PTS"].transform(lambda s: s.rolling(10, min_periods=1).mean())
    p["min_avg_after"] = p.groupby("PLAYER_ID")["MIN"].transform(lambda s: s.rolling(10, min_periods=1).mean())

    out = []
    for (team_id, season), g in p.groupby(["TEAM_ID", "SEASON"]):
        n_games = g["TEAM_GAME_NUM"].max() + 1
        full = range(n_games)
        # Table: rows = team's games in order, columns = players, values = their average (NaN = didn't play).
        pts = g.pivot_table(index="TEAM_GAME_NUM", columns="PLAYER_ID", values="pts_avg_after").reindex(full)
        mins = g.pivot_table(index="TEAM_GAME_NUM", columns="PLAYER_ID", values="min_avg_after").reindex(full)
        played = pts.notna()

        played_recently = played.shift(1, fill_value=False).astype(int).rolling(5, min_periods=1).max() > 0
        missing = played_recently & ~played                      # was around lately, but not tonight
        last_known_pts = pts.ffill().shift(1)                    # his average as of his last game
        last_known_min = mins.ffill().shift(1)

        out.append(pd.DataFrame({
            "TEAM_ID": team_id, "SEASON": season, "TEAM_GAME_NUM": list(full),
            "MISSING_PTS": last_known_pts.where(missing).sum(axis=1).values,
            "MISSING_MIN": last_known_min.where(missing).sum(axis=1).values,
        }))
    missing = pd.concat(out, ignore_index=True)
    ids = team_table[["GAME_ID", "TEAM_ID", "TEAM_GAME_NUM"]]
    # Merge game ids back in (TEAM_GAME_NUM is per team-season; SEASON comes from p).
    seasons = p[["GAME_ID", "SEASON"]].drop_duplicates()
    ids = ids.merge(seasons, on="GAME_ID")
    return ids.merge(missing, on=["TEAM_ID", "SEASON", "TEAM_GAME_NUM"])[["GAME_ID", "TEAM_ID", "MISSING_PTS", "MISSING_MIN"]]


# =============================================================================
# Step 4: put it all together
# =============================================================================
def build_features(player_raw, team_raw, ages):
    p = player_raw.copy()
    p = p[p["MIN"] > 0]                                          # drop "didn't play" rows (see EDA)
    p["GAME_DATE"] = pd.to_datetime(p["GAME_DATE"])
    p["HOME"] = p["MATCHUP"].str.contains("vs.").astype(int)
    p["OPPONENT"] = p["MATCHUP"].str.split().str[-1]
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    team_table = build_team_table(team_raw)
    p = p.merge(team_table[["GAME_ID", "TEAM_ID", "OPP_TEAM_ID", "TEAM_GAME_NUM", "TEAM_REST_DAYS", "TEAM_PLAYS", "TEAM_MIN"]],
                on=["GAME_ID", "TEAM_ID"], how="left")
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    player, season = ["PLAYER_ID"], ["PLAYER_ID", "SEASON"]

    # --- 1. Recent form and long-run averages ----------------------------------
    for stat in ["PTS", "MIN", "FGA", "FG3A", "FTA"]:
        for w in [3, 5, 10, 30]:
            p[f"{stat}_L{w}"] = past_mean(p, stat, player, w)
        # Season so far; on a player's first game of a season, fall back to his last 10.
        p[f"{stat}_SEASON"] = past_mean(p, stat, season).fillna(p[f"{stat}_L10"])
    p["PTS_STD_L10"] = p.groupby("PLAYER_ID")["PTS"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    # Exponentially weighted averages: every past game counts, but recent ones count more.
    # Smaller span = forgets faster (more "recent form"), bigger span = more "long-run".
    for span in [5, 10, 20]:
        p[f"PTS_EWM{span}"] = p.groupby("PLAYER_ID")["PTS"].transform(lambda s: s.shift(1).ewm(span=span).mean())

    # --- 2. Scoring skill and shot profile (ratios of sums over last 20 games) ---
    s = {c: past_sum(p, c, player, 20) for c in ["PTS", "FGA", "FG3A", "FTA", "FTM", "MIN"]}
    p["TS_PCT_L20"] = safe_divide(s["PTS"], 2 * (s["FGA"] + 0.44 * s["FTA"]))   # true shooting %
    p["FT_PCT_L20"] = safe_divide(s["FTM"], s["FTA"])
    p["FG3_RATE_L20"] = safe_divide(s["FG3A"], s["FGA"])                        # share of shots that are 3s
    p["FT_RATE_L20"] = safe_divide(s["FTA"], s["FGA"])                          # free throws per shot
    p["PTS_PER_MIN_L20"] = safe_divide(s["PTS"], s["MIN"])

    # Usage rate: his plays / team plays while he's on the floor, over last 10 games.
    p["plays"] = p["FGA"] + 0.44 * p["FTA"] + p["TOV"]
    p["team_plays_on_floor"] = p["TEAM_PLAYS"] * p["MIN"] / (p["TEAM_MIN"] / 5)
    p["USG_L10"] = safe_divide(past_sum(p, "plays", player, 10), past_sum(p, "team_plays_on_floor", player, 10))

    # --- 2b. Shooting profile for the Phase 8 simulation (last 50 games) --------
    # Shot attempts PER MINUTE and make rates for 2s, 3s and free throws.
    # Make rates are SHRUNK toward the league average: a player who went 4-for-5
    # from three is not really an 80% shooter. We pretend he also took K extra
    # "league-average" shots. Few real attempts -> estimate stays near the league
    # average; many attempts -> his own record dominates. (A simple Bayesian idea.)
    p["FG2M"], p["FG2A"] = p["FGM"] - p["FG3M"], p["FGA"] - p["FG3A"]
    t = {c: past_sum(p, c, player, 50) for c in ["MIN", "FG2M", "FG2A", "FG3M", "FG3A", "FTM", "FTA"]}
    p["SIM_2PA_PER_MIN"] = safe_divide(t["FG2A"], t["MIN"])
    p["SIM_3PA_PER_MIN"] = safe_divide(t["FG3A"], t["MIN"])
    p["SIM_FTA_PER_MIN"] = safe_divide(t["FTA"], t["MIN"])
    for shot, league_pct, k in [("2", 0.53, 50), ("3", 0.36, 150), ("FT", 0.78, 30)]:
        makes, attempts = (t["FTM"], t["FTA"]) if shot == "FT" else (t[f"FG{shot}M"], t[f"FG{shot}A"])
        name = "SIM_FT_PCT" if shot == "FT" else f"SIM_{shot}P_PCT"
        p[name] = ((makes.fillna(0) + k * league_pct) / (attempts.fillna(0) + k))

    # --- 3. Minutes-related: role, fouls, rest, absences ------------------------
    p["PF_PER36_L10"] = 36 * safe_divide(past_sum(p, "PF", player, 10), past_sum(p, "MIN", player, 10))
    # Role proxy: rank among tonight's teammates by recent minutes (1 = biggest role).
    p["TEAM_MIN_RANK"] = p.groupby(["GAME_ID", "TEAM_ID"])["MIN_L10"].rank(ascending=False, method="first")
    p["TEAM_PTS_RANK"] = p.groupby(["GAME_ID", "TEAM_ID"])["PTS_L10"].rank(ascending=False, method="first")

    p["REST_DAYS"] = (p.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days - 1).clip(upper=10).fillna(10)
    p["BACK_TO_BACK"] = (p["REST_DAYS"] == 0).astype(int)

    prev_team_game = p.groupby(["PLAYER_ID", "TEAM_ID", "SEASON"])["TEAM_GAME_NUM"].shift(1)
    p["TEAM_GAMES_MISSED"] = (p["TEAM_GAME_NUM"] - prev_team_game - 1).fillna(p["TEAM_GAME_NUM"]).clip(upper=20)

    p["GAMES_WITH_TEAM"] = p.groupby(["PLAYER_ID", "TEAM_ID"]).cumcount()       # small = new (trade/signing)
    p["GAMES_PLAYED_BEFORE"] = p.groupby("PLAYER_ID").cumcount()                # how much history we have

    # --- 3b. Season timing, role changes and returns (version 2, added after the daily review) ----
    # The review of 2026-27 found the model too high early in the season: teams use deeper
    # rotations in the first weeks, and the model had no way to know "it is early" or
    # "this long rest was HIS absence, not the team's schedule". These 8 features cut the
    # error on that season's unseen games from 4.445 to 4.415 (and 4.458 -> 4.451 on 2024-25).
    # Two league-wide "scoring climate" features were also tried and dropped: no extra gain.
    p["SEASON_GAME_NUM"] = p.groupby(season).cumcount()                        # his games so far THIS season
    p["MIN_LAST"] = p.groupby("PLAYER_ID")["MIN"].shift(1)                     # minutes in his previous game
    p["MIN_TREND"] = p["MIN_L3"] - p["MIN_L10"]                                # role growing (+) or shrinking (-)
    # Rest beyond the team's own days off = games HE sat out (injury, rest, coach's decision).
    p["PLAYER_EXTRA_REST"] = (p["REST_DAYS"] - p["TEAM_REST_DAYS"]).clip(lower=0, upper=10)
    # Games played since his last absence of 3+ team games (0 = first game back; 10 = long ago / never).
    came_back = p["TEAM_GAMES_MISSED"] >= 3
    spell = came_back.groupby(p["PLAYER_ID"]).cumsum()
    p["GAMES_SINCE_RETURN"] = p.groupby(["PLAYER_ID", spell]).cumcount().clip(upper=10).where(spell > 0, 10)
    # How many players his team used per game lately (deep rotation = fewer minutes each).
    used = (p.groupby(["TEAM_ID", "GAME_ID"]).agg(n_used=("PLAYER_ID", "size"), date=("GAME_DATE", "first"))
            .reset_index().sort_values(["TEAM_ID", "date"]))
    used["TEAM_PLAYERS_USED_L5"] = used.groupby("TEAM_ID")["n_used"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    p = p.merge(used[["TEAM_ID", "GAME_ID", "TEAM_PLAYERS_USED_L5"]], on=["TEAM_ID", "GAME_ID"], how="left")
    p = p.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # --- 4. Age -----------------------------------------------------------------
    p = p.merge(ages.drop_duplicates(["PLAYER_ID", "SEASON"]), on=["PLAYER_ID", "SEASON"], how="left")

    # --- 5. Team strength and opponent ---------------------------------------
    rating_cols = ["OFF_RTG", "DEF_RTG", "PACE", "NET_RTG", "3PA_RATE_ALLOWED", "FTA_RATE_ALLOWED"]
    own = team_table[["GAME_ID", "TEAM_ID", "OFF_RTG", "PACE", "NET_RTG"]].rename(
        columns={"OFF_RTG": "TEAM_OFF_RTG", "PACE": "TEAM_PACE", "NET_RTG": "TEAM_NET_RTG"})
    opp = team_table[["GAME_ID", "TEAM_ID"] + rating_cols].rename(
        columns={"TEAM_ID": "OPP_TEAM_ID", **{c: f"OPP_{c}" for c in rating_cols}})
    p = p.merge(own, on=["GAME_ID", "TEAM_ID"], how="left").merge(opp, on=["GAME_ID", "OPP_TEAM_ID"], how="left")
    # Expected strength gap: a big gap (either way) raises the chance of a blowout -> starters sit.
    p["NET_RTG_GAP"] = p["TEAM_NET_RTG"] - p["OPP_NET_RTG"]
    p["ABS_NET_RTG_GAP"] = p["NET_RTG_GAP"].abs()

    # --- 6. Who's missing tonight ---------------------------------------------
    miss = build_missing_players(p, team_table)
    p = p.merge(miss.rename(columns={"MISSING_PTS": "TEAMMATES_OUT_PTS", "MISSING_MIN": "TEAMMATES_OUT_MIN"}),
                on=["GAME_ID", "TEAM_ID"], how="left")
    p = p.merge(miss.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "MISSING_PTS": "OPP_PLAYERS_OUT_PTS",
                                     "MISSING_MIN": "OPP_PLAYERS_OUT_MIN"}),
                on=["GAME_ID", "OPP_TEAM_ID"], how="left")

    p = p.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"]).reset_index(drop=True)
    return p[ID_COLS + FEATURES + SIM_COLS + TARGET_COLS]


# =============================================================================
# The official column lists - models import these so everyone uses the same set.
# =============================================================================
ID_COLS = ["SEASON", "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT"]

FEATURES = (
    [f"{s}_L{w}" for s in ["PTS", "MIN", "FGA", "FG3A", "FTA"] for w in [3, 5, 10, 30]]
    + [f"{s}_SEASON" for s in ["PTS", "MIN", "FGA", "FG3A", "FTA"]]
    + ["PTS_EWM5", "PTS_EWM10", "PTS_EWM20"]
    + ["PTS_STD_L10", "TS_PCT_L20", "FT_PCT_L20", "FG3_RATE_L20", "FT_RATE_L20", "PTS_PER_MIN_L20", "USG_L10",
       "PF_PER36_L10", "TEAM_MIN_RANK", "TEAM_PTS_RANK", "REST_DAYS", "BACK_TO_BACK", "TEAM_GAMES_MISSED",
       "GAMES_WITH_TEAM", "GAMES_PLAYED_BEFORE", "AGE", "HOME",
       "TEAM_OFF_RTG", "TEAM_PACE", "TEAM_NET_RTG",
       "OPP_DEF_RTG", "OPP_OFF_RTG", "OPP_PACE", "OPP_NET_RTG", "OPP_3PA_RATE_ALLOWED", "OPP_FTA_RATE_ALLOWED",
       "NET_RTG_GAP", "ABS_NET_RTG_GAP",
       "TEAMMATES_OUT_PTS", "TEAMMATES_OUT_MIN", "OPP_PLAYERS_OUT_PTS", "OPP_PLAYERS_OUT_MIN"]
    # Version 2 (see section 3b in build_features):
    + ["SEASON_GAME_NUM", "TEAM_GAME_NUM", "MIN_LAST", "MIN_TREND", "TEAM_REST_DAYS", "PLAYER_EXTRA_REST",
       "GAMES_SINCE_RETURN", "TEAM_PLAYERS_USED_L5"]
)

# Inputs for the Phase 8 simulation only (kept out of FEATURES so earlier models are unchanged).
SIM_COLS = ["SIM_2PA_PER_MIN", "SIM_3PA_PER_MIN", "SIM_FTA_PER_MIN", "SIM_2P_PCT", "SIM_3P_PCT", "SIM_FT_PCT"]

# What actually happened. PTS is our target; MIN is kept because later we may
# build a separate "how many minutes will he play" model.
TARGET_COLS = ["PTS", "MIN"]


def main():
    player, team, ages = load_raw()
    print("Building features (takes ~1 minute) ...")
    df = build_features(player, team, ages)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out, index=False)

    print(f"\nSaved {len(df):,} rows x {len(FEATURES)} features -> {out}")
    missing = df[FEATURES].isna().mean().sort_values(ascending=False)
    print("\nFeatures with the most blanks (mostly a player's first few games):")
    print((missing[missing > 0].head(8) * 100).round(1).astype(str) + "%")


if __name__ == "__main__":
    main()
