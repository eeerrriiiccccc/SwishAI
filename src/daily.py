"""
Phase 11 - The daily loop: predict, grade, learn. Runs by itself.

Each run does the same six steps, and is safe to run any number of times a day:

    1. DOWNLOAD   this season's box scores, the schedule and the injury report.
    2. GRADE      every earlier prediction whose game has finished (attach the real points).
    3. FILL IN    games that finished while the computer was off: predicted after the
                  fact, but ONLY from pre-game data (the leakage test guarantees that).
                  They are marked "backfill" so they are never confused with live picks.
    4. LEARN      src/review.py compares the models, tests whether the blend weights
                  should move, and looks for the reasons behind the misses.
    5. PREDICT    every player expected to play on the next game day, before tip-off.
    6. REPORT     tracking/REPORT.md, in plain language, with the reasoning for each action.

WHO do we predict? Everyone expected to play (with 5+ games of history), about 150-250
players on a full night. Not a hand-picked few, for three reasons:
    * It costs nothing extra: the slow part (building features) is done once for all.
    * Learning speed depends on sample size. Telling a 0.3-point lean from luck takes
      roughly 1,500 graded games: one week of everyone, or months of 10 stars.
    * A hand-picked group (say, stars only) would hide problems elsewhere (bench, rookies).
The report still highlights the top expected scorers, so it stays readable.

Files (all in tracking/):
    predictions.csv   one row per player per game: every model's guess, then the result
    decisions.csv     every automatic change, with the reason
    state.json        blend weights in use
    REPORT.md         the latest report
    logs/daily.log    what each run did

Run by hand:   .venv\\Scripts\\python.exe -m src.daily            (add --force to redo everything)
Retrain:       .venv\\Scripts\\python.exe -m src.daily --retrain  (only when the report recommends it)
"""
import argparse
import json
import pickle
import re
import shutil
import sys
import time
import traceback
import unicodedata
import urllib.request
import warnings

import numpy as np
import pandas as pd

from src import review
from src.config import CURRENT_DIR, CURRENT_SEASON, MODELS_DIR, TRACK_DIR
from src.evaluate import MIN_GAMES
from src.features import SAME_GAME_STATS, SIM_COLS, build_features
from src.models.blend import BLEND_WEIGHTS
from src.predict import MODEL_FILE, fetch_current, load_all, load_model

PRED_FILE = TRACK_DIR / "predictions.csv"
STATE_FILE = TRACK_DIR / "state.json"
DECISIONS_FILE = TRACK_DIR / "decisions.csv"
REPORT_FILE = TRACK_DIR / "REPORT.md"
LOG_FILE = TRACK_DIR / "logs" / "daily.log"
LOCK_FILE = TRACK_DIR / ".lock"
INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# Pre-game facts saved next to each prediction, so the review can group misses later.
SNAPSHOT = ["PTS_SEASON", "PTS_EWM20", "MIN_L10", "GAMES_PLAYED_BEFORE", "AGE", "HOME", "REST_DAYS", "BACK_TO_BACK",
            "TEAM_GAMES_MISSED", "TEAMMATES_OUT_PTS", "OPP_PLAYERS_OUT_PTS", "OPP_DEF_RTG", "ABS_NET_RTG_GAP"]
PRED_COLS = ["p_baseline", "p_ridge", "p_hybrid", "p_nn", "p_sim", "p_blend_static", "p_blend", "q10", "q90",
             "exp_min", "exp_pts_per_tsa"]
ACTUAL_COLS = ["PTS", "MIN", "FGA", "FTA", "actual_TEAMMATES_OUT_PTS"]
COLUMNS = (["GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT", "status", "source",
            "predicted_at", "model_version"] + PRED_COLS + SNAPSHOT + ACTUAL_COLS)


def log(message):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}  {message}"
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if sys.stdout is not None:                    # no console when started by the scheduler
        print(line)


# =============================================================================
# Saved files
# =============================================================================
def load_predictions():
    if PRED_FILE.exists():
        return pd.read_csv(PRED_FILE, dtype={"GAME_ID": str, "GAME_DATE": str, "status": str, "source": str})
    return pd.DataFrame(columns=COLUMNS)


def save_predictions(preds):
    preds = preds[COLUMNS].sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION", "PLAYER_NAME"])
    preds.to_csv(PRED_FILE, index=False, float_format="%.3f")


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"policy": "static", "weights": dict(BLEND_WEIGHTS), "offset": 0.0,
            "model_version": "v2: 68 features, trained on 2021-22 to 2025-26", "model_cutoff": "2026-06-30"}


def append_decisions(decisions):
    if decisions:
        pd.DataFrame(decisions).to_csv(DECISIONS_FILE, mode="a", index=False, header=not DECISIONS_FILE.exists())


# =============================================================================
# Who is expected to play?
# =============================================================================
def norm_name(name):
    """'Nikola Jokić' and 'nikola jokic jr.' -> 'nikola jokic' (so names from two websites match)."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    name = re.sub(r"[^a-z ]", "", name)
    return " ".join(w for w in name.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


def fetch_injuries():
    """Players listed as out, from ESPN's public injury report. Returns None if it can't be reached."""
    try:
        request = urllib.request.Request(INJURY_URL, headers={"User-Agent": "Mozilla/5.0 (SwishAI hobby project)"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
        out = set()
        for team in data.get("injuries", []):
            for item in team.get("injuries", []):
                name = (item.get("athlete") or {}).get("displayName")
                if name and item.get("status") in {"Out", "Suspension"}:
                    out.add(norm_name(name))
        return out
    except Exception as error:
        log(f"injury report unavailable ({error}); using 'played the last game' only")
        return None


def expected_roster(cur, latest_team, team_abbr, injured):
    """
    Who do we expect to play for this team?

    Checked on the 2025-26 season: a player who played his team's LAST game plays the
    next one 86% of the time; one who missed it (but played in the last 5) only 37%.
    So: played the last game, and not on the injury report. If the injury report loaded,
    we also bring back players who missed the last game but played 3+ of the last 5 and
    are no longer listed as out (most likely returning from a short absence).
    """
    team = cur[(cur["TEAM_ABBREVIATION"] == team_abbr)]
    games = team.drop_duplicates("GAME_ID").sort_values("GAME_DATE")["GAME_ID"].tail(5).tolist()
    if not games:
        return team.head(0), []
    recent = team[team["GAME_ID"].isin(games)]
    recent = recent[recent["PLAYER_ID"].map(latest_team) == team_abbr]            # still on this team
    n_played = recent.groupby("PLAYER_ID")["GAME_ID"].nunique()
    played_last = set(recent.loc[recent["GAME_ID"] == games[-1], "PLAYER_ID"])
    roster = recent.drop_duplicates("PLAYER_ID")[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]]
    keep = roster["PLAYER_ID"].isin(played_last)
    if injured is not None:
        keep |= roster["PLAYER_ID"].map(n_played) >= 3
        is_out = roster["PLAYER_NAME"].map(norm_name).isin(injured)
        left_out = roster.loc[keep & is_out, "PLAYER_NAME"].tolist()
        keep &= ~is_out
    else:
        left_out = []
    return roster[keep], left_out


def placeholder_rows(player, team, games, injured):
    """Blank rows for every expected player in every upcoming game (stats unknown = 0, never used)."""
    cur = player[(player["SEASON"] == CURRENT_SEASON) & (player["MIN"] > 0)]
    latest_team = player.sort_values("GAME_DATE").groupby("PLAYER_ID")["TEAM_ABBREVIATION"].last()
    team_ids = team.drop_duplicates("TEAM_ABBREVIATION").set_index("TEAM_ABBREVIATION")["TEAM_ID"]
    rows_p, rows_t, assumed_out = [], [], []
    for g in games.itertuples():
        for abbr, other, home in [(g.home, g.away, True), (g.away, g.home, False)]:
            roster, left_out = expected_roster(cur, latest_team, abbr, injured)
            assumed_out += left_out
            base = {"GAME_ID": g.gameId, "GAME_DATE": g.date, "MATCHUP": f"{abbr} vs. {other}" if home else f"{abbr} @ {other}",
                    "SEASON": CURRENT_SEASON, "TEAM_ABBREVIATION": abbr, **{c: 0 for c in SAME_GAME_STATS}}
            rows_p += [{**base, "PLAYER_ID": r.PLAYER_ID, "PLAYER_NAME": r.PLAYER_NAME, "TEAM_ID": team_ids[abbr], "MIN": 1}
                       for r in roster.itertuples()]
            rows_t.append({**base, "TEAM_ID": team_ids[abbr], "MIN": 240})
    return pd.DataFrame(rows_p), pd.DataFrame(rows_t), sorted(set(assumed_out))


# =============================================================================
# Predicting and grading
# =============================================================================
def predict_rows(model, rows, state, source, now):
    """Every model's guess for these feature rows, plus the pre-game snapshot."""
    hybrid, nn, sim = (model.components[k] for k in ("hybrid", "nn", "sim"))
    out = rows[["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "OPPONENT"] + SNAPSHOT].copy()
    out["GAME_DATE"] = pd.to_datetime(rows["GAME_DATE"]).dt.strftime("%Y-%m-%d").to_numpy()
    out["p_baseline"] = rows["PTS_EWM20"].to_numpy()                 # no machine learning: the bar to beat
    out["p_ridge"] = hybrid.linear.predict(rows)
    out["p_hybrid"] = hybrid.predict(rows)
    out["p_nn"] = nn.predict(rows)
    cdf = np.cumsum(sim.pmf(rows, n_sims=4000, chunk=500), axis=1)   # play each game 4,000 times
    out["p_sim"] = (cdf < 0.5).sum(axis=1)
    out["q10"], out["q90"] = (cdf < 0.1).sum(axis=1), (cdf < 0.9).sum(axis=1)
    parts = out[review.COMPONENTS].to_numpy(float)
    out["p_blend_static"] = parts @ review.PRIOR
    live = source == "live"                                         # only live picks use the adaptive blend
    weights = np.array([state["weights"][k] for k in review.KEYS]) if live else review.PRIOR
    out["p_blend"] = parts @ weights
    if live:                                                        # bias corrections chosen by the review
        tier_offsets = np.array([state.get("tier_offsets", {}).get(t, 0.0) for t in review.TIER_NAMES])
        out["p_blend"] += state["offset"] + tier_offsets[review.tier_of(out["PTS_SEASON"])]
    out["exp_min"] = sim.minutes_model.predict(rows)
    s = rows[SIM_COLS].fillna(sim.sim_medians)
    out["exp_pts_per_tsa"] = ((2 * s["SIM_2PA_PER_MIN"] * s["SIM_2P_PCT"] + 3 * s["SIM_3PA_PER_MIN"] * s["SIM_3P_PCT"]
                               + s["SIM_FTA_PER_MIN"] * s["SIM_FT_PCT"])
                              / (s["SIM_2PA_PER_MIN"] + s["SIM_3PA_PER_MIN"] + 0.44 * s["SIM_FTA_PER_MIN"])).to_numpy()
    out["status"], out["source"], out["predicted_at"] = "pending", source, now
    out["model_version"] = state["model_version"]
    for col in ACTUAL_COLS:
        out[col] = np.nan
    return out[COLUMNS]


def attach_results(rows, logs, real_feats):
    """Add what really happened. A predicted player with no minutes is marked 'dnp' (did not play)."""
    actual = logs.loc[logs["MIN"] > 0, ["GAME_ID", "PLAYER_ID", "PTS", "MIN", "FGA", "FTA"]]
    truth = real_feats[["GAME_ID", "PLAYER_ID", "TEAMMATES_OUT_PTS"]].rename(
        columns={"TEAMMATES_OUT_PTS": "actual_TEAMMATES_OUT_PTS"})
    rows = (rows.drop(columns=ACTUAL_COLS)
            .merge(actual, on=["GAME_ID", "PLAYER_ID"], how="left")
            .merge(truth, on=["GAME_ID", "PLAYER_ID"], how="left"))
    rows["status"] = np.where(rows["PTS"].notna(), "graded", "dnp")
    return rows[COLUMNS]


# =============================================================================
# Retraining (only on request)
# =============================================================================
def retrain(feats, final_ids, state, today):
    from src.models.blend import Blend
    train = feats[(feats["GAMES_PLAYED_BEFORE"] >= MIN_GAMES)
                  & ((feats["SEASON"] != CURRENT_SEASON) | feats["GAME_ID"].isin(final_ids))].reset_index(drop=True)
    # Each model holds out the NEWEST season to decide when to stop training. A season that
    # is only a few weeks old is too small (and too early-season) for that job, so for
    # training we count this season's games as part of last season: the held-out set
    # becomes "last season + this season so far".
    seasons = sorted(train["SEASON"].unique())
    if len(seasons) > 1:
        train["SEASON"] = train["SEASON"].replace(CURRENT_SEASON, seasons[-2])
    log(f"retraining the blend on {len(train):,} games (a few minutes) ...")
    model = Blend(BLEND_WEIGHTS).fit(train)
    if MODEL_FILE.exists():
        shutil.copy(MODEL_FILE, MODELS_DIR / f"swishai_blend_before_{today}.pkl")     # keep the old one
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    cutoff = pd.to_datetime(train["GAME_DATE"]).max().strftime("%Y-%m-%d")
    state.update({"model_version": f"retrained {today} (games through {cutoff})", "model_cutoff": cutoff})
    return model, [{"date": today, "action": "retrain", "change": f"model now trained through {cutoff}",
                    "reasoning": "Requested with --retrain. The previous model was saved next to it as a backup."}]


# =============================================================================
# One run
# =============================================================================
def run(args):
    now = pd.Timestamp.now(tz="UTC")
    stamp = f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"
    today = stamp[:10]

    # ---- 1. download ---------------------------------------------------------
    for attempt in range(1, 4):
        try:
            fetch_current(force=True)
            break
        except Exception as error:
            log(f"download attempt {attempt} failed: {error}")
            if attempt == 3:
                log("giving up for now; the next scheduled run will try again")
                return
            time.sleep(30 * attempt)
    player, team, ages = load_all()
    sched = pd.read_parquet(CURRENT_DIR / "schedule.parquet")
    sched = sched[sched["gameId"].str.startswith("002")].copy()              # regular season only
    sched["date"] = pd.to_datetime(sched["gameDateEst"]).dt.strftime("%Y-%m-%d")
    sched["tip"] = pd.to_datetime(sched["gameDateTimeUTC"], utc=True)
    sched = sched.rename(columns={"homeTeam_teamTricode": "home", "awayTeam_teamTricode": "away"})
    logs = player[player["SEASON"] == CURRENT_SEASON]
    final_ids = set(sched.loc[sched["gameStatus"] == 3, "gameId"]) & set(logs["GAME_ID"])

    upcoming = sched[(sched["gameStatus"] == 1) & (sched["tip"] > now)]
    upcoming = upcoming[upcoming["date"] == upcoming["date"].min()]          # the next game day
    next_day = upcoming["date"].min() if len(upcoming) else None

    preds = load_predictions()
    done = set(preds.loc[preds["status"] != "pending", "GAME_ID"])
    pending = preds[preds["status"] == "pending"]
    fresh = (len(upcoming) == 0) or (
        set(upcoming["gameId"]) <= set(pending["GAME_ID"])
        and (pd.Timestamp(stamp) - pd.to_datetime(pending.loc[pending["GAME_ID"].isin(upcoming["gameId"]), "predicted_at"]).min()
             < pd.Timedelta(hours=3)))
    if not (final_ids - done) and fresh and not (args.force or args.retrain):
        log("nothing new: all finished games are graded and the next predictions are fresh")
        return

    # ---- features for everything (finished games + placeholders for the next day) ----
    injured = fetch_injuries()
    extra_p, extra_t, assumed_out = placeholder_rows(player, team, upcoming, injured) if len(upcoming) else (None, None, [])
    log(f"building features ({len(final_ids)} finished games, {len(upcoming)} upcoming on {next_day}) ...")
    feats = build_features(pd.concat([player, extra_p], ignore_index=True) if extra_p is not None else player,
                           pd.concat([team, extra_t], ignore_index=True) if extra_t is not None else team, ages)
    feats = feats[feats["GAMES_PLAYED_BEFORE"] >= MIN_GAMES]
    cur_feats = feats[feats["SEASON"] == CURRENT_SEASON]
    real = cur_feats[cur_feats["GAME_ID"].isin(final_ids)]
    state = load_state()
    decisions = []
    model = load_model()

    # ---- 2. grade pending predictions whose game is final ----------------------
    to_grade = (preds["status"] == "pending") & preds["GAME_ID"].isin(final_ids)
    if to_grade.any():
        graded_now = attach_results(preds[to_grade], logs, real)
        preds = pd.concat([preds[~to_grade], graded_now], ignore_index=True)
        log(f"graded {len(graded_now)} live predictions "
            f"({(graded_now['status'] == 'dnp').sum()} did not play)")
    # pending rows for games that never happened (postponed): stop waiting after 4 days
    stale = (preds["status"] == "pending") & (preds["GAME_DATE"] < str((pd.Timestamp(today) - pd.Timedelta(days=4)).date()))
    preds.loc[stale, "status"] = "void"

    # ---- 3. fill in finished games we have no prediction for -------------------
    have = set(zip(preds["GAME_ID"], preds["PLAYER_ID"]))
    missing = real[[k not in have for k in zip(real["GAME_ID"], real["PLAYER_ID"])]]
    if len(missing):
        live_games = set(preds.loc[preds["source"] == "live", "GAME_ID"])
        filled = attach_results(predict_rows(model, missing, state, "backfill", stamp), logs, real)
        filled["source"] = np.where(filled["GAME_ID"].isin(live_games), "unexpected", "backfill")
        preds = pd.concat([preds, filled], ignore_index=True)
        log(f"filled in {len(filled)} finished player-games "
            f"({(filled['source'] == 'unexpected').sum()} were players we did not expect to play)")

    # ---- 4. learn ----------------------------------------------------------------
    if args.retrain:
        model, d = retrain(feats, final_ids, state, today)
        decisions += d
    graded = preds[preds["status"] == "graded"]
    state, d = review.update_policy(graded, state, today)
    decisions += d
    for x in decisions:
        log(f"DECISION {x['action']}: {x['change']}")

    # ---- 5. predict the next game day ---------------------------------------------
    if len(upcoming):
        rows = cur_feats[cur_feats["GAME_ID"].isin(upcoming["gameId"])]
        new = predict_rows(model, rows, state, "live", stamp)
        preds = pd.concat([preds[~((preds["status"] == "pending") & preds["GAME_ID"].isin(upcoming["gameId"]))], new],
                          ignore_index=True)
        log(f"predicted {len(new)} players in {new['GAME_ID'].nunique()} games on {next_day}")

    # ---- 6. save + report -----------------------------------------------------------
    TRACK_DIR.mkdir(exist_ok=True)
    save_predictions(preds)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    append_decisions(decisions)
    info = {"now": stamp, "next_day": next_day, "assumed_out": assumed_out, "injuries_ok": injured is not None, "notes": []}
    REPORT_FILE.write_text(review.build_report(load_predictions(), state, decisions, info), encoding="utf-8")
    log(f"report written: {REPORT_FILE}")


def main():
    ap = argparse.ArgumentParser(description="SwishAI daily loop: predict, grade, learn")
    ap.add_argument("--force", action="store_true", help="run every step even if nothing looks new")
    ap.add_argument("--retrain", action="store_true", help="retrain the models on all games played so far")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    TRACK_DIR.mkdir(exist_ok=True)
    if LOCK_FILE.exists() and time.time() - LOCK_FILE.stat().st_mtime < 45 * 60:
        log("another run is in progress; skipping")
        return
    LOCK_FILE.write_text(str(time.time()))
    try:
        start = time.time()
        run(args)
        log(f"done in {time.time() - start:.0f}s")
    except Exception:
        log("FAILED:\n" + traceback.format_exc())
        raise
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
