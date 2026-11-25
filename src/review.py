"""
Phase 11 (part 2) - SwishAI reviews its own predictions.

src/daily.py saves every prediction BEFORE the game and the real result AFTER it.
This file reads that record and answers, with numbers:

  1. SCOREBOARD     Which model is doing better or worse this season, and is the
                    difference real or just luck?
  2. BLEND WEIGHTS  Should the 0.5 / 0.4 / 0.1 weights change? (adaptive weights)
  3. ERROR BUDGET   When we miss, WHY? Each miss is split into three parts:
                        minutes  - he played more/less than expected
                        volume   - he took more/fewer shots per minute than expected
                        shooting - his shots went in more/less often than usual (mostly luck)
  4. BLIND SPOTS    Groups of players where we are wrong in the SAME direction
                    again and again (that is fixable; random misses are not).
  5. ACTIONS        What was changed automatically, what is recommended, and why.

Everything is written to tracking/REPORT.md in plain language.

Sign convention (same as src/evaluate.py):  bias = prediction - actual.
    positive bias = we guessed too HIGH,  negative bias = we guessed too LOW.
"""
import numpy as np
import pandas as pd

from src.models.blend import BLEND_WEIGHTS

KEYS = ["hybrid", "nn", "sim"]
COMPONENTS = [f"p_{k}" for k in KEYS]
PRIOR = np.array([BLEND_WEIGHTS[k] for k in KEYS])          # the weights chosen on 2024-25

MODEL_NAMES = {
    "p_baseline": "Simple weighted average (no ML)",
    "p_ridge": "Ridge regression",
    "p_hybrid": "Hybrid: linear + LightGBM",
    "p_nn": "Neural network ensemble",
    "p_sim": "Simulation median",
    "p_blend_static": "Blend, fixed 0.5 / 0.4 / 0.1",
}

# --- Settings for the adaptive weights (see adaptive_params) -------------------
MIN_HISTORY = 500      # graded predictions needed before we even try to adapt
HALF_LIFE = 3000       # a game this many predictions ago counts half as much (~3 weeks)
K_WEIGHTS = 4000       # how strongly we trust the original weights (in "predictions worth")
K_OFFSET = 1500        # same, for the bias correction
# Recommend retraining after this many new graded games: roughly most of a season.
# Was 5,000 until 2026-11-25, when a fair check (train through Nov 17, test on Nov 18-24)
# showed retraining on 4,451 new games gave no gain (4.302 vs 4.295): that is only ~3% of the
# ~130,000 training rows, so the model barely moves.
RETRAIN_AFTER = 20000

# Every weight combination in steps of 0.05 that adds up to 1 (231 of them).
GRID = np.array([(a / 20, b / 20, (20 - a - b) / 20) for a in range(21) for b in range(21 - a)])


# =============================================================================
# 1. Adaptive blend weights
# =============================================================================
def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, cum[-1] / 2)])


TIER_NAMES = ["bench", "mid", "star"]          # season scoring average: under 10 / 10 to 20 / 20 and up


def tier_of(pts_season):
    return np.digitize(np.nan_to_num(np.asarray(pts_season, float)), [10, 20])


def adaptive_params(P, y, tier, shrink=True):
    """
    Best blend weights (and small bias corrections) given the games graded so far.

    P: (n, 3) predictions of hybrid / nn / sim, oldest first.   y: (n,) actual points.
    tier: (n,) 0 / 1 / 2 for bench / mid / star.
    Returns (weights, one offset for everyone, one offset per tier).

    Three ideas, each there for a reason:
      * RECENCY: recent games count more (half-life HALF_LIFE predictions), so if one
        model drifts as the season changes, the weights can follow.
      * SEARCH: try all 231 weight combinations, keep the one with the lowest
        (recency-weighted) error. Same idea as blend.choose_weights().
      * SHRINKAGE: the three models are within ~0.01 points of each other, so over a
        few weeks the "best" combination is mostly luck. We only move PART of the way
        from the original weights toward the new best: n / (n + K_WEIGHTS).
        Little data -> stay near 0.5/0.4/0.1.  Lots of data -> trust the new evidence.
    An offset fixes a constant lean (always a bit high or low). The offset that
    minimises absolute error is the MEDIAN leftover error, shrunk the same way.
    The per-tier offsets do the same separately for bench players, mid scorers and
    stars, because a season can run "low" for one group only (2026-27: mid scorers).
    """
    n = len(y)
    if n < MIN_HISTORY:
        return PRIOR.copy(), 0.0, np.zeros(3)
    recency = 0.5 ** (np.arange(n)[::-1] / HALF_LIFE)
    n_eff = recency.sum()
    mae = (np.abs(P @ GRID.T - y[:, None]) * recency[:, None]).sum(axis=0) / n_eff
    w_best = GRID[mae.argmin()]
    w = PRIOR + (n_eff / (n_eff + K_WEIGHTS) if shrink else 1.0) * (w_best - PRIOR)
    left = y - P @ w
    offset = weighted_median(left, recency) * (n_eff / (n_eff + K_OFFSET) if shrink else 1.0)
    tier_offsets = np.zeros(3)
    for t in range(3):
        m = tier == t
        if m.sum() >= 100:
            n_t = recency[m].sum()
            tier_offsets[t] = weighted_median(left[m], recency[m]) * (n_t / (n_t + K_OFFSET) if shrink else 1.0)
    return w, float(offset), tier_offsets


def walk_forward(g):
    """
    An honest test of the adaptive idea: replay the season one game day at a time.
    For each day, weights are computed ONLY from earlier days, then used on that day.
    (Same no-peeking rule as everything else in SwishAI.)
    """
    g = g.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"])
    P, y = g[COMPONENTS].to_numpy(float), g["PTS"].to_numpy(float)
    tier = tier_of(g["PTS_SEASON"])
    dates = np.asarray(g["GAME_DATE"].astype(str))
    out = {k: np.full(len(g), np.nan) for k in ["weights", "weights_offset", "weights_tier_offset", "no_shrink"]}
    for d in np.unique(dates):
        past, today = dates < d, dates == d
        if past.sum() < MIN_HISTORY:
            continue
        w, b, bt = adaptive_params(P[past], y[past], tier[past])
        w_raw, b_raw, _ = adaptive_params(P[past], y[past], tier[past], shrink=False)
        out["weights"][today] = P[today] @ w
        out["weights_offset"][today] = P[today] @ w + b
        out["weights_tier_offset"][today] = P[today] @ w + bt[tier[today]]
        out["no_shrink"][today] = P[today] @ w_raw + b_raw
    scored = ~np.isnan(out["weights"])
    return pd.DataFrame({"y": y, "static": P @ PRIOR, **out}, index=g.index)[scored]


def beaten(pred, actual):
    """Share of games where he scored MORE than predicted (an exact hit counts as half)."""
    return float((actual > pred).mean() + 0.5 * (actual == pred).mean())


def paired(err_a, err_b):
    """Average of |error A| - |error B| and its standard error. Negative = A is better."""
    d = np.abs(err_a) - np.abs(err_b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan


def update_policy(graded, state, today):
    """
    Decide how tomorrow's blend is made. Returns (state, list of decisions).

    POLICY GATE: adaptive weights are only switched on if the walk-forward replay
    shows they would have beaten the fixed weights by more than 1 standard error.
    Otherwise we keep the fixed weights: a change that can't prove itself isn't made.
    """
    decisions = []
    if len(graded) == state.get("n_graded_at_update"):
        return state, decisions                                   # nothing new to learn from
    g = graded.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"])
    wf = walk_forward(g)
    policy, why, table = "static", "", []
    if len(wf) < 1000:
        why = (f"Only {len(wf):,} predictions can be replayed so far (need 1,000). "
               "Too few to tell skill from luck, so the fixed weights stay.")
    else:
        base = wf["static"] - wf["y"]
        table.append({"policy": "Fixed weights 0.5/0.4/0.1", "MAE": np.abs(base).mean(), "vs fixed": 0.0, "SE": 0.0})
        results = {}
        for name, label in [("weights", "Adaptive weights"), ("weights_offset", "Adaptive weights + one bias fix"),
                            ("weights_tier_offset", "Adaptive weights + bias fix per tier"),
                            ("no_shrink", "Adaptive, NO shrinkage (for comparison)")]:
            diff, se = paired(wf[name] - wf["y"], base)
            results[name] = (diff, se)
            table.append({"policy": label, "MAE": np.abs(wf[name] - wf["y"]).mean(), "vs fixed": diff, "SE": se})
        cand = min(["weights", "weights_offset", "weights_tier_offset"], key=lambda k: results[k][0])
        diff, se = results[cand]
        if diff < -se:
            policy = cand
            why = (f"Replaying {len(wf):,} predictions day by day, '{cand.replace('_', ' + ')}' beat the fixed weights "
                   f"by {-diff:.4f} points per game (uncertainty +/-{se:.4f}). The gain is bigger than its "
                   "uncertainty, so it is switched on.")
        else:
            why = (f"Replaying {len(wf):,} predictions day by day, the best adaptive option changed the error by "
                   f"{diff:+.4f} points per game (uncertainty +/-{se:.4f}). That is not a clear win, "
                   "so the fixed weights stay. No change is made without evidence.")

    w, b, bt = adaptive_params(g[COMPONENTS].to_numpy(float), g["PTS"].to_numpy(float), tier_of(g["PTS_SEASON"]))
    new_w = PRIOR if policy == "static" else w
    new_b = b if policy == "weights_offset" else 0.0
    new_bt = bt if policy == "weights_tier_offset" else np.zeros(3)
    old_w = np.array([state["weights"][k] for k in KEYS])
    old_bt = np.array([state.get("tier_offsets", {}).get(t, 0.0) for t in TIER_NAMES])
    if (policy != state.get("policy") or np.abs(new_w - old_w).max() >= 0.01
            or abs(new_b - state.get("offset", 0)) >= 0.05 or np.abs(new_bt - old_bt).max() >= 0.05):
        show = lambda v: "/".join(f"{x:+.2f}" for x in v)
        decisions.append({
            "date": today, "action": "blend policy",
            "change": (f"{state.get('policy')} -> {policy}; weights "
                       + "/".join(f"{x:.2f}" for x in old_w) + " -> " + "/".join(f"{x:.2f}" for x in new_w)
                       + f"; offset {state.get('offset', 0):+.2f} -> {new_b:+.2f}"
                       + f"; bench/mid/star offsets {show(old_bt)} -> {show(new_bt)}"),
            "reasoning": why})
    state.update({
        "policy": policy, "policy_reason": why,
        "weights": {k: round(float(x), 4) for k, x in zip(KEYS, new_w)},
        "offset": round(float(new_b), 3),
        "tier_offsets": {t: round(float(x), 3) for t, x in zip(TIER_NAMES, new_bt)},
        "suggested_weights": {k: round(float(x), 4) for k, x in zip(KEYS, w)}, "suggested_offset": round(b, 3),
        "suggested_tier_offsets": {t: round(float(x), 3) for t, x in zip(TIER_NAMES, bt)},
        "walk_forward": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in table],
        "n_graded_at_update": len(graded), "updated_on": today})
    return state, decisions


# =============================================================================
# 2. Scoreboard: which model is better / worse?
# =============================================================================
def scoreboard(g, models):
    base = g["p_blend_static"] - g["PTS"]
    rows = []
    for col in models:
        err = g[col] - g["PTS"]
        diff, se = paired(err, base)
        if col == "p_blend_static":
            verdict = "the reference"
        elif diff > 2 * se:
            verdict = "worse than the blend"
        elif diff < -2 * se:
            verdict = "BETTER than the blend"
        else:
            verdict = "no real difference yet"
        rows.append({"Model": MODEL_NAMES[col], "MAE": err.abs().mean(), "Bias": err.mean(),
                     "Beaten": beaten(g[col], g["PTS"]), "vs blend": diff, "+/-": 2 * se, "Verdict": verdict})
    return pd.DataFrame(rows)


# =============================================================================
# 3. Error budget: minutes / shot volume / shooting
# =============================================================================
def decompose(g, pred_col="p_blend_static"):
    """
    Split every miss (actual - prediction) into three parts that add up exactly.

        prediction = expected minutes x expected points per minute
        actual     = real minutes     x real points per minute

      minutes part  = (real minutes - expected minutes) x expected points per minute
      volume part   = extra shots per minute x his usual points per shot
      shooting part = what is left: his shots were worth more/less than usual

    "Shots" = field goal attempts + 0.44 x free throws (the standard "true shot attempts").
    """
    g = g.copy()
    pred, exp_min, mins = g[pred_col].clip(lower=0.1), g["exp_min"].clip(lower=1), g["MIN"]
    rate = pred / exp_min                                           # expected points per minute
    shots = g["FGA"] + 0.44 * g["FTA"]
    per_shot = g["exp_pts_per_tsa"].clip(lower=0.5)
    g["exp_shots"] = exp_min * rate / per_shot
    g["shots"] = shots
    g["miss"] = g["PTS"] - g[pred_col]                              # + = he scored MORE than we said
    g["part_minutes"] = (mins - exp_min) * rate
    g["part_volume"] = (shots / mins - rate / per_shot) * mins * per_shot
    g["part_shooting"] = g["miss"] - g["part_minutes"] - g["part_volume"]
    return g


def error_budget(d):
    """How low could the error go if we knew more? (An upper limit, not a promise.)"""
    return pd.DataFrame([
        {"If we knew...": "only what we know now", "MAE": d["miss"].abs().mean()},
        {"If we knew...": "+ his exact minutes", "MAE": (d["miss"] - d["part_minutes"]).abs().mean()},
        {"If we knew...": "+ how many shots he took", "MAE": d["part_shooting"].abs().mean()},
    ])


def main_cause(r):
    parts = {"minutes": r["part_minutes"], "volume": r["part_volume"], "shooting": r["part_shooting"]}
    cause = max(parts, key=lambda k: abs(parts[k]))
    if cause == "minutes":
        return f"minutes: played {r['MIN']:.0f}, expected {r['exp_min']:.0f} ({parts[cause]:+.1f} pts)"
    if cause == "volume":
        return f"shot volume: {r['shots']:.0f} shots, expected {r['exp_shots']:.0f} ({parts[cause]:+.1f} pts)"
    made = r["PTS"] / r["shots"] if r["shots"] > 0 else 0
    return f"shooting: {made:.2f} pts per shot, usually {r['exp_pts_per_tsa']:.2f} ({parts[cause]:+.1f} pts)"


# =============================================================================
# 4. Blind spots: groups where we lean the same way every time
# =============================================================================
SEGMENTS = [
    ("Bench players (<10 ppg)", lambda g: g["PTS_SEASON"] < 10),
    ("Mid scorers (10-20 ppg)", lambda g: g["PTS_SEASON"].between(10, 20, inclusive="left")),
    ("Stars (20+ ppg)", lambda g: g["PTS_SEASON"] >= 20),
    ("Young players (22 or younger)", lambda g: g["AGE"] <= 22),
    ("Veterans (33 or older)", lambda g: g["AGE"] >= 33),
    ("New to our data (<60 games)", lambda g: g["GAMES_PLAYED_BEFORE"] < 60),
    ("Home games", lambda g: g["HOME"] == 1),
    ("Away games", lambda g: g["HOME"] == 0),
    ("Second night of a back-to-back", lambda g: g["BACK_TO_BACK"] == 1),
    ("3+ days of rest", lambda g: g["REST_DAYS"] >= 3),
    ("Returning after missing 3+ team games", lambda g: g["TEAM_GAMES_MISSED"] >= 3),
    ("Teammates out worth 15+ ppg", lambda g: g["TEAMMATES_OUT_PTS"] >= 15),
    ("No teammates out", lambda g: g["TEAMMATES_OUT_PTS"] < 1),
    ("Opponent missing 15+ ppg", lambda g: g["OPP_PLAYERS_OUT_PTS"] >= 15),
    ("Likely blowout (big strength gap)", lambda g: g["ABS_NET_RTG_GAP"] >= 10),
    ("Expected under 15 minutes", lambda g: g["exp_min"] < 15),
    ("Expected 30+ minutes", lambda g: g["exp_min"] >= 30),
]

ADVICE = {
    "Young players (22 or younger)": "Young players improve faster than their averages show. Next idea: retrain so their player fingerprints refresh.",
    "New to our data (<60 games)": "Rookies and newcomers share the 'unknown player' fingerprint. A mid-season retrain did not help them (tested 2026-11-25). Next idea: add draft-pick or prior-league info.",
    "Returning after missing 3+ team games": "Version 2 added 'games since return', 'player extra rest' and 'minutes last game' for exactly this. If the lean stays, the next idea is a minutes-limit signal from injury news.",
    "3+ days of rest": "Version 2 added 'player extra rest', which separates the team's days off from games he sat out. If the lean stays, the next idea is the reason for the absence from the injury report.",
    "Mid scorers (10-20 ppg)": "This season teams use deeper rotations, which costs mid scorers the most. Version 2 added rotation depth and season timing, which lowered the error but not the lean. The per-tier bias correction now works on it; it applies only part of the gap at first and more as games confirm it.",
    "Teammates out worth 15+ ppg": "When stars sit, the model shifts points to teammates by one average rule. Next idea: split 'teammates out' by position or by who usually shares the floor.",
    "Second night of a back-to-back": "Teams rest players on back-to-backs. Next idea: each player's own history of sitting or playing less on back-to-backs.",
    "Likely blowout (big strength gap)": "Blowouts cut starters' minutes. Next idea: add the betting point spread, which measures blowout risk better than our own ratings.",
    "Expected under 15 minutes": "Deep-bench minutes are the least predictable. Next idea: coach rotation patterns, or flag these as low-confidence.",
    "Bench players (<10 ppg)": "Bench roles change quickly from night to night. Next idea: the coach's recent substitution patterns.",
    "Stars (20+ ppg)": "Stars carry the largest swings. Next idea: opponent defense against his position and the game's betting total.",
}


def blind_spots(d):
    """
    Our models aim for the MIDDLE outcome (the median), because that gives the lowest
    average miss. So a fair prediction is beaten exactly half the time. If a group of
    players beats our number 58% of the time, we are too low for that group, and that
    is fixable. (The plain average of the misses would mislead here: rare 30-point
    nights pull it up even when the prediction is perfectly fair.)
    """
    rows = []
    for label, mask in SEGMENTS:
        s = d[mask(d).fillna(False)]
        if len(s) < 80:
            continue
        over = beaten(s["p_blend_static"], s["PTS"])                # share who scored MORE than predicted
        z = (over - 0.5) / np.sqrt(0.25 / len(s))                   # how far from 50%, in standard errors
        lean = -s["miss"].median()                                  # typical (prediction - actual)
        comp_over = {k: abs(beaten(s[f"p_{k}"], s["PTS"]) - 0.5) for k in KEYS}
        rows.append({
            "Group": label, "Games": len(s), "MAE": s["miss"].abs().mean(), "over": over, "lean": lean, "z": z,
            "over_fixed": beaten(s["replay"], s["PTS"]) if "replay" in s else np.nan,
            "minutes": -s["part_minutes"].median(), "volume": -s["part_volume"].median(),
            "shooting": -s["part_shooting"].median(),
            "worst": max(comp_over, key=comp_over.get), "best": min(comp_over, key=comp_over.get),
            "flag": abs(z) >= 3 and abs(lean) >= 0.4})
    return pd.DataFrame(rows)


# =============================================================================
# 5. The report
# =============================================================================
def md_table(df, fmt=None):
    fmt = fmt or {}
    def cell(col, v):
        if col in fmt:
            return fmt[col].format(v)
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(" --- " for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def build_report(preds, state, decisions, info):
    """info: dict with run details (now, next_day, assumed_out, injuries_ok, notes)."""
    graded = preds[preds["status"] == "graded"]
    pending = preds[preds["status"] == "pending"]
    live = graded[graded["source"] == "live"]
    L = [f"# SwishAI daily report", "",
         f"Updated {info['now']}. This file is rewritten automatically by `python -m src.daily`.", ""]

    # ---------------------------------------------------------------- summary
    if graded.empty:
        return "\n".join(L + ["No graded predictions yet. They appear after the first games finish."])
    days = graded["GAME_DATE"].nunique()
    blend_mae = (graded["p_blend_static"] - graded["PTS"]).abs().mean()
    base_mae = (graded["p_baseline"] - graded["PTS"]).abs().mean()
    L += ["## Where things stand", "",
          f"- **{len(graded):,} predictions graded** over {days} game days this season "
          f"({len(live):,} made live before tip-off, the rest filled in afterwards from pre-game data).",
          f"- **Blend error this season: {blend_mae:.2f} points per game** "
          f"(simple weighted average: {base_mae:.2f}; last season's locked test: 4.42).",
          f"- **Blend in use: {state['policy'].replace('_', ' + ')}**, weights hybrid {state['weights']['hybrid']:.2f} / "
          f"neural nets {state['weights']['nn']:.2f} / simulation {state['weights']['sim']:.2f}; "
          f"offset {state['offset']:+.2f}; bench / mid / star offsets "
          + " / ".join(f"{state.get('tier_offsets', {}).get(t, 0):+.2f}" for t in TIER_NAMES) + ".",
          f"- **{len(pending)} predictions waiting** for games on {info.get('next_day') or 'the next game day'}.", ""]

    # ------------------------------------------------------------ last night
    last = graded["GAME_DATE"].max()
    night = decompose(graded[graded["GAME_DATE"] == last])
    inside = ((night["PTS"] >= night["q10"]) & (night["PTS"] <= night["q90"])).mean()
    L += [f"## Last graded night: {last}", "",
          f"{len(night)} players graded. Blend missed by **{night['miss'].abs().mean():.2f}** points on average; "
          f"{inside:.0%} of scores landed inside the 80% range.", "",
          "**Top expected scorers**", ""]
    top = night.sort_values("p_blend", ascending=False).head(8)
    L += [md_table(pd.DataFrame({
        "Player": top["PLAYER_NAME"], "Game": top["TEAM_ABBREVIATION"] + " v " + top["OPPONENT"],
        "Predicted": top["p_blend"], "80% range": top["q10"].astype(int).astype(str) + "-" + top["q90"].astype(int).astype(str),
        "Actual": top["PTS"].astype(int), "Miss": top["miss"]}), {"Predicted": "{:.1f}", "Miss": "{:+.1f}"}), "",
        "**Biggest misses, and the main reason for each**", ""]
    worst = night.reindex(night["miss"].abs().sort_values(ascending=False).index).head(6)
    L += [md_table(pd.DataFrame({
        "Player": worst["PLAYER_NAME"], "Predicted": worst["p_blend_static"], "Actual": worst["PTS"].astype(int),
        "Hybrid": worst["p_hybrid"], "Neural": worst["p_nn"], "Sim": worst["p_sim"],
        "Main reason": [main_cause(r) for _, r in worst.iterrows()]}),
        {"Predicted": "{:.1f}", "Hybrid": "{:.1f}", "Neural": "{:.1f}", "Sim": "{:.0f}"}), ""]

    # ------------------------------------------------------------ scoreboard
    models = list(MODEL_NAMES)
    L += ["## Which model is doing better or worse?", "",
          "MAE = average points missed (lower is better). Bias = average of prediction minus actual. "
          "Beaten = how often the player scored more than predicted. The ML models aim for the typical (median) game, "
          "so Beaten near 50% is the sign of a fair model; a slightly negative Bias is normal, because rare huge "
          "nights pull the average up. 'vs blend' compares each model with the fixed blend on the same games; "
          "the verdict only says better or worse when the gap is bigger than its uncertainty (+/-).", "",
          f"**Season to date ({len(graded):,} games)**", "",
          md_table(scoreboard(graded, models), {"MAE": "{:.3f}", "Bias": "{:+.2f}", "Beaten": "{:.0%}", "vs blend": "{:+.3f}", "+/-": "{:.3f}"}), ""]
    recent_days = sorted(graded["GAME_DATE"].unique())[-7:]
    recent = graded[graded["GAME_DATE"].isin(recent_days)]
    if days > 7:
        L += [f"**Last 7 game days ({len(recent):,} games)**", "",
              md_table(scoreboard(recent, models), {"MAE": "{:.3f}", "Bias": "{:+.2f}", "Beaten": "{:.0%}", "vs blend": "{:+.3f}", "+/-": "{:.3f}"}), ""]
    if len(live) >= 30:
        off = live["p_blend"] - live["PTS"]
        fix = live["p_blend_static"] - live["PTS"]
        L += [f"**Live picks only ({len(live):,} games predicted before tip-off):** official SwishAI number "
              f"{off.abs().mean():.3f} MAE, fixed blend {fix.abs().mean():.3f} MAE.", ""]

    # ---------------------------------------------------------- blend weights
    L += ["## Should the blend weights change?", "", state.get("policy_reason", ""), ""]
    if state.get("walk_forward"):
        L += [md_table(pd.DataFrame(state["walk_forward"]), {"MAE": "{:.4f}", "vs fixed": "{:+.4f}", "SE": "{:.4f}"}), "",
              f"Weights the adaptive method currently suggests: hybrid {state['suggested_weights']['hybrid']:.2f} / "
              f"neural nets {state['suggested_weights']['nn']:.2f} / simulation {state['suggested_weights']['sim']:.2f}, "
              f"one offset {state['suggested_offset']:+.2f}, or per tier (bench / mid / star) "
              + " / ".join(f"{state['suggested_tier_offsets'][t]:+.2f}" for t in TIER_NAMES)
              + ". An option is only used if the table above shows it winning clearly.", ""]

    # ----------------------------------------------------------- error budget
    d = decompose(graded)
    budget = error_budget(d)
    share = d[["part_minutes", "part_volume", "part_shooting"]].abs().mean()
    L += ["## Why are we off? The error budget", "",
          "Every miss is split into three parts that add up exactly: he played more or fewer **minutes** than expected, "
          "he took more or fewer **shots** per minute, or his shots **went in** more or less than usual.", "",
          md_table(pd.DataFrame({"Part of the miss": ["Minutes", "Shot volume", "Shooting (mostly luck)"],
                                 "Average size (points)": share.to_numpy()})), "",
          md_table(budget, {"MAE": "{:.2f}"}), "",
          f"Reading: knowing exact minutes would cut the error from {budget['MAE'][0]:.2f} to {budget['MAE'][1]:.2f}. "
          f"Even knowing minutes AND shot count, {budget['MAE'][2]:.2f} points of error remain: that is shooting luck, "
          "which no pre-game model can remove. So the realistic gains are in predicting minutes and role.", ""]

    # ------------------------------------------------------------ blind spots
    # What the blend policy in use would have predicted, replayed day by day (no peeking).
    if state["policy"] != "static":
        d["replay"] = walk_forward(graded)[state["policy"]].reindex(d.index).fillna(d["p_blend_static"])
    spots = blind_spots(d)
    flagged = spots[spots["flag"]] if len(spots) else spots
    L += ["## Blind spots: groups where we lean the same way", "",
          "A fair prediction is beaten half the time. 'Beaten' shows how often players in each group scored more than "
          "the models predicted (before any automatic correction). A group is flagged only when that is far from 50% "
          "beyond chance (z of 3 or more) AND the typical gap is at least 0.4 points. Lean = typical prediction minus "
          "actual (negative = too low). The last three columns show the typical miss from each part.", ""]
    if len(spots):
        show = spots.sort_values("z", key=abs, ascending=False).head(10)
        L += [md_table(pd.DataFrame({
            "Group": show["Group"], "Games": show["Games"], "MAE": show["MAE"], "Beaten": show["over"], "Lean": show["lean"],
            "z": show["z"], "minutes": show["minutes"], "volume": show["volume"], "shooting": show["shooting"],
            "Flag": np.where(show["flag"], "YES", "")}),
            {"Beaten": "{:.0%}", "Lean": "{:+.2f}", "z": "{:+.1f}", "minutes": "{:+.2f}", "volume": "{:+.2f}",
             "shooting": "{:+.2f}"}), ""]
        if not len(flagged):
            L += ["No group is flagged: the misses look like noise, not a repeatable lean.", ""]

    # ---------------------------------------------------------- lineup check
    if len(preds[preds["source"] == "live"]):
        lv = preds[(preds["source"] == "live") & (preds["status"].isin(["graded", "dnp"]))]
        unexpected = int((graded["source"] == "unexpected").sum())
        if len(lv):
            L += ["## Did we guess the lineups right? (live picks)", "",
                  f"- Players we expected to play: {len(lv):,}. Did not play: {(lv['status'] == 'dnp').sum():,} "
                  f"({(lv['status'] == 'dnp').mean():.0%}).",
                  f"- Players who played but we did not expect: {unexpected:,}.",
                  "- Wrong lineup guesses hurt twice: that player's own prediction, and his teammates' "
                  "(their 'teammates out' input was wrong).", ""]

    # ---------------------------------------------------------------- actions
    L += ["## Actions and reasoning", "", "**Done automatically this run**", ""]
    L += [f"- **{x['action']}**: {x['change']}. *Why:* {x['reasoning']}" for x in decisions] or \
         ["- No change this run. The blend policy from the last review stays in place "
          "(see 'Should the blend weights change?' above)."]
    L += ["", "**Recommended (waiting for your go-ahead)**", ""]
    recs = []
    new_games = int((graded["GAME_DATE"] > state.get("model_cutoff", "")).sum())
    if new_games >= RETRAIN_AFTER:
        recs.append(f"**Retrain the models** (`python -m src.daily --retrain`). *Why:* {new_games:,} games have been played "
                    "since the models were last trained, enough to be a real share of the training data. Retraining lets "
                    "them learn this season's patterns and gives rookies their own player fingerprint. It takes a few "
                    "minutes and replaces the saved model, so it waits for you. Check it fairly before keeping it: train a "
                    "copy without the last week and compare both models on that week.")
    else:
        recs.append(f"**Do not retrain yet.** *Why:* only {new_games:,} new games since the last training "
                    f"(threshold {RETRAIN_AFTER:,}). On 2026-11-25 a retrain on about 4,500 new games was tested fairly "
                    "and gave no gain (they are only ~3% of the training data), so it was rolled back.")
    for _, r in flagged.iterrows():
        direction = "too HIGH" if r["lean"] > 0 else "too LOW"
        part = max(["minutes", "volume", "shooting"], key=lambda k: abs(r[k]))
        recs.append(f"**{r['Group']}**: predictions run {direction}. Players beat our number {r['over']:.0%} of the time "
                    f"over {r['Games']} games (50% is fair), typical gap {abs(r['lean']):.2f} points, mostly through "
                    f"**{part}**. Most off: {r['worst']} model; least off: {r['best']}. "
                    + (f"With the automatic correction in use, that becomes {r['over_fixed']:.0%}. "
                       if pd.notna(r["over_fixed"]) else "")
                    + ADVICE.get(r["Group"], "Idea: add a feature that describes this situation directly."))
    gain = budget["MAE"][0] - budget["MAE"][1]
    recs.append(f"**Biggest lever overall: minutes.** *Why:* perfect minutes knowledge would be worth {gain:.2f} points of MAE, "
                "far more than any model swap (the models differ by about 0.01). Better injury, lineup and blowout "
                "information is where accuracy can still be won.")
    L += [f"- {x}" for x in recs] + [""]

    # ------------------------------------------------------- next predictions
    if len(pending):
        nxt = pending.sort_values("p_blend", ascending=False)
        L += [f"## Next predictions: {info.get('next_day')}", "",
              f"{len(nxt)} players across {nxt['GAME_ID'].nunique()} games. Top 25 shown; all are in `tracking/predictions.csv`.", ""]
        if info.get("injuries_ok"):
            L += ["Lineups assume: played his team's last game, and not listed as out on the injury report. "
                  + (f"Left out as injured: {', '.join(info['assumed_out'])}." if info.get("assumed_out") else ""), ""]
        else:
            L += ["Injury report could not be loaded, so lineups assume everyone who played the last game plays again.", ""]
        t = nxt.head(25)
        L += [md_table(pd.DataFrame({
            "Player": t["PLAYER_NAME"], "Game": t["TEAM_ABBREVIATION"] + np.where(t["HOME"] == 1, " v ", " @ ") + t["OPPONENT"],
            "Predicted": t["p_blend"], "80% range": t["q10"].astype(int).astype(str) + "-" + t["q90"].astype(int).astype(str),
            "Exp. minutes": t["exp_min"], "Teammates out (ppg)": t["TEAMMATES_OUT_PTS"]}),
            {"Predicted": "{:.1f}", "Exp. minutes": "{:.0f}", "Teammates out (ppg)": "{:.0f}"}), ""]
    for note in info.get("notes", []):
        L += [f"> {note}", ""]
    return "\n".join(L)
