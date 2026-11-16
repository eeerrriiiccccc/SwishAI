"""
Phase 8 - Monte Carlo game simulation.

Instead of predicting points directly, we play each game 2,000 times in the
computer, building points from basketball parts:

    1. MINUTES   expected minutes from a model, plus a realistic "surprise"
                 drawn from the minutes model's past errors (foul trouble,
                 blowouts, garbage time, injuries mid-game...)
    2. SHOTS     2-pt, 3-pt and free-throw attempts ~ Poisson(rate per minute x minutes)
                 Free throws come in PAIRS (trips to the line), which is why
                 1-point games are so rare in real life.
                 A shared "aggressiveness" factor per simulated game: some nights
                 a player just shoots more of everything.
    3. MAKES     each attempt goes in with the player's (shrunk) make rate.
    4. POINTS    2 x (2s made) + 3 x (3s made) + free throws made

The 2,000 results form the predicted distribution. Its shape comes from
basketball logic, not from a textbook formula.

ANCHORING: the shooting rates come from long-run history, which doesn't know
about tonight's context (teammates out, opponent, role changes). So by default
we scale the shot volume so the simulation's average matches the Phase 5
linear model's prediction: the ML model sets the CENTER, the simulation sets
the SHAPE.

Run with:
    .venv\\Scripts\\python.exe -m src.models.simulate
"""
import numpy as np
import pandas as pd

from src.evaluate import evaluate, get_split, load_data
from src.features import FEATURES, SIM_COLS
from src.models.distributions import K_MAX, DistributionModel, distribution_scores
from src.models.linear import LinearRegressionScratch

MINUTE_BINS = [0, 8, 12, 16, 20, 24, 28, 32, 36, 60]


class MinutesModel:
    """LightGBM predicting MINUTES (not points) from the same pre-game features."""

    def fit(self, train):
        import lightgbm as lgb
        params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
                  "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0}
        self.booster = lgb.train(params, lgb.Dataset(train[FEATURES], train["MIN"]), num_boost_round=400)
        return self

    def predict(self, df):
        return np.clip(self.booster.predict(df[FEATURES]), 1, 48)


class GameSimulator:
    def __init__(self, n_sims=2000, anchored=True, seed=0):
        self.n_sims, self.anchored, self.seed = n_sims, anchored, seed

    # ------------------------------------------------------------------ fit
    def fit(self, train):
        newest = sorted(train["SEASON"].unique())[-1]
        inner_train, inner_valid = train[train["SEASON"] != newest], train[train["SEASON"] == newest]
        self.sim_medians = train[SIM_COLS].median()

        # 1. "Minutes surprises": errors of a minutes model on a season it did NOT train on,
        #    grouped by how many minutes were expected (a 34-min starter and a 9-min
        #    bench player have very different kinds of surprises).
        inner_minutes = MinutesModel().fit(inner_train)
        expected = inner_minutes.predict(inner_valid)
        errors = inner_valid["MIN"].to_numpy() - expected
        bins = np.digitize(expected, MINUTE_BINS)
        self.minute_errors = {b: errors[bins == b] for b in np.unique(bins)}

        # 2. Tune the "aggressiveness" spread on the held-out training season (by log score).
        self.minutes_model = inner_minutes
        self.center_model = LinearRegressionScratch(FEATURES, interactions=True).fit(inner_train)
        sample = inner_valid.sample(5000, random_state=0)
        scores = {}
        for shape in [4, 8, 16, 32, 1e6]:
            self.shot_shape = shape
            scores[shape] = distribution_scores(self.pmf(sample, n_sims=1000), sample)["LOG_SCORE"]
        self.shot_shape = max(scores, key=scores.get)
        self.tuning_scores = scores

        # 3. Refit the minutes and center models on ALL training seasons.
        self.minutes_model = MinutesModel().fit(train)
        self.center_model = LinearRegressionScratch(FEATURES, interactions=True).fit(train)
        return self

    # ------------------------------------------------------------- simulate
    def simulate(self, df, n_sims=None, return_parts=False):
        """Returns an array (rows, n_sims) of simulated point totals."""
        n_sims = n_sims or self.n_sims
        rng = np.random.default_rng(self.seed)
        n = len(df)
        sim = df[SIM_COLS].fillna(self.sim_medians).to_numpy(dtype=float)
        r2, r3, rft, p2, p3, pft = (sim[:, i:i + 1] for i in range(6))     # each shape (n, 1)

        # 1. Minutes: expected + a surprise drawn from the matching bin's past errors.
        expected_min = self.minutes_model.predict(df)
        bins = np.digitize(expected_min, MINUTE_BINS)
        surprise = np.empty((n, n_sims))
        for b in np.unique(bins):
            rows = bins == b
            pool = self.minute_errors.get(b, self.minute_errors[min(self.minute_errors, key=lambda k: abs(k - b))])
            surprise[rows] = rng.choice(pool, size=(rows.sum(), n_sims))
        minutes = np.clip(expected_min[:, None] + surprise, 1, 53)

        # Anchoring: scale shot volume so the simulated AVERAGE equals the ML model's prediction.
        points_per_minute = 2 * r2 * p2 + 3 * r3 * p3 + rft * pft
        if self.anchored:
            target = np.clip(self.center_model.predict(df), 0.3, None)[:, None]
            volume = target / (minutes.mean(axis=1, keepdims=True) * points_per_minute)
        else:
            volume = 1.0

        # 2. Shots. One "aggressiveness" factor per simulated game, average 1.
        aggressive = rng.gamma(self.shot_shape, 1 / self.shot_shape, size=(n, n_sims))
        scale = minutes * volume * aggressive
        att2 = rng.poisson(r2 * scale)
        att3 = rng.poisson(r3 * scale)
        att_ft = 2 * rng.poisson(rft * scale / 2)              # free throws come in pairs

        # 3. Makes.
        made2 = rng.binomial(att2, np.broadcast_to(p2, att2.shape))
        made3 = rng.binomial(att3, np.broadcast_to(p3, att3.shape))
        made_ft = rng.binomial(att_ft, np.broadcast_to(pft, att_ft.shape))

        points = 2 * made2 + 3 * made3 + made_ft
        if return_parts:
            return points, {"minutes": minutes, "2PA": att2, "3PA": att3, "FTA": att_ft}
        return points

    # ----------------------------------------------- distribution interface
    def pmf(self, df, n_sims=None, chunk=2000):
        """Share of simulations landing on each point total 0..90 (lightly smoothed)."""
        out = []
        for start in range(0, len(df), chunk):
            part = df.iloc[start:start + chunk]
            pts = np.clip(self.simulate(part, n_sims), 0, K_MAX)
            idx = (np.arange(len(part))[:, None] * (K_MAX + 1) + pts).ravel()
            counts = np.bincount(idx, minlength=len(part) * (K_MAX + 1)).reshape(len(part), K_MAX + 1)
            out.append((counts + 0.02) / (counts.sum(axis=1, keepdims=True) + 0.02 * (K_MAX + 1)))
        return np.vstack(out)

    def predict(self, df):
        """Median of the simulations (for the MAE leaderboard)."""
        cdf = np.cumsum(self.pmf(df), axis=1)
        return (cdf < 0.5).sum(axis=1)


def main():
    df = load_data()
    train, valid = get_split(df, "valid")
    y = valid["PTS"].to_numpy()

    rows = []
    for name, model in [
        ("Normal, spread grows with mean (Phase 6)", DistributionModel("normal_scaled").fit(train)),
        ("Negative Binomial (Phase 6)", DistributionModel("negbin").fit(train)),
        ("Simulation, pure basketball rates", GameSimulator(anchored=False).fit(train)),
        ("Simulation, anchored to ML center", GameSimulator(anchored=True).fit(train)),
    ]:
        pmf = model.pmf(valid)
        s = distribution_scores(pmf, valid)
        s["P(0 PTS)"] = pmf[:, 0].mean()
        s["MAE (median)"] = np.mean(np.abs((np.cumsum(pmf, axis=1) < 0.5).sum(axis=1) - y))
        rows.append({"model": name, **s})
        if isinstance(model, GameSimulator):
            print(f"{name}: aggressiveness shape picked = {model.shot_shape:g} "
                  f"(log scores {', '.join(f'{k:g}: {v:.4f}' for k, v in model.tuning_scores.items())})")

    table = pd.DataFrame(rows).set_index("model")
    print("\n=== DISTRIBUTIONS vs SIMULATION (validation, 2024-25) ===")
    print(table.round(4).to_string())
    print(f"Actual share of 0-point games: {np.mean(y == 0):.4f}")
    table.to_csv("results/simulation_scores.csv")

    print()
    evaluate(GameSimulator(anchored=True), "Simulation, anchored (median)", df=df)


if __name__ == "__main__":
    main()
