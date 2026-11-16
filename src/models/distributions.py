"""
Phase 6 - Distribution models: predict the full RANGE of outcomes, not one number.

Instead of "he'll score 24.3", a distribution model says:
    P(0 pts) = 0.1%, P(1 pt) = 0.1%, ..., P(24 pts) = 4.8%, ..., P(45 pts) = 0.2%

That lets us answer questions a single number can't:
    "What's the chance he scores 30+?"   "What's a realistic range for tonight?"

How the models here are built
-----------------------------
1. CENTER: the best linear model from Phase 5 predicts the expected points (mu).
2. SPREAD: a probability distribution is wrapped around mu. Its spread
   parameters are learned by MAXIMUM LIKELIHOOD: choose the parameters that
   make the outcomes we actually saw in training as probable as possible.

The four models (simplest -> most flexible):
    normal_const  Bell curve, SAME spread for everyone.
    normal_scaled Bell curve, variance = b + a * mu.
    negbin        Negative Binomial (skewed count distribution), variance = b + a * mu.
    negbin_player Negative Binomial where 'a' depends on the PLAYER: 3-point-heavy
                  and historically streaky players get a wider distribution.

Why "b + a * mu"? Two sources of uncertainty add up:
    b      - our CENTER is imperfect; that error is roughly the same size for everyone
    a * mu - the game's own randomness, which grows with how much a player scores
             (Phase 1: bigger scorers have bigger swings)

Run with:
    .venv\\Scripts\\python.exe -m src.models.distributions
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.config import RESULTS_DIR
from src.evaluate import evaluate, get_split, load_data
from src.features import FEATURES
from src.models.linear import LinearRegressionScratch

K_MAX = 90                      # we model outcomes 0, 1, ..., 90 points
POINTS = np.arange(K_MAX + 1)


# =============================================================================
# Turning parameters into probabilities for every point total 0..90
# =============================================================================
def normal_pmf(mu, sigma):
    """Bell curve, chopped into whole-point buckets: P(k) = P(k-0.5 < X < k+0.5).
    Everything below 0.5 goes into the 0 bucket (you can't score negative points)."""
    edges = np.append(POINTS[:-1] + 0.5, np.inf)                   # upper edge of each bucket (last one: no limit)
    cdf = stats.norm.cdf((edges[None, :] - mu[:, None]) / sigma[:, None])
    return np.diff(np.column_stack([np.zeros(len(mu)), cdf]), axis=1)


def negbin_params(mu, var):
    """scipy's (n, p) for a Negative Binomial with the given mean and variance (needs var > mu)."""
    return mu ** 2 / (var - mu), mu / var


def negbin_pmf(mu, var):
    """Negative Binomial probabilities for 0..90 points, given each row's mean and variance."""
    n, p = negbin_params(mu, var)
    pmf = stats.nbinom.pmf(POINTS[None, :], n[:, None], p[:, None])
    pmf[:, -1] += 1 - pmf.sum(axis=1)                              # put the far tail in the last bucket
    return pmf


# =============================================================================
# The model
# =============================================================================
class DistributionModel:
    def __init__(self, kind):
        assert kind in ("normal_const", "normal_scaled", "negbin", "negbin_player")
        self.kind = kind

    # --- player-level "streakiness" inputs for negbin_player -----------------
    def _volatility_inputs(self, df):
        three_rate = df["FG3_RATE_L20"]
        # His recent std relative to what a typical player with his average would have.
        rel_std = np.log((df["PTS_STD_L10"] + 0.5) / np.sqrt(df["PTS_EWM20"] + 1))
        Z = pd.DataFrame({"three_rate": three_rate, "rel_std": rel_std})
        return ((Z.fillna(self.z_median) - self.z_mean) / self.z_std).to_numpy()

    def _variance(self, theta, mu, Z=None):
        """variance = b + a * mu.  theta = the raw numbers the optimizer adjusts:
        b = exp(theta[0]) and a = 1 + exp(theta[1] + ...) keep b > 0 and a > 1."""
        b = np.exp(theta[0])
        if self.kind == "negbin_player":
            a = 1 + np.exp(theta[1] + Z @ theta[2:])
        else:
            a = 1 + np.exp(theta[1])
        return b + a * mu

    def fit(self, train):
        # 1. The center: Phase 5's best model.
        self.mean_model = LinearRegressionScratch(FEATURES, interactions=True).fit(train)
        mu = self._mu(train)
        y = train["PTS"].to_numpy()
        residual = y - mu

        if self.kind == "normal_const":
            self.sigma = residual.std()                            # the MLE for a normal's spread
            return self

        if self.kind == "negbin_player":
            Z_raw = pd.DataFrame({"three_rate": train["FG3_RATE_L20"],
                                  "rel_std": np.log((train["PTS_STD_L10"] + 0.5) / np.sqrt(train["PTS_EWM20"] + 1))})
            self.z_median, self.z_mean, self.z_std = Z_raw.median(), Z_raw.fillna(Z_raw.median()).mean(), Z_raw.std()
            Z = self._volatility_inputs(train)
            start = np.array([np.log(5), np.log(1.5), 0.0, 0.0])
        else:
            Z, start = None, np.array([np.log(5), np.log(1.5)])

        # 2. Maximum likelihood: find theta that minimizes the NEGATIVE log-likelihood.
        def negative_log_likelihood(theta):
            var = self._variance(theta, mu, Z)
            if self.kind == "normal_scaled":
                return -stats.norm.logpdf(y, mu, np.sqrt(var)).sum()
            return -stats.nbinom.logpmf(y, *negbin_params(mu, var)).sum()

        result = optimize.minimize(negative_log_likelihood, start, method="BFGS")
        self.theta = result.x
        return self

    def _mu(self, df):
        return np.clip(self.mean_model.predict(df), 0.3, None)       # expected points must be positive

    def pmf(self, df):
        """Probability of every point total 0..90, one row per player-game."""
        mu = self._mu(df)
        if self.kind == "normal_const":
            return normal_pmf(mu, np.full(len(mu), self.sigma))
        Z = self._volatility_inputs(df) if self.kind == "negbin_player" else None
        var = self._variance(self.theta, mu, Z)
        if self.kind == "normal_scaled":
            return normal_pmf(mu, np.sqrt(var))
        return negbin_pmf(mu, var)

    def quantile(self, df, q):
        """Smallest k with P(points <= k) >= q.  q=0.5 -> the median."""
        cdf = np.cumsum(self.pmf(df), axis=1)
        return (cdf < q).sum(axis=1)

    def predict(self, df):
        """For the leaderboard: the MEDIAN, because the median minimizes absolute error (MAE)."""
        return self.quantile(df, 0.5)

    def describe(self):
        if self.kind == "normal_const":
            return f"sigma = {self.sigma:.2f} points for everyone"
        b, a = np.exp(self.theta[0]), 1 + np.exp(self.theta[1])
        text = f"variance = {b:.1f} + {a:.2f} x expected points"
        if self.kind == "negbin_player":
            text += (f" (typical player); +1 std 3pt-rate: (a-1) x{np.exp(self.theta[2]):.2f}, "
                     f"+1 std streakiness: (a-1) x{np.exp(self.theta[3]):.2f}")
        return text


# =============================================================================
# Scoring a distribution
# =============================================================================
def betting_line(df):
    """A sportsbook-style line: his weighted average, rounded to x.5 (so no ties)."""
    return np.floor(df["PTS_EWM20"].to_numpy()) + 0.5


def randomized_pit(pmf, y, seed=0):
    """P(points < actual) + random share of P(points == actual). Uniform(0,1) if the model is honest."""
    rows = np.arange(len(y))
    cdf_below = np.cumsum(pmf, axis=1)[rows, y] - pmf[rows, y]
    return cdf_below + np.random.default_rng(seed).uniform(size=len(y)) * pmf[rows, y]


def distribution_scores(pmf, df):
    """
    LOG SCORE  - average log-probability given to what actually happened. Higher (closer to 0)
                 is better. Rewards being confident AND right; heavily punishes 'impossible' surprises.
    COVER 80   - how often the actual result lands in the model's middle 80%. Should be 80%.
                 Less = overconfident (ranges too narrow), more = too cautious.
                 (Computed with the 'randomized PIT' trick, which handles whole-number points fairly:
                  PIT = where the actual result falls in the predicted distribution, from 0 to 1.
                  If the model is honest, PIT values are spread evenly between 0 and 1.)
    BRIER OVER - for the question 'will he go OVER his line?': mean squared error of the probability.
                 Lower is better. Always saying 50% scores 0.25.
    """
    y = np.clip(df["PTS"].to_numpy(), 0, K_MAX)
    cdf = np.cumsum(pmf, axis=1)
    q10, q90 = (cdf < 0.1).sum(axis=1), (cdf < 0.9).sum(axis=1)
    line = betting_line(df)
    p_over = 1 - cdf[np.arange(len(y)), np.floor(line).astype(int)]
    went_over = (df["PTS"].to_numpy() > line).astype(float)
    pit = randomized_pit(pmf, y)
    return {
        "LOG_SCORE": np.mean(np.log(np.clip(pmf[np.arange(len(y)), y], 1e-12, None))),
        "COVER_80": np.mean((pit > 0.1) & (pit < 0.9)),
        "AVG_RANGE_WIDTH": np.mean(q90 - q10),
        "BRIER_OVER": np.mean((p_over - went_over) ** 2),
    }


MODELS = {
    "normal_const": "Normal, same spread for all",
    "normal_scaled": "Normal, spread grows with mean",
    "negbin": "Negative Binomial",
    "negbin_player": "Negative Binomial, player-specific spread",
}


def main():
    df = load_data()
    train, valid = get_split(df, "valid")
    rows = []
    for kind, name in MODELS.items():
        model = DistributionModel(kind).fit(train)
        s = distribution_scores(model.pmf(valid), valid)
        rows.append({"model": name, **s})
        print(f"{name:<44} {model.describe()}")
    table = pd.DataFrame(rows).set_index("model")
    print("\n=== DISTRIBUTION SCORES (validation, 2024-25) ===")
    print(table.round(4).to_string())
    print("(Brier if you always say 50%: 0.2500)")
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "distribution_scores.csv")

    # Point predictions from the distributions (the MEDIAN) go on the main leaderboard.
    print("\nUsing each distribution's MEDIAN as a point prediction:")
    for kind in ["normal_scaled", "negbin_player"]:
        evaluate(DistributionModel(kind), f"{MODELS[kind]} (median)", df=df)


if __name__ == "__main__":
    main()
