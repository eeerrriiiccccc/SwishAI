"""
Phase 5 - Linear regression, built from scratch with NumPy.

The model
---------
    predicted_points = w0 + w1*x1 + w2*x2 + ... + wk*xk

x1..xk are features (last-10 avg, opponent defense, ...). w0..wk are WEIGHTS
the model learns. "Learning" = finding the weights that make the squared
prediction errors on the training data as small as possible.

Two ways to find those weights:
  1. Normal equation: a direct formula, w = (X^T X)^-1 X^T y. One step, exact.
  2. Gradient descent: start with w = 0, repeatedly nudge each weight in the
     direction that reduces the error. Slower, approximate - but it's the
     method that scales to huge models (neural networks use it).

Plus RIDGE regularization: add a penalty  l2 * (sum of squared weights)
so the model can't lean too hard on any one feature. Helps when features
are highly correlated (our last-5 / last-10 / season averages are).

Run all linear models with:
    .venv\\Scripts\\python.exe -m src.models.linear
"""
import numpy as np
import pandas as pd

from src.evaluate import evaluate, get_split, load_data, score, show_leaderboard
from src.features import FEATURES


# =============================================================================
# Preprocessing: fill blanks + put every feature on the same scale
# =============================================================================
class Preprocessor:
    """
    1. Fill missing values with the training MEDIAN of that column.
    2. STANDARDIZE: (value - mean) / std, so every feature is centered at 0
       with spread 1. Without this, a feature measured in big numbers
       (defensive rating ~115) and one in small numbers (FT% ~0.78) would get
       wildly different-sized weights, and gradient descent would struggle.

    All the numbers (medians, means, stds) come from TRAINING data only.
    """

    def fit(self, X: pd.DataFrame):
        self.medians = X.median()
        filled = X.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std().replace(0, 1)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((X.fillna(self.medians) - self.means) / self.stds).to_numpy(dtype=float)


# =============================================================================
# Interactions: hand-made "this AND that" features
# =============================================================================
def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    A linear model can only ADD effects. But some effects MULTIPLY:
    a bad defense helps a 30-ppg star more (in points) than a 5-ppg bench player.
    So we create product features by hand. (Tree models in Phase 7 find these
    automatically - that's one of their big advantages.)
    """
    df = df.copy()
    df["X_EWM_x_OPP_DEF"] = df["PTS_EWM20"] * df["OPP_DEF_RTG"] / 115      # star x bad defense
    df["X_EWM_x_OPP_PACE"] = df["PTS_EWM20"] * df["OPP_PACE"] / 100       # star x fast game
    df["X_USG_x_OUT"] = df["USG_L10"] * df["TEAMMATES_OUT_PTS"]           # ball-dominant x teammates missing
    df["X_MIN_x_RATE"] = df["MIN_L10"] * df["PTS_PER_MIN_L20"]            # minutes x scoring rate
    df["X_EWM_x_GAP"] = df["PTS_EWM20"] * df["ABS_NET_RTG_GAP"]           # star x blowout risk
    return df


INTERACTIONS = ["X_EWM_x_OPP_DEF", "X_EWM_x_OPP_PACE", "X_USG_x_OUT", "X_MIN_x_RATE", "X_EWM_x_GAP"]


# =============================================================================
# The model
# =============================================================================
class LinearRegressionScratch:
    def __init__(self, features, method="normal", l2=0.0, lr=0.03, epochs=3000, interactions=False):
        self.features = list(features) + (INTERACTIONS if interactions else [])
        self.method, self.l2, self.lr, self.epochs = method, l2, lr, epochs
        self.interactions = interactions

    def _design_matrix(self, df):
        """Features -> standardized numbers, with a column of 1s in front for the intercept w0."""
        if self.interactions:
            df = add_interactions(df)
        X = self.prep.transform(df[self.features])
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, train):
        source = add_interactions(train) if self.interactions else train
        self.prep = Preprocessor().fit(source[self.features])
        X = self._design_matrix(train)
        y = train["PTS"].to_numpy(dtype=float)
        n, k = X.shape

        # Penalty matrix: penalize every weight EXCEPT the intercept.
        penalty = self.l2 * np.eye(k)
        penalty[0, 0] = 0

        if self.method == "normal":
            # Solve (X^T X + penalty) w = X^T y.  (np.linalg.solve is more
            # accurate than literally computing the inverse.)
            self.w = np.linalg.solve(X.T @ X + penalty, X.T @ y)

        elif self.method == "gd":
            self.w = np.zeros(k)
            self.loss_history = []
            for _ in range(self.epochs):
                error = X @ self.w - y                                   # how wrong we are, per row
                gradient = (2 / n) * (X.T @ error + penalty @ self.w)    # slope of the loss for each weight
                self.w -= self.lr * gradient                             # step downhill
                self.loss_history.append(np.mean(error ** 2))
        else:
            raise ValueError("method must be 'normal' or 'gd'")
        return self

    def predict(self, df):
        return self._design_matrix(df) @ self.w

    def coefficients(self) -> pd.Series:
        """Weights per feature. Because features are standardized, each weight
        = change in predicted points when that feature goes up by 1 std."""
        return pd.Series(self.w[1:], index=self.features)


# =============================================================================
# Feature sets, from tiny to everything
# =============================================================================
ONE_FEATURE = ["PTS_EWM20"]
CORE = ["PTS_EWM20", "PTS_L5", "PTS_SEASON", "MIN_L10", "USG_L10", "PTS_PER_MIN_L20",
        "TEAMMATES_OUT_PTS", "OPP_DEF_RTG", "OPP_PACE", "HOME", "REST_DAYS", "TEAM_GAMES_MISSED"]
ALL = FEATURES


def tune_l2(df, features, interactions, grid=(0, 10, 100, 1_000, 10_000, 100_000)):
    """Try several penalty strengths; pick the one with lowest VALIDATION MAE.
    (This is what the validation set is for. The test set stays locked.)"""
    results = {}
    for l2 in grid:
        m, _ = evaluate(LinearRegressionScratch(features, l2=l2, interactions=interactions),
                        f"ridge l2={l2}", df=df, save=False, verbose=False)
        results[l2] = m["MAE"]
    best = min(results, key=results.get)
    print("   ridge validation MAE by l2: " + ", ".join(f"{k}: {v:.4f}" for k, v in results.items())
          + f" -> best l2 = {best}")
    return best


def check_against_sklearn(df):
    """Sanity check: our from-scratch math should match scikit-learn's."""
    from sklearn.linear_model import Ridge
    train, valid = get_split(df, "valid")
    ours = LinearRegressionScratch(ALL, l2=100).fit(train)
    X = ours.prep.transform(train[ALL])
    theirs = Ridge(alpha=100).fit(X, train["PTS"])
    diff = np.max(np.abs(ours.w[1:] - theirs.coef_))
    print(f"\nSanity check vs scikit-learn Ridge: biggest weight difference = {diff:.2e} "
          f"({'MATCH' if diff < 1e-6 else 'MISMATCH!'})")


def main():
    df = load_data()
    evaluate(LinearRegressionScratch(ONE_FEATURE), "Linear: 1 feature (EWMA)", df=df)
    evaluate(LinearRegressionScratch(CORE), "Linear: 12 core features", df=df)
    evaluate(LinearRegressionScratch(ALL), "Linear: all 60 features", df=df)
    evaluate(LinearRegressionScratch(ALL, method="gd"), "Linear (gradient descent): all 60", df=df)

    best_l2 = tune_l2(df, ALL, interactions=True)
    evaluate(LinearRegressionScratch(ALL, l2=best_l2, interactions=True),
             "Ridge: all 60 + interactions", df=df)

    check_against_sklearn(df)
    print("\n=== LEADERBOARD (validation, 2024-25) ===")
    show_leaderboard("valid")


if __name__ == "__main__":
    main()
