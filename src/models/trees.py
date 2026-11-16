"""
Phase 7 - Tree models.

A DECISION TREE predicts by asking a chain of yes/no questions:

    Is PTS_EWM20 > 14.2?
      no  -> Is MIN_L10 > 18.5? ...
      yes -> Is TEAMMATES_OUT_PTS > 12? ...
                 -> leaf: "predict 23.1" (the average of training games that ended up here)

Learning a tree = choosing the questions. At each step we try EVERY feature and
EVERY possible cut-off, and keep the one that makes the two groups most
"pure" (smallest squared error around each group's average). Then repeat
inside each group, until we reach max_depth or the groups get too small.

Why trees? They find "it depends" patterns on their own: a question about
teammates-out can appear ONLY on the star branch. Linear regression needed
hand-made interaction features for that.

One tree is jumpy, so we combine many:
    Random Forest     - many trees on random subsets of data, averaged (reduces jumpiness)
    Gradient Boosting - trees built ONE AFTER ANOTHER, each one fixing the
                        leftover errors of all the previous ones. (LightGBM is a
                        fast, industrial-strength version.)

Run with:
    .venv\\Scripts\\python.exe -m src.models.trees
"""
import numpy as np
import pandas as pd

from src.config import TRAIN_SEASONS
from src.evaluate import evaluate, load_data, show_leaderboard
from src.features import FEATURES


# =============================================================================
# Part 1: a regression tree, from scratch (works on plain NumPy arrays)
# =============================================================================
class TreeCore:
    def __init__(self, max_depth=6, min_samples_leaf=100):
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf

    def fit(self, X, y, sorted_orders=None):
        """
        X: (rows, features) numbers, no NaN.  y: target.
        sorted_orders: for each feature, the row indices sorted by that feature.
        Sorting is the slow part, so boosting computes it once and reuses it.
        """
        self.X, self.y = X, y
        self.n_features = X.shape[1]
        self.importance = np.zeros(self.n_features)       # total error reduction per feature
        # The tree is stored as parallel lists; node 0 is the root.
        self.feature, self.threshold, self.left, self.right, self.value = [], [], [], [], []
        if sorted_orders is None:
            sorted_orders = [np.argsort(X[:, j], kind="stable") for j in range(self.n_features)]
        self._grow(sorted_orders, depth=0)
        del self.X, self.y                                 # don't keep the training data around
        return self

    def _new_node(self, value):
        for lst, v in [(self.feature, -1), (self.threshold, np.nan), (self.left, -1), (self.right, -1), (self.value, value)]:
            lst.append(v)
        return len(self.value) - 1

    def _grow(self, orders, depth):
        rows = orders[0]                                   # any feature's order contains all rows of this node
        node = self._new_node(self.y[rows].mean())         # a leaf predicts the average of its rows
        if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf:
            return node

        split = self._best_split(orders)
        if split is None:
            return node
        j, threshold, gain = split
        self.importance[j] += gain

        # Partition every feature's sorted list into left/right, keeping them sorted.
        go_left = np.zeros(len(self.y), dtype=bool)
        go_left[rows[self.X[rows, j] <= threshold]] = True
        left_orders = [o[go_left[o]] for o in orders]
        right_orders = [o[~go_left[o]] for o in orders]

        self.feature[node], self.threshold[node] = j, threshold
        self.left[node] = self._grow(left_orders, depth + 1)
        self.right[node] = self._grow(right_orders, depth + 1)
        return node

    def _best_split(self, orders):
        """Try every feature and every cut-off; return the one that reduces squared error most."""
        msl = self.min_samples_leaf
        n = len(orders[0])
        total = self.y[orders[0]].sum()
        n_left = np.arange(1, n)                           # split after position i -> i+1 rows on the left
        n_right = n - n_left
        allowed_size = (n_left >= msl) & (n_right >= msl)
        base = total ** 2 / n
        best = None
        for j, order in enumerate(orders):
            xs, ys = self.X[order, j], self.y[order]
            left_sum = np.cumsum(ys)[:-1]
            # Squared error after the split = const - (left_sum^2/n_left + right_sum^2/n_right).
            # So maximizing this "score" = minimizing the error.
            score = left_sum ** 2 / n_left + (total - left_sum) ** 2 / n_right
            ok = allowed_size & (xs[:-1] != xs[1:])        # can only cut between DIFFERENT values
            if not ok.any():
                continue
            score = np.where(ok, score, -np.inf)
            i = int(np.argmax(score))
            gain = score[i] - base
            if gain > 1e-9 and (best is None or gain > best[2]):
                best = (j, (xs[i] + xs[i + 1]) / 2, gain)
        return best

    def predict(self, X):
        """Send every row down the tree at once."""
        out = np.empty(len(X))
        stack = [(0, np.arange(len(X)))]
        while stack:
            node, rows = stack.pop()
            if self.feature[node] == -1:                   # leaf
                out[rows] = self.value[node]
                continue
            goes_left = X[rows, self.feature[node]] <= self.threshold[node]
            stack += [(self.left[node], rows[goes_left]), (self.right[node], rows[~goes_left])]
        return out

    def rules(self, names, node=0, indent=""):
        """The tree as readable text."""
        if self.feature[node] == -1:
            return f"{indent}-> predict {self.value[node]:.1f}\n"
        name, t = names[self.feature[node]], self.threshold[node]
        return (f"{indent}{name} <= {t:.2f}?\n"
                + f"{indent}  yes:\n" + self.rules(names, self.left[node], indent + "    ")
                + f"{indent}  no:\n" + self.rules(names, self.right[node], indent + "    "))


# =============================================================================
# Part 2: gradient boosting, from scratch (uses TreeCore)
# =============================================================================
class BoostingCore:
    """
    Start by predicting the average for everyone. Then repeat:
        1. residual = actual - current prediction      (what we still get wrong)
        2. fit a small tree to the RESIDUALS            (learn the leftover pattern)
        3. prediction += learning_rate * tree's output  (fix a little of it)
    Many small, careful fixes add up to a strong model.
    """

    def __init__(self, n_rounds=150, learning_rate=0.1, max_depth=4, min_samples_leaf=200):
        self.n_rounds, self.lr = n_rounds, learning_rate
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf

    def fit(self, X, y):
        orders = [np.argsort(X[:, j], kind="stable") for j in range(X.shape[1])]   # sort once, reuse
        self.base = y.mean()
        pred = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.n_rounds):
            tree = TreeCore(self.max_depth, self.min_samples_leaf).fit(X, y - pred, orders)
            pred += self.lr * tree.predict(X)
            self.trees.append(tree)
        return self

    def staged_predict(self, X):
        """Prediction after 1, 2, ..., n_rounds trees (to watch the model improve)."""
        pred = np.full(len(X), self.base)
        for tree in self.trees:
            pred = pred + self.lr * tree.predict(X)
            yield pred

    def predict(self, X):
        *_, last = self.staged_predict(X)
        return last


# =============================================================================
# Part 3: wrappers so every model works with evaluate() (fit/predict on DataFrames)
# =============================================================================
class _ArrayModel:
    """Fills blanks with training medians, then hands plain arrays to `self.core`."""
    features = FEATURES

    def _X(self, df):
        return df[self.features].fillna(self.medians).to_numpy(dtype=float)

    def fit(self, train):
        self.medians = train[self.features].median()
        self.core.fit(self._X(train), train["PTS"].to_numpy(dtype=float))
        return self

    def predict(self, df):
        return self.core.predict(self._X(df))


class ScratchTree(_ArrayModel):
    def __init__(self, max_depth=8, min_samples_leaf=100):
        self.core = TreeCore(max_depth, min_samples_leaf)


class ScratchBoosting(_ArrayModel):
    def __init__(self, **kwargs):
        self.core = BoostingCore(**kwargs)


class SklearnTree(_ArrayModel):
    """scikit-learn's tree with the same settings - to check our from-scratch version."""

    def __init__(self, max_depth=8, min_samples_leaf=100):
        from sklearn.tree import DecisionTreeRegressor
        self.core = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)


class RandomForest(_ArrayModel):
    def __init__(self, n_trees=200, min_samples_leaf=50, max_features=0.33):
        from sklearn.ensemble import RandomForestRegressor
        self.core = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=min_samples_leaf,
                                          max_features=max_features, n_jobs=-1, random_state=0)


class LightGBM:
    """
    Industrial gradient boosting. Handles blanks (NaN) by itself.

    How many trees? Too few = underfit, too many = overfit. We decide with
    EARLY STOPPING, using only training seasons: train on the older seasons,
    watch the error on the newest training season, stop when it stops improving.
    Then retrain on all training seasons with that number of trees.
    (The validation season is never used for this - it stays a fair test.)

    HYBRID (hybrid=True): trees are great at "it depends" patterns but clumsy at
    straight-line relationships (they approximate a line with a staircase).
    Points are mostly a straight line in a player's averages. So: let the Phase 5
    linear model make the base prediction, and let the trees learn only the
    LEFTOVER errors. Each model does what it's good at.
    """

    def __init__(self, objective="regression", hybrid=False, **params):
        self.params = {"objective": objective, "learning_rate": 0.03, "num_leaves": 31,
                       "min_data_in_leaf": 200, "feature_fraction": 0.7, "bagging_fraction": 0.8,
                       "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1, "seed": 0, **params}
        self.hybrid = hybrid
        self.features = FEATURES

    def _base(self, fit_on, predict_on):
        """Linear model's prediction (the starting point), or None if not hybrid."""
        if not self.hybrid:
            return None
        from src.models.linear import LinearRegressionScratch
        return LinearRegressionScratch(FEATURES, interactions=True).fit(fit_on).predict(predict_on)

    def fit(self, train):
        import lightgbm as lgb
        newest = sorted(train["SEASON"].unique())[-1]
        inner_train, inner_valid = train[train["SEASON"] != newest], train[train["SEASON"] == newest]
        probe = lgb.train(
            self.params,
            lgb.Dataset(inner_train[self.features], inner_train["PTS"], init_score=self._base(inner_train, inner_train)),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(inner_valid[self.features], inner_valid["PTS"],
                                    init_score=self._base(inner_train, inner_valid))],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        # Slightly more trees for the bigger final training set.
        self.n_trees = int(probe.best_iteration * 1.1)
        if self.hybrid:
            from src.models.linear import LinearRegressionScratch
            self.linear = LinearRegressionScratch(FEATURES, interactions=True).fit(train)
        init = self.linear.predict(train) if self.hybrid else None
        self.booster = lgb.train(self.params, lgb.Dataset(train[self.features], train["PTS"], init_score=init),
                                 num_boost_round=self.n_trees)
        return self

    def predict(self, df):
        trees = self.booster.predict(df[self.features])
        return trees + self.linear.predict(df) if self.hybrid else trees

    def importance(self):
        gain = self.booster.feature_importance("gain")
        return pd.Series(gain / gain.sum(), index=self.features).sort_values(ascending=False)


def main():
    df = load_data()
    import time

    for name, model in [
        ("Tree from scratch (depth 8)", ScratchTree(8, 100)),
        ("sklearn tree (depth 8) - check", SklearnTree(8, 100)),
        ("Boosting from scratch (150 trees)", ScratchBoosting(n_rounds=150)),
        ("Random forest (200 trees)", RandomForest()),
        ("LightGBM (squared error)", LightGBM("regression")),
        ("LightGBM (absolute error)", LightGBM("regression_l1")),
        ("Hybrid: linear + LightGBM (abs. error)", LightGBM("regression_l1", hybrid=True)),
    ]:
        start = time.time()
        evaluate(model, name, df=df)
        extra = f", {model.n_trees} trees" if isinstance(model, LightGBM) else ""
        print(f"   ({time.time() - start:.0f}s{extra})")

    print("\n=== LEADERBOARD (validation, 2024-25) ===")
    show_leaderboard("valid")


if __name__ == "__main__":
    main()
