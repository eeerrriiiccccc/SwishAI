"""
Phase 9 - Neural networks.

A neural network is linear regression, stacked, with a bend in between:

    layer 1:  h1 = relu(X  @ W1 + b1)     60 features -> 64 numbers
    layer 2:  h2 = relu(h1 @ W2 + b2)     64 -> 32
    output:   y  =      h2 @ W3 + b3      32 -> 1 prediction

relu(z) = max(z, 0) is the "bend". Without it, stacking linear layers would just
give... one big linear regression. With it, the network can learn curves and
"it depends" patterns, like trees do, but smoothly.

Learning = gradient descent again (Phase 5!). The new part is BACKPROPAGATION:
using the chain rule to work out, layer by layer from the output backwards,
how much each weight contributed to the error.

Models in this file:
    MLPScratch    - the network above in pure NumPy, backprop written by hand
    TorchModel    - PyTorch versions:
                      * "wide & deep": a linear part + a deep part, added together
                        (the Phase 7 lesson: the straight-line part is best left to a linear model)
                      * optional PLAYER EMBEDDINGS: a learned 8-number "fingerprint" per player
                      * optional "gauss" loss: the network predicts BOTH the expected points
                        AND its own uncertainty (sigma) for every game

Run with:
    .venv\\Scripts\\python.exe -m src.models.neural_net
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluate import evaluate, get_split, load_data
from src.features import FEATURES
from src.models.distributions import distribution_scores, normal_pmf
from src.models.linear import Preprocessor


def inner_split(train):
    """Hold out the newest TRAINING season to decide when to stop training (early stopping)."""
    newest = sorted(train["SEASON"].unique())[-1]
    return train[train["SEASON"] != newest], train[train["SEASON"] == newest]


# =============================================================================
# Part 1: a neural network from scratch (NumPy)
# =============================================================================
class MLPScratch:
    def __init__(self, hidden=(64, 32), lr=1e-3, max_epochs=40, batch_size=512, patience=4, seed=0):
        self.hidden, self.lr, self.max_epochs = hidden, lr, max_epochs
        self.batch_size, self.patience, self.seed = batch_size, patience, seed

    # ---------------------------------------------------------- the network
    def _init_params(self, n_in, rng):
        """He initialization: random weights scaled so signals don't explode or vanish through ReLU layers."""
        sizes = [n_in, *self.hidden, 1]
        self.params = {}
        for i in range(len(sizes) - 1):
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
            self.params[f"b{i}"] = np.zeros(sizes[i + 1])
        self.n_layers = len(sizes) - 1

    def _forward(self, X):
        """Returns the prediction and a cache of every layer's values (needed for backprop)."""
        cache = {"a0": X}
        a = X
        for i in range(self.n_layers):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = np.maximum(z, 0) if i < self.n_layers - 1 else z      # ReLU on hidden layers only
            cache[f"z{i}"], cache[f"a{i + 1}"] = z, a
        return a[:, 0], cache

    def _backward(self, pred, y, cache):
        """
        BACKPROPAGATION. Loss = mean((pred - y)^2). Walk backwards with the chain rule:
          d loss / d pred     = 2 (pred - y) / n
          d loss / d W        = (layer input)^T @ (d loss / d layer output)
          d loss / d input    = (d loss / d layer output) @ W^T
          through ReLU        : the gradient passes where z > 0, and is blocked where z <= 0
        """
        grads = {}
        delta = (2 * (pred - y) / len(y))[:, None]                     # d loss / d output
        for i in reversed(range(self.n_layers)):
            grads[f"W{i}"] = cache[f"a{i}"].T @ delta
            grads[f"b{i}"] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.params[f"W{i}"].T) * (cache[f"z{i - 1}"] > 0)
        return grads

    def _adam_step(self, grads):
        """Adam: gradient descent with momentum + a per-weight step size. The standard optimizer."""
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g                  # running average of the gradient
            self.v[k] = b2 * self.v[k] + (1 - b2) * g ** 2             # running average of its size
            m_hat, v_hat = self.m[k] / (1 - b1 ** self.t), self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def _train(self, X, y, epochs, X_val=None, y_val=None):
        rng = np.random.default_rng(self.seed)
        self._init_params(X.shape[1], rng)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.history = []
        best, best_epoch, best_params = np.inf, 0, None
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(y))                            # shuffle every epoch
            for start in range(0, len(y), self.batch_size):
                batch = order[start:start + self.batch_size]
                pred, cache = self._forward(X[batch])
                self._adam_step(self._backward(pred, y[batch], cache))
            if X_val is not None:
                val_loss = np.mean((self._forward(X_val)[0] - y_val) ** 2)
                self.history.append(val_loss)
                if val_loss < best:
                    best, best_epoch = val_loss, epoch
                    best_params = {k: v.copy() for k, v in self.params.items()}
                elif epoch - best_epoch >= self.patience:
                    break
        if best_params is not None:
            self.params = best_params
        return best_epoch

    # ------------------------------------------------- evaluate() interface
    def fit(self, train):
        inner_train, inner_valid = inner_split(train)
        self.prep = Preprocessor().fit(train[FEATURES])
        self.y_mean, self.y_std = train["PTS"].mean(), train["PTS"].std()
        X = lambda d: self.prep.transform(d[FEATURES])
        ys = lambda d: ((d["PTS"] - self.y_mean) / self.y_std).to_numpy()   # train on standardized points
        # 1. Find the best number of epochs (early stopping on the held-out training season).
        self.best_epoch = self._train(X(inner_train), ys(inner_train), self.max_epochs, X(inner_valid), ys(inner_valid))
        self.inner_history = self.history
        # 2. Retrain on ALL training seasons for that many epochs.
        self._train(X(train), ys(train), self.best_epoch)
        return self

    def predict(self, df):
        return self._forward(self.prep.transform(df[FEATURES]))[0] * self.y_std + self.y_mean


def gradient_check(seed=0):
    """
    Is our hand-written backprop correct? Compare it with a brute-force estimate:
    nudge each weight by +/- a tiny amount and measure how the loss changes.
    Returns the largest relative difference (anything below ~1e-6 means correct).
    """
    rng = np.random.default_rng(seed)
    net = MLPScratch(hidden=(5, 4))
    net._init_params(3, rng)
    X, y = rng.normal(size=(20, 3)), rng.normal(size=20)
    pred, cache = net._forward(X)
    analytic = net._backward(pred, y, cache)
    worst, h = 0.0, 1e-6
    for k, W in net.params.items():
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + h
            plus = np.mean((net._forward(X)[0] - y) ** 2)
            W[idx] = old - h
            minus = np.mean((net._forward(X)[0] - y) ** 2)
            W[idx] = old
            numeric = (plus - minus) / (2 * h)
            worst = max(worst, abs(numeric - analytic[k][idx]) / max(1e-8, abs(numeric) + abs(analytic[k][idx])))
    return worst


# =============================================================================
# Part 2: PyTorch
# =============================================================================
class WideAndDeep(nn.Module):
    """wide = a linear-regression layer; deep = hidden layers with ReLU. Output = wide + deep."""

    def __init__(self, n_in, n_players, emb_dim, hidden, dropout, n_out):
        super().__init__()
        self.embedding = nn.Embedding(n_players, emb_dim) if n_players else None
        width = n_in + (emb_dim if n_players else 0)
        self.wide = nn.Linear(width, n_out)                           # the linear-regression part
        layers, prev = [], width
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.deep = nn.Sequential(*layers, nn.Linear(prev, n_out))    # the "it depends" part

    def forward(self, x, player):
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(player)], dim=1)
        return self.wide(x) + self.deep(x)


class TorchModel:
    """
    loss="mse"   predicts the average           (like Phase 5)
    loss="l1"    predicts the median, best MAE  (like LightGBM 'absolute error')
    loss="gauss" predicts mean AND sigma per game; trained by maximum likelihood (like Phase 6,
                 but every game gets its own spread, learned from all features + player embedding)

    embeddings=True: each player gets a learned vector of `emb_dim` numbers. Players never seen
    in training (e.g. rookies) share one "unknown player" vector. During training we randomly
    hide 10% of player identities so that "unknown" vector learns to mean "a typical player like this".
    """

    def __init__(self, loss="l1", embeddings=False, emb_dim=8, hidden=(128, 64), dropout=0.1,
                 lr=1e-3, weight_decay=1e-5, max_epochs=60, batch_size=512, patience=6, seed=0):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    # ------------------------------------------------------------ helpers
    def _player_index(self, df):
        return df["PLAYER_ID"].map(self.player_map).fillna(0).astype(int).to_numpy() if self.embeddings \
            else np.zeros(len(df), dtype=int)

    def _tensors(self, df, with_target=True):
        import torch
        X = torch.tensor(self.prep.transform(df[FEATURES]), dtype=torch.float32)
        P = torch.tensor(self._player_index(df), dtype=torch.long)
        if not with_target:
            return X, P
        y = torch.tensor(((df["PTS"] - self.y_mean) / self.y_std).to_numpy(), dtype=torch.float32)
        return X, P, y

    def _loss(self, out, y):
        import torch
        if self.loss == "mse":
            return ((out[:, 0] - y) ** 2).mean()
        if self.loss == "l1":
            return (out[:, 0] - y).abs().mean()
        log_sigma = out[:, 1].clamp(-4, 3)
        return (log_sigma + 0.5 * ((y - out[:, 0]) / torch.exp(log_sigma)) ** 2).mean()   # Gaussian NLL

    def _train(self, train_df, epochs, valid_df=None):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_out = 2 if self.loss == "gauss" else 1
        n_players = len(self.player_map) + 1 if self.embeddings else 0
        self.net = WideAndDeep(len(FEATURES), n_players, self.emb_dim, self.hidden, self.dropout, n_out)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        X, P, y = self._tensors(train_df)
        if valid_df is not None:
            Xv, Pv, yv = self._tensors(valid_df)
        best, best_epoch, best_state, self.history = np.inf, 0, None, []
        for epoch in range(1, epochs + 1):
            self.net.train()
            order = torch.tensor(rng.permutation(len(y)))
            for start in range(0, len(y), self.batch_size):
                b = order[start:start + self.batch_size]
                players = P[b].clone()
                if self.embeddings:                                     # hide 10% of identities
                    players[torch.rand(len(b)) < 0.1] = 0
                loss = self._loss(self.net(X[b], players), y[b])
                opt.zero_grad()
                loss.backward()                                         # PyTorch does backprop for us
                opt.step()
            if valid_df is not None:
                self.net.eval()
                with torch.no_grad():
                    val = self._loss(self.net(Xv, Pv), yv).item()
                self.history.append(val)
                if val < best:
                    best, best_epoch = val, epoch
                    best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                elif epoch - best_epoch >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return best_epoch

    # ------------------------------------------------- evaluate() interface
    def fit(self, train):
        self.prep = Preprocessor().fit(train[FEATURES])
        self.y_mean, self.y_std = train["PTS"].mean(), train["PTS"].std()
        inner_train, inner_valid = inner_split(train)
        # Early stopping run: only players seen in inner_train get their own embedding.
        self.player_map = {p: i + 1 for i, p in enumerate(inner_train["PLAYER_ID"].unique())}
        self.best_epoch = self._train(inner_train, self.max_epochs, inner_valid)
        self.inner_history = self.history
        # Final run on all training seasons.
        self.player_map = {p: i + 1 for i, p in enumerate(train["PLAYER_ID"].unique())}
        self._train(train, self.best_epoch)
        return self

    def _raw(self, df):
        import torch
        self.net.eval()
        with torch.no_grad():
            return self.net(*self._tensors(df, with_target=False)).numpy()

    def predict(self, df):
        return self._raw(df)[:, 0] * self.y_std + self.y_mean

    def mu_sigma(self, df):
        out = self._raw(df)
        mu = np.clip(out[:, 0] * self.y_std + self.y_mean, 0.3, None)
        sigma = np.exp(np.clip(out[:, 1], -4, 3)) * self.y_std
        return mu, sigma

    def pmf(self, df):
        return normal_pmf(*self.mu_sigma(df))


class NNEnsemble:
    """
    Train the same network N times with different random starting weights, then average.
    Each single network is a bit "lucky" or "unlucky" in its own way (~+/-0.01 MAE);
    averaging cancels much of that out. Cheap and reliable.
    """

    def __init__(self, n=5, **kwargs):
        self.models = [TorchModel(seed=s, **kwargs) for s in range(n)]

    def fit(self, train):
        for m in self.models:
            m.fit(train)
        self.best_epoch = round(np.mean([m.best_epoch for m in self.models]))
        return self

    def predict(self, df):
        return np.mean([m.predict(df) for m in self.models], axis=0)


def main():
    import time
    print(f"Gradient check (hand-written backprop vs brute force): max relative error = {gradient_check():.1e}\n")
    df = load_data()
    for name, model in [
        ("NN from scratch (NumPy, 64-32)", MLPScratch()),
        ("NN PyTorch, wide & deep (squared error)", TorchModel(loss="mse")),
        ("NN PyTorch, wide & deep (abs. error)", TorchModel(loss="l1")),
        ("NN + player embeddings (abs. error)", TorchModel(loss="l1", embeddings=True)),
        ("NN ensemble: 5 nets + embeddings (abs. error)", NNEnsemble(5, loss="l1", embeddings=True)),
        ("NN + embeddings, predicts mean & sigma", TorchModel(loss="gauss", embeddings=True)),
    ]:
        start = time.time()
        evaluate(model, name, df=df)
        print(f"   ({time.time() - start:.0f}s, {model.best_epoch} epochs)")

    # Distribution quality of the mean & sigma network vs the earlier best.
    train, valid = get_split(df, "valid")
    s = distribution_scores(model.pmf(valid), valid)
    print("\nNN mean & sigma, distribution scores:", {k: round(v, 4) for k, v in s.items()})
    print("(Phase 8 simulation: LOG_SCORE -2.9554, COVER_80 0.798, BRIER_OVER 0.2364)")


if __name__ == "__main__":
    main()
