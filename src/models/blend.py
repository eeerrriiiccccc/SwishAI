"""
Phase 10 - Blending: a weighted average of our best, most DIFFERENT models.

    hybrid  - linear regression + LightGBM trees (Phase 7)
    nn      - 5 averaged neural networks with player embeddings (Phase 9)
    sim     - median of the Monte Carlo game simulation (Phase 8)

Why blend? Each model makes somewhat different mistakes. Averaging them cancels
part of each one's errors (the same reason averaging 5 networks helped).

The weights are chosen on the VALIDATION season (2024-25). The test season
(2025-26) is never used for any choice.

Run with:
    .venv\\Scripts\\python.exe -m src.models.blend
"""
import itertools

import numpy as np

from src.evaluate import evaluate, get_split, load_data, score
from src.models.neural_net import NNEnsemble
from src.models.simulate import GameSimulator
from src.models.trees import LightGBM


# Chosen by choose_weights() on the 2024-25 validation season (python -m src.models.blend):
#   hybrid alone 4.4620, nn alone 4.4666, sim alone 4.4647, equal thirds 4.4585, these weights 4.4577
BLEND_WEIGHTS = {"hybrid": 0.5, "nn": 0.4, "sim": 0.1}


def make_components():
    return {
        "hybrid": LightGBM("regression_l1", hybrid=True),
        "nn": NNEnsemble(5, loss="l1", embeddings=True),
        "sim": GameSimulator(),
    }


class Blend:
    def __init__(self, weights):
        self.weights = weights                     # e.g. {"hybrid": 0.4, "nn": 0.4, "sim": 0.2}

    def fit(self, train):
        self.components = {k: m.fit(train) for k, m in make_components().items() if self.weights.get(k, 0) > 0}
        return self

    def predict(self, df):
        return sum(w * self.components[k].predict(df) for k, w in self.weights.items() if w > 0)


def choose_weights(df):
    """Fit each component once on 2021-24, then try every weight combination (steps of 0.1) on 2024-25."""
    train, valid = get_split(df, "valid")
    preds = {}
    for name, model in make_components().items():
        print(f"  training {name} ...")
        preds[name] = model.fit(train).predict(valid)

    best = None
    for w_h, w_n in itertools.product(np.arange(0, 1.01, 0.1), repeat=2):
        w_s = 1 - w_h - w_n
        if w_s < -1e-9:
            continue
        weights = {"hybrid": round(float(w_h), 1), "nn": round(float(w_n), 1), "sim": round(float(max(w_s, 0)), 1)}
        mae = score(valid["PTS"], sum(w * preds[k] for k, w in weights.items()))["MAE"]
        if best is None or mae < best[1]:
            best = (weights, mae)

    for name in preds:
        print(f"  {name:<7} alone: validation MAE {score(valid['PTS'], preds[name])['MAE']:.4f}")
    print(f"  equal 1/3 each: validation MAE {score(valid['PTS'], np.mean(list(preds.values()), axis=0))['MAE']:.4f}")
    print(f"  best weights {best[0]}: validation MAE {best[1]:.4f}")
    return best[0]


def main():
    df = load_data()
    weights = choose_weights(df)
    evaluate(Blend(weights), "Blend: hybrid + NN ensemble + simulation", df=df)
    return weights


if __name__ == "__main__":
    main()
