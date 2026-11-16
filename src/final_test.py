"""
Phase 10 - The ONE-TIME final test.

Until now, every choice (features, settings, blend weights) was made using the
2021-24 training seasons and the 2024-25 validation season. The 2025-26 season
has been locked away. Now each finalist is retrained on 2021-25 and scored on
2025-26 exactly once. This is the honest estimate of how SwishAI performs on
games it has never seen.

Run with:
    .venv\\Scripts\\python.exe -m src.final_test
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR
from src.evaluate import evaluate, get_split, load_data
from src.features import FEATURES
from src.models.baselines import LeagueAverage, TunedEWMA
from src.models.blend import BLEND_WEIGHTS, Blend
from src.models.distributions import DistributionModel, distribution_scores
from src.models.linear import LinearRegressionScratch
from src.models.neural_net import NNEnsemble
from src.models.simulate import GameSimulator
from src.models.trees import LightGBM

FINALISTS = [
    ("League average (same for everyone)", lambda: LeagueAverage()),
    ("Tuned weighted avg (EWMA)", lambda: TunedEWMA()),
    ("Ridge: all 60 + interactions", lambda: LinearRegressionScratch(FEATURES, interactions=True)),
    ("Hybrid: linear + LightGBM (abs. error)", lambda: LightGBM("regression_l1", hybrid=True)),
    ("NN ensemble: 5 nets + embeddings (abs. error)", lambda: NNEnsemble(5, loss="l1", embeddings=True)),
    ("Simulation, anchored (median)", lambda: GameSimulator()),
    ("Blend: hybrid + NN ensemble + simulation", lambda: Blend(BLEND_WEIGHTS)),
]


def main():
    df = load_data()
    rows = []
    trained = {}
    for name, make in FINALISTS:
        start = time.time()
        model = make()
        metrics, _ = evaluate(model, name, split="test", df=df, verbose=False)
        trained[name] = model
        rows.append({"model": name, **metrics})
        print(f"{name:<46} test MAE {metrics['MAE']:.3f}   ({time.time() - start:.0f}s)")

    test = pd.DataFrame(rows).set_index("model")
    valid = pd.read_csv(RESULTS_DIR / "leaderboard.csv").query("split == 'valid'").set_index("model")["MAE"]
    test.insert(0, "valid MAE", valid.reindex(test.index))
    test.to_csv(RESULTS_DIR / "final_test.csv")

    # Distribution quality on the test season.
    train, test_rows = get_split(df, "test")
    dist = {}
    for name, model in [("Normal, spread grows with mean", DistributionModel("normal_scaled").fit(train)),
                        ("Simulation", trained["Simulation, anchored (median)"])]:
        dist[name] = distribution_scores(model.pmf(test_rows), test_rows)
    dist = pd.DataFrame(dist).T
    dist.to_csv(RESULTS_DIR / "final_test_distributions.csv")

    print("\n=== FINAL TEST: 2025-26 (never used for any decision) ===")
    print(test[["valid MAE", "MAE", "RMSE", "BIAS", "R2", "MAE bench (<10 ppg)", "MAE mid (10-20)", "MAE star (20+)"]].round(3).to_string())
    print("\nDistributions on the test season:")
    print(dist.round(4).to_string())

    # Chart
    plot = test.drop(index="League average (same for everyone)").sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(len(plot))
    ax.barh(y + 0.2, plot["valid MAE"], height=0.38, color="#c3c2b7", label="validation (2024-25)")
    ax.barh(y - 0.2, plot["MAE"], height=0.38, color="#2a78d6", label="final test (2025-26)")
    for i, v in enumerate(plot["MAE"]):
        ax.text(v + 0.005, i - 0.2, f"{v:.3f}", va="center", color="#52514e", fontsize=9)
    ax.set_yticks(y, plot.index)
    ax.set_xlim(4.3, 4.8)
    ax.set_xlabel("MAE (points per game missed) - lower is better", color="#52514e")
    ax.set_title("SwishAI final test: validation vs locked test season", fontweight="bold")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e7e2")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "final_test.png", dpi=120)
    print(f"\nSaved results/final_test.csv, results/final_test_distributions.csv, results/final_test.png")


if __name__ == "__main__":
    main()
