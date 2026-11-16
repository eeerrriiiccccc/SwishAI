"""
Phase 4 - Baselines: the simplest possible guesses.

Baselines don't use machine learning. They answer: "how good can you get
with common sense?" Every fancier model must beat the best baseline, or
it isn't worth its complexity.

Run all baselines with:
    .venv\\Scripts\\python.exe -m src.models.baselines
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR
from src.evaluate import evaluate, load_data, score, show_leaderboard


class LeagueAverage:
    """Guess the same number for everyone: the average of all training games.
    This is the 'know nothing about the player' floor."""

    def fit(self, train):
        self.value = train["PTS"].mean()   # the ONLY thing this model learns

    def predict(self, df):
        return np.full(len(df), self.value)


class ColumnBaseline:
    """Predict a pre-computed average directly, e.g. his last-10-game average.
    Nothing to learn - fit() does nothing."""

    def __init__(self, column):
        self.column = column

    def fit(self, train):
        pass

    def predict(self, df):
        return df[self.column].to_numpy()


class MinutesTimesRate:
    """points = minutes x points-per-minute.
    Expected minutes = last-10 average; scoring rate = last-20 points per minute.
    Splitting the problem in two like this is an idea we'll come back to."""

    def fit(self, train):
        pass

    def predict(self, df):
        return (df["MIN_L10"] * df["PTS_PER_MIN_L20"]).to_numpy()


class TunedEWMA:
    """Exponentially weighted average where the 'span' (how fast old games
    fade) is CHOSEN using the training data.

    This is our first taste of learning: try each option on the training set,
    keep the one with the lowest error. (We never look at the validation set
    to choose - that would be cheating.)"""

    def __init__(self, spans=(5, 10, 20)):
        self.spans = spans

    def fit(self, train):
        errors = {s: score(train["PTS"], train[f"PTS_EWM{s}"])["MAE"] for s in self.spans}
        self.best_span = min(errors, key=errors.get)
        print(f"   (TunedEWMA training MAE by span: "
              + ", ".join(f"{s}: {e:.3f}" for s, e in errors.items())
              + f" -> picked span {self.best_span})")

    def predict(self, df):
        return df[f"PTS_EWM{self.best_span}"].to_numpy()


def plot_leaderboard(board, out_file):
    board = board.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(board) + 1.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    best = board["MAE"].min()
    colors = ["#eb6834" if m == best else "#2a78d6" for m in board["MAE"]]
    ax.barh(board["model"], board["MAE"], color=colors, height=0.6)
    for i, m in enumerate(board["MAE"]):
        ax.text(m + 0.05, i, f"{m:.2f}", va="center", color="#52514e")
    ax.set_xlabel("MAE on 2024-25 (points per game missed) - lower is better", color="#52514e")
    ax.set_title("Baselines: the scores to beat", fontweight="bold")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e7e2")
    plt.tight_layout()
    fig.savefig(out_file, dpi=120)
    print(f"\nChart saved -> {out_file}")


def main():
    df = load_data()
    print(f"Scoring on 2024-25: {len(df[df.SEASON == '2024-25']):,} player-games\n")

    models = {
        "League average (same for everyone)": LeagueAverage(),
        "Last 3 games avg": ColumnBaseline("PTS_L3"),
        "Last 5 games avg": ColumnBaseline("PTS_L5"),
        "Last 10 games avg": ColumnBaseline("PTS_L10"),
        "Last 30 games avg": ColumnBaseline("PTS_L30"),
        "Season avg so far": ColumnBaseline("PTS_SEASON"),
        "Minutes x points-per-minute": MinutesTimesRate(),
        "Tuned weighted avg (EWMA)": TunedEWMA(),
    }
    for name, model in models.items():
        evaluate(model, name, split="valid", df=df)

    print("\n=== LEADERBOARD (validation, 2024-25) ===")
    board = show_leaderboard("valid")
    plot_leaderboard(board, RESULTS_DIR / "baselines_valid.png")


if __name__ == "__main__":
    main()
