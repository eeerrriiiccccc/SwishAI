"""
Phase 3 - The evaluation harness: ONE fair way to score every model.

Every model in this project, from "use his season average" to the neural
network, is scored by the function `evaluate()` below. Same data, same
split, same metrics, so the comparison is always apples to apples.

What a "model" must look like
-----------------------------
Any object with two methods:

    model.fit(train_df)          # learn from the training rows
    model.predict(df) -> array   # one predicted point total per row

That's it. `train_df` and `df` are rows of data/processed/features.parquet.

The split (by TIME, never random)
---------------------------------
    split="valid": learn from 2021-24, score on 2024-25   <- use this while building
    split="test":  learn from 2021-25, score on 2025-26   <- only at the very end!

Why keep the test set locked away? Every time you look at a score and tweak
your model, you "learn" a little from that data. If we tune on the test set,
its score stops being an honest estimate of the real world.
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RESULTS_DIR, TEST_SEASONS, TRAIN_SEASONS, VALID_SEASONS

LEADERBOARD = RESULTS_DIR / "leaderboard.csv"

# We only predict for players with at least this many previous games in our data.
# (With 0-4 games of history, "his recent average" barely exists.)
MIN_GAMES = 5

# Player tiers, by scoring average so far - errors are naturally bigger for stars.
TIER_BINS = [-1, 10, 20, 100]
TIER_NAMES = ["bench (<10 ppg)", "mid (10-20)", "star (20+)"]


def load_data() -> pd.DataFrame:
    """The feature table, limited to rows we can fairly predict."""
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    return df[df["GAMES_PLAYED_BEFORE"] >= MIN_GAMES].reset_index(drop=True)


def get_split(df: pd.DataFrame, split: str):
    """Return (train rows, rows to score)."""
    if split == "valid":
        train_seasons, eval_seasons = TRAIN_SEASONS, VALID_SEASONS
    elif split == "test":
        train_seasons, eval_seasons = TRAIN_SEASONS + VALID_SEASONS, TEST_SEASONS
    else:
        raise ValueError("split must be 'valid' or 'test'")
    train = df[df["SEASON"].isin(train_seasons)].reset_index(drop=True)
    evaluation = df[df["SEASON"].isin(eval_seasons)].reset_index(drop=True)
    return train, evaluation


def score(actual, predicted) -> dict:
    """
    The metrics, in plain English:
      MAE  - Mean Absolute Error: on average, how many points we miss by. Our main metric.
      RMSE - Root Mean Squared Error: like MAE, but punishes BIG misses extra hard.
      BIAS - average (prediction - actual). Positive = we guess too high on average.
      R2   - share of the game-to-game variation we explain. 1.0 = perfect, 0 = no better than one constant guess.
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    error = predicted - actual
    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "BIAS": np.mean(error),
        "R2": 1 - np.sum(error ** 2) / np.sum((actual - actual.mean()) ** 2),
    }


def evaluate(model, name: str, split: str = "valid", df: pd.DataFrame | None = None,
             save: bool = True, verbose: bool = True):
    """Train `model`, score it, print + save the result. Returns (metrics, predictions)."""
    if split == "test" and verbose:
        print("NOTE: scoring on the TEST set - only do this for final results!")
    if df is None:
        df = load_data()
    train, evaluation = get_split(df, split)

    model.fit(train)
    predictions = np.asarray(model.predict(evaluation), dtype=float)
    if len(predictions) != len(evaluation):
        raise ValueError(f"{name}: got {len(predictions)} predictions for {len(evaluation)} rows")
    if np.isnan(predictions).any():
        raise ValueError(f"{name}: {np.isnan(predictions).sum()} predictions are NaN")

    metrics = score(evaluation["PTS"], predictions)

    # Error broken down by player tier.
    tier = pd.cut(evaluation["PTS_SEASON"], TIER_BINS, labels=TIER_NAMES)
    abs_err = pd.Series(np.abs(predictions - evaluation["PTS"].to_numpy()))
    for t, mae in abs_err.groupby(tier.to_numpy(), observed=True).mean().items():
        metrics[f"MAE {t}"] = mae

    if verbose:
        print(f"{name:<38} MAE {metrics['MAE']:.3f} | RMSE {metrics['RMSE']:.3f} | "
              f"bias {metrics['BIAS']:+.2f} | R2 {metrics['R2']:.3f}")
    if save:
        _save(name, split, metrics, len(evaluation))
    return metrics, predictions


def _save(name, split, metrics, n_rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"model": name, "split": split, "rows": n_rows,
                         **{k: round(v, 4) for k, v in metrics.items()}}])
    if LEADERBOARD.exists():
        board = pd.read_csv(LEADERBOARD)
        board = board[~((board["model"] == name) & (board["split"] == split))]  # replace old entry
        row = pd.concat([board, row], ignore_index=True)
    row.to_csv(LEADERBOARD, index=False)


def show_leaderboard(split: str = "valid") -> pd.DataFrame:
    """All saved results for a split, best (lowest MAE) first."""
    board = pd.read_csv(LEADERBOARD)
    board = board[board["split"] == split].sort_values("MAE").reset_index(drop=True)
    board.index += 1
    cols = ["model", "MAE", "RMSE", "BIAS", "R2"] + [f"MAE {t}" for t in TIER_NAMES]
    print(board[cols].round(3).to_string())
    return board
