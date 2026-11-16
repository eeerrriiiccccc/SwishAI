"""
Central settings for SwishAI.

Every other file imports from here, so if you want to change which seasons
we use or where data is stored, you only change it in ONE place.
"""
from pathlib import Path

# --- Folders -----------------------------------------------------------------
# Path(__file__) is this file; .parent.parent walks up to the SwishAI folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Seasons -----------------------------------------------------------------
# NBA seasons are written like "2023-24". We use five complete seasons.
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

# How we split by TIME (never randomly - see the README for why):
TRAIN_SEASONS = ["2021-22", "2022-23", "2023-24"]  # the model learns from these
VALID_SEASONS = ["2024-25"]                        # we tune choices on this
TEST_SEASONS = ["2025-26"]                         # touched only for final scores

# --- Live predictions (src/predict.py) ------------------------------------------
CURRENT_SEASON = "2026-27"                         # the season in progress
CURRENT_DIR = RAW_DIR / "current"                  # re-downloaded daily, never used for training
MODELS_DIR = PROJECT_ROOT / "models"               # saved trained models
