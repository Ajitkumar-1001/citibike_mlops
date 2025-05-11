import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Define the project root directory (Citibike_prediction_aml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define directories relative to the project root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

FEATURE_GROUP_NAME = "time_series_hourly_feature_group_citi_bike"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_hourly_feature_view_citi_bike"
FEATURE_VIEW_VERSION = 1

MODEL_NAME = "citibike_hourly_predictor_model"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "citibike_model_prediction"