from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"

DATA_PATH = RESOURCES_DIR / "data" / "ci_data.csv"
MONTHLY_FEATURES_PATH = RESOURCES_DIR / "data" / "monthly_features.csv"
MODEL_PATH = RESOURCES_DIR / "models" / "expense_forecast.joblib"

RANDOM_STATE = 42
FORECAST_HORIZON = 1  # месяцев вперёд
 