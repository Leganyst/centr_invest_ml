from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ci_data.csv"
MODEL_PATH = BASE_DIR / "models" / "transaction_classifier.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TYPE = "catboost"  # варианты: "simple", "advanced", "ensemble", "catboost"
