from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "ci_data.csv"
MODEL_PATH = BASE_DIR / "models" / "transaction_classifier.joblib"
UNLABELED_DATA_PATH = BASE_DIR / "data" / "unlabeled.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TYPE = "neural"  # варианты: "simple", "advanced", "ensemble", "catboost", "neural"
USE_LABEL_PROPAGATION = False
LABEL_PROPAGATION_CONFIDENCE = 0.9
USE_OVERSAMPLING = False
OVERSAMPLING_MAX_PER_CLASS = None  # или число, если нужно ограничение
