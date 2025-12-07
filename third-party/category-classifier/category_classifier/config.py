from category_classifier.utils import RESOURCES_DIR


DATA_PATH = RESOURCES_DIR / "data" / "ci_data.csv"
MODEL_PATH = RESOURCES_DIR / "models" / "transaction_classifier.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TYPE = "catboost"  # варианты: "simple", "advanced", "ensemble", "catboost"
