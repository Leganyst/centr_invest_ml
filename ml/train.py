"""Скрипт обучения классификатора транзакций."""
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


from . import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("Date", "Withdrawal", "Deposit", "Balance", "Category")
RAW_FEATURE_COLUMNS = ("Date", "Withdrawal", "Deposit", "Balance")
FEATURE_COLUMNS = ("Withdrawal", "Deposit", "Balance", "day", "month", "weekday")
DATE_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%d/%m/%y",
)


def _resolve_path(
    user_path: Path | str | None,
    default_path: Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Преобразует пользовательский путь в абсолютный и валидирует существование."""
    if user_path is None:
        path = default_path
    else:
        path = Path(user_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if must_exist and not path.exists():
        raise ValueError(f"Указанного пути не существует: {path}")
    return path


def _clean_date_series(series: pd.Series) -> pd.Series:
    """Удаляет лишние пробелы, BOM и заменяет пустые строки на NA."""
    return (
        series.astype("string")
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
        .replace({"": pd.NA})
    )


def _parse_date_series(series: pd.Series, column_name: str) -> tuple[pd.Series, pd.Series]:
    """Пытается распарсить колонку дат, используя несколько форматов."""
    cleaned = _clean_date_series(series)
    parsed = pd.Series(pd.NaT, index=series.index)
    format_hits: dict[str, int] = {}

    for fmt in DATE_FORMATS:
        mask = parsed.isna()
        if not mask.any():
            break
        attempt = pd.to_datetime(cleaned[mask], format=fmt, errors="coerce")
        hits = int(attempt.notna().sum())
        if hits:
            parsed.loc[mask] = attempt
            format_hits[fmt] = hits

    mask = parsed.isna()
    if mask.any():
        fallback = pd.to_datetime(cleaned[mask], errors="coerce", dayfirst=True)
        hits = int(fallback.notna().sum())
        if hits:
            parsed.loc[mask] = fallback
            format_hits["dayfirst_parse"] = hits

    logger.info(
        "Колонка %s: распознано %d дат (статистика по форматам: %s)",
        column_name,
        int(parsed.notna().sum()),
        format_hits or "не удалось определить форматы",
    )

    invalid_mask = parsed.isna()
    if invalid_mask.any():
        examples = cleaned[invalid_mask].dropna().unique().tolist()[:5]
        logger.warning(
            "Колонка %s: %d дат не распарсились. Примеры: %s",
            column_name,
            int(invalid_mask.sum()),
            examples,
        )

    return parsed, cleaned


def _load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype_backend="numpy_nullable")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют обязательные столбцы: {missing_columns}")
    logger.info("Загружено %d строк.", len(df))

    # Парсим обе колонки дат: основную и дублированную
    date_main, raw_main = _parse_date_series(df["Date"], "Date")
    if "Date.1" in df.columns:
        date_alt, raw_alt = _parse_date_series(df["Date.1"], "Date.1")
    else:
        date_alt = pd.Series(pd.NaT, index=df.index)
        raw_alt = pd.Series(pd.NA, index=df.index, dtype="string")

    df["Date"] = date_main.fillna(date_alt)

    invalid_mask = df["Date"].isna()
    if invalid_mask.any():
        invalid_indices = df[invalid_mask].index
        samples = []
        for idx in invalid_indices[:5]:
            samples.append(
                {
                    "Date": raw_main.iloc[idx],
                    "Date.1": raw_alt.iloc[idx],
                }
            )
        logger.warning(
            "После объединения Date и Date.1 осталось %d некорректных дат. Примеры: %s",
            int(invalid_mask.sum()),
            samples,
        )
        df = df.dropna(subset=["Date"])
        logger.info("После удаления строк без даты: %d строк.", len(df))

    for column in ("Withdrawal", "Deposit", "Balance"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    initial_rows = len(df)
    df = df.dropna(how="all")
    removed_all = initial_rows - len(df)
    if removed_all:
        logger.info("Удалено %d полностью пустых строк.", removed_all)

    before_category = len(df)
    df = df.dropna(subset=["Category"])
    removed_category = before_category - len(df)
    if removed_category:
        logger.info("Удалено %d строк без категории.", removed_category)

    logger.info("Датасет после очистки: %d строк.", len(df))
    return df.reset_index(drop=True)



def _feature_builder(frame: pd.DataFrame) -> pd.DataFrame:
    """Строит фичи из исходного DataFrame."""
    if not pd.api.types.is_datetime64_any_dtype(frame["Date"]):
        raise ValueError("Колонка 'Date' должна быть datetime64")

    date_features = pd.DataFrame(
        {
            "day": frame["Date"].dt.day,      # type: ignore
            "month": frame["Date"].dt.month,  # type: ignore
            "weekday": frame["Date"].dt.weekday,  # type: ignore
        },
        index=frame.index,
    )

    numeric_features = frame[["Withdrawal", "Deposit", "Balance"]]
    features = pd.concat([numeric_features, date_features], axis=1)

    # ВАЖНО: привести FEATURE_COLUMNS к list
    return features[list(FEATURE_COLUMNS)]


def _feature_names_out(transformer: FunctionTransformer, input_features: list[str] | None) -> list[str]:
    """Преобразует входные имена фичей в выходные."""
    if input_features is None:
        names = list(RAW_FEATURE_COLUMNS)
    else:
        names = list(input_features)

    numeric_features = names[1:]  # Пропускаем 'Date'
    date_features = ["day", "month", "weekday"]
    return numeric_features + date_features

def _build_model_pipeline() -> Pipeline:
    feature_transformer = FunctionTransformer(
        _feature_builder,
        validate=False,
        feature_names_out=_feature_names_out,  # Заменяем "one-to-one"
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, list(FEATURE_COLUMNS))],
        remainder="drop",
    )
    classifier = LogisticRegression(solver="lbfgs", multi_class="auto")
    return Pipeline(
        steps=[
            ("features", feature_transformer),
            ("preprocess", preprocessor),
            ("clf", classifier),
        ]
    )



def train_model(
    data_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> None:
    """Обучает классификатор транзакций и сохраняет артефакт на диск."""
    data_path = _resolve_path(data_path, config.DATA_PATH, must_exist=True)
    model_path = _resolve_path(model_path, config.MODEL_PATH)

    logger.info("Загрузка данных из %s", data_path)
    df = _load_dataset(data_path)

    X = df[list(RAW_FEATURE_COLUMNS)].copy()
    y = df["Category"].astype(str)
    logger.info("Обнаружено %d уникальных категорий.", y.nunique())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    model = _build_model_pipeline()
    try:
        model.fit(X_train, y_train)
    except Exception as exc:  # pragma: no cover - обучение должно проходить успешно
        logger.exception("Ошибка обучения модели: %s", exc)
        raise

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("=== Classification report ===\n%s", report)
    logger.info("=== Class distribution (train) ===\n%s", y_train.value_counts())
    logger.info("=== Class distribution (test) ===\n%s", y_test.value_counts())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    payload = {
        "model": model,
        "feature_cols": list(FEATURE_COLUMNS),
        "input_cols": list(RAW_FEATURE_COLUMNS),
        "scaler": scaler,
        "classifier": model.named_steps["clf"],
    }
    joblib.dump(payload, model_path)
    logger.info("Модель сохранена в %s", model_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение классификатора транзакций.")
    parser.add_argument("--data", type=str, help="Путь к CSV с транзакциями.", default=None)
    parser.add_argument("--model", type=str, help="Путь для сохранения модели.", default=None)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    data_arg = Path(args.data) if args.data else None
    model_arg = Path(args.model) if args.model else None
    train_model(data_path=data_arg, model_path=model_arg)


if __name__ == "__main__":
    main()
