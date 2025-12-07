"""Скрипт обучения классификатора транзакций."""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


from category_classifier import config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    sys.modules["category_classifier.train"] = sys.modules[__name__]

REQUIRED_COLUMNS = ("Date", "Withdrawal", "Deposit", "Balance", "Category")
RAW_FEATURE_COLUMNS = ("Date", "Withdrawal", "Deposit", "Balance")
FEATURE_COLUMNS = [
    "Withdrawal",
    "Deposit",
    "Balance",
    "day",
    "month",
    "weekday",
    "is_weekend",
    "transaction_amount",
    "is_withdrawal",
    "is_deposit",
    "amount_low",
    "amount_medium",
    "amount_high",
    "balance_after_low",
    "shopping_pattern",
    "month_shopping_season",
]
DATE_FORMATS = (
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%d/%m/%y",
)
_MODULE_NAME = "category_classifier.train"


def _feature_names_out(transformer, feature_names_in):
    """Имена выходных фичей для FunctionTransformer."""
    return list(FEATURE_COLUMNS)


def _extract_feature_importance(classifier) -> list[dict[str, float]]:
    """Возвращает топ-10 фич по важности, если модель поддерживает вычисление."""
    importances: np.ndarray | None = None

    try:
        if hasattr(classifier, "get_feature_importance"):
            importances = np.asarray(
                classifier.get_feature_importance(type="FeatureImportance")
            )
        elif hasattr(classifier, "feature_importances_"):
            importances = np.asarray(classifier.feature_importances_)
        elif hasattr(classifier, "coef_"):
            importances = np.asarray(classifier.coef_)
            if importances.ndim > 1:
                importances = np.mean(np.abs(importances), axis=0)
            else:
                importances = np.abs(importances)
    except Exception:
        importances = None

    if importances is None or importances.size != len(FEATURE_COLUMNS):
        return []

    importance_pairs = sorted(
        zip(FEATURE_COLUMNS, importances.tolist()),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    top_pairs = importance_pairs[:10]
    return [
        {"feature": feature, "importance": float(value)} for feature, value in top_pairs
    ]


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


def _parse_date_series(
    series: pd.Series, column_name: str
) -> tuple[pd.Series, pd.Series]:
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
        raise ValueError(
            f"В датасете отсутствуют обязательные столбцы: {missing_columns}"
        )
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
    """Строит фичи из исходного DataFrame с бизнес-логикой."""
    df = frame.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Колонка 'Date' должна быть datetime64")

    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    withdrawal = df["Withdrawal"].fillna(0.0).astype(float)
    deposit = df["Deposit"].fillna(0.0).astype(float)
    balance = df["Balance"].fillna(0.0).astype(float)

    df["transaction_amount"] = (withdrawal - deposit).abs()
    df["is_withdrawal"] = (withdrawal > 0).astype(int)
    df["is_deposit"] = (deposit > 0).astype(int)

    df["amount_low"] = (df["transaction_amount"] < 100).astype(int)
    df["amount_medium"] = (
        (df["transaction_amount"] >= 100) & (df["transaction_amount"] < 1000)
    ).astype(int)
    df["amount_high"] = (df["transaction_amount"] >= 1000).astype(int)

    df["balance_after_low"] = (balance < 100).astype(int)

    df["shopping_pattern"] = (
        (df["transaction_amount"] < 500)
        & (df["is_withdrawal"] == 1)
        & (df["weekday"] < 5)
        & ~(df["transaction_amount"] > 1000)
    ).astype(int)

    df["month_shopping_season"] = df["month"].isin([11, 12, 1]).astype(int)

    return df[FEATURE_COLUMNS].fillna(0.0)


# Обеспечиваем корректное имя модуля для сериализации пайплайна.
_feature_builder.__module__ = _MODULE_NAME
_feature_names_out.__module__ = _MODULE_NAME


def _build_model_pipeline(model_type: str = "advanced") -> Pipeline:
    feature_transformer = FunctionTransformer(
        _feature_builder,
        validate=False,
        feature_names_out=_feature_names_out,
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
    if model_type == "simple":
        classifier = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
    elif model_type == "advanced":
        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=8,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_type == "ensemble":
        classifier = VotingClassifier(
            estimators=[
                (
                    "simple",
                    LogisticRegression(
                        solver="lbfgs",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
                (
                    "advanced",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        max_depth=8,
                        min_samples_leaf=2,
                        random_state=config.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ],
            voting="soft",
            weights=[0.6, 0.4],
            n_jobs=-1,
        )
    elif model_type == "catboost":
        classifier = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            depth=6,
            learning_rate=0.1,
            iterations=400,
            l2_leaf_reg=3.0,
            random_state=config.RANDOM_STATE,
            verbose=False,
            auto_class_weights="Balanced",
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
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
    output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Загрузка данных из %s", data_path)
    df = _load_dataset(data_path)

    X = df[list(RAW_FEATURE_COLUMNS)].copy()
    y = df["Category"].astype(str)
    logger.info("Обнаружено %d уникальных категорий.", y.nunique())

    logger.info("Запуск 5-fold StratifiedKFold по Macro F1...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    base_model = _build_model_pipeline(config.MODEL_TYPE)
    cv_scores = cross_val_score(
        base_model,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    logger.info(
        "5-fold CV Macro F1: %.3f ± %.3f",
        cv_scores.mean(),
        cv_scores.std(),
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    df_train = df.loc[X_train.index].copy()
    df_test = df.loc[X_test.index].copy()
    train_split_path = output_dir / "train_split.csv"
    test_split_path = output_dir / "test_split.csv"
    df_train.to_csv(train_split_path, index=False)
    df_test.to_csv(test_split_path, index=False)
    logger.info(
        "Сплиты сохранены: train=%s, test=%s", train_split_path, test_split_path
    )

    logger.info("Обучение модели в режиме: %s", config.MODEL_TYPE)
    model = _build_model_pipeline(config.MODEL_TYPE)
    try:
        model.fit(X_train, y_train)
    except Exception as exc:  # pragma: no cover - обучение должно проходить успешно
        logger.exception("Ошибка обучения модели: %s", exc)
        raise

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    logger.info("=== Classification report ===\n%s", report)

    labels_sorted = sorted(y_test.unique())
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels_sorted,
        zero_division=0,
    )
    class_metrics = pd.DataFrame(
        {
            "Class": labels_sorted,
            "Precision": precision,
            "Recall": recall,
            "F1": f1_per_class,
            "Support": support,
        }
    )
    logger.info("=== Per-class metrics ===")
    logger.info("\n%s", class_metrics.to_string(index=False))
    per_class_metrics = [
        {
            "category": str(row["Class"]),
            "precision": float(row["Precision"]),
            "recall": float(row["Recall"]),
            "f1": float(row["F1"]),
            "support": int(row["Support"]),
        }
        for _, row in class_metrics.iterrows()
    ]

    logger.info("=== Balanced Accuracy (равный вес классам) ===")
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    logger.info("Balanced Accuracy: %.3f", balanced_acc)

    logger.info("=== Macro F1 (средний F1 по классам) ===")
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    logger.info("Macro F1: %.3f", macro_f1)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision_weighted = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    logger.info(
        "Accuracy=%.3f, Weighted F1=%.3f, Precision(weighted)=%.3f, Recall(weighted)=%.3f",
        accuracy,
        weighted_f1,
        precision_weighted,
        recall_weighted,
    )

    if config.MODEL_TYPE in {"ensemble", "catboost"}:
        baseline_map = {
            "simple": "Simple (LogReg)",
            "advanced": "Advanced (RF)",
        }
        baseline_scores: dict[str, float] = {}
        for base_type, label in baseline_map.items():
            baseline_model = _build_model_pipeline(base_type)
            baseline_model.fit(X_train, y_train)
            base_pred = baseline_model.predict(X_test)
            score = f1_score(y_test, base_pred, average="macro", zero_division=0)
            baseline_scores[label] = score
            logger.info("%s Macro F1: %.3f", label, score)

        reference_log = (
            "CatBoost" if config.MODEL_TYPE == "catboost" else "Ensemble (Voting)"
        )
        logger.info(
            "%s Macro F1 vs baselines: %.3f vs LogReg %.3f vs RF %.3f",
            reference_log,
            macro_f1,
            baseline_scores[baseline_map["simple"]],
            baseline_scores[baseline_map["advanced"]],
        )

    logger.info("=== Confusion Matrix (порядок классов как в y_test.unique()) ===")
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    logger.info("Labels order: %s", labels_sorted)
    logger.info("Confusion matrix:\n%s", cm)
    confusion_info = {
        "labels": [str(label) for label in labels_sorted],
        "matrix": cm.astype(int).tolist(),
    }

    logger.info("=== Calibrated model + custom thresholds (offline analysis) ===")
    try:
        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)

        proba = calibrated.predict_proba(X_test)
        classes = list(calibrated.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        custom_thresholds = {
            "Shopping": 0.7,
            "Rent": 0.3,
            "Transport": 0.3,
            "Salary": 0.4,
        }

        y_pred_adjusted: list[str] = []
        for row_probs in proba:
            best_idx = int(np.argmax(row_probs))
            best_class = classes[best_idx]
            best_score = 0.0
            for cls, idx in class_to_idx.items():
                threshold = custom_thresholds.get(cls, 0.5)
                score = row_probs[idx]
                if score >= threshold and score > best_score:
                    best_class = cls
                    best_score = score
            y_pred_adjusted.append(best_class)

        adjusted_macro_f1 = f1_score(
            y_test,
            y_pred_adjusted,
            average="macro",
            zero_division=0,
        )
        adjusted_balanced_acc = balanced_accuracy_score(y_test, y_pred_adjusted)
        logger.info(
            "Adjusted Macro F1 (custom thresholds): %.3f",
            adjusted_macro_f1,
        )
        logger.info(
            "Adjusted Balanced Accuracy (custom thresholds): %.3f",
            adjusted_balanced_acc,
        )
    except Exception as exc:  # pragma: no cover - вспомогательный анализ
        logger.warning("Не удалось выполнить калибровку/threshold tuning: %s", exc)

    logger.info("=== Class distribution (train) ===\n%s", y_train.value_counts())
    logger.info("=== Class distribution (test) ===\n%s", y_test.value_counts())

    scaler = (
        model.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )
    payload = {
        "model": model,
        "feature_cols": list(FEATURE_COLUMNS),
        "input_cols": list(RAW_FEATURE_COLUMNS),
        "scaler": scaler,
        "classifier": model.named_steps["clf"],
    }
    joblib.dump(payload, model_path)
    logger.info("Модель сохранена в %s", model_path)

    feature_importance = _extract_feature_importance(model.named_steps["clf"])
    metrics_summary = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "balanced_accuracy": float(balanced_acc),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "cv_macro_f1_mean": float(cv_scores.mean()),
        "cv_macro_f1_std": float(cv_scores.std()),
        "classification_report": report,
    }
    model_report = {
        "model": {
            "type": config.MODEL_TYPE,
            "version": model_path.name,
            "random_state": config.RANDOM_STATE,
        },
        "metrics": metrics_summary,
        "per_class": per_class_metrics,
        "confusion_matrix": confusion_info,
        "feature_importance": feature_importance,
    }
    report_path = output_dir / "transaction_classifier_report.json"
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(model_report, report_file, ensure_ascii=False, indent=2)
    logger.info("Отчёт о модели сохранён в %s", report_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение классификатора транзакций.")
    parser.add_argument(
        "--data", type=str, help="Путь к CSV с транзакциями.", default=None
    )
    parser.add_argument(
        "--model", type=str, help="Путь для сохранения модели.", default=None
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    data_arg = Path(args.data) if args.data else None
    model_arg = Path(args.model) if args.model else None
    train_model(data_path=data_arg, model_path=model_arg)


if __name__ == "__main__":
    main()
