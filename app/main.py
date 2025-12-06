"""FastAPI application exposing inference endpoints."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from sklearn.metrics import balanced_accuracy_score, f1_score

from . import schemas
from .config import get_settings
from .ml_adapter import get_classifier_or_503, load_classifier, predict_dataframe

LOGGER = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="Transaction Classifier API", version="0.1.0")


@app.on_event("startup")
def _startup() -> None:  # pragma: no cover - framework hook
    try:
        load_classifier()
    except Exception as exc:
        LOGGER.exception("Classifier failed to load on startup: %s", exc)


@app.get("/api/health", response_model=schemas.HealthResponse)
def healthcheck() -> schemas.HealthResponse:
    model_loaded = False
    try:
        get_classifier_or_503()
        model_loaded = True
    except HTTPException:
        model_loaded = False
    return schemas.HealthResponse(
        model_loaded=model_loaded,
        model_type=settings.model_type,
        model_path=str(settings.model_path),
    )


@app.post(
    "/api/v1/transactions/upload",
    response_model=schemas.CSVUploadResponse,
    summary="Классификация транзакций из CSV",
    response_description=(
        "JSON с метаданными (`meta`), агрегатами (`summary`), "
        "списком строк (`rows`) и метриками качества (`metrics`). "
        "`summary.by_category` используется для круговых/столбчатых диаграмм, "
        "`summary.timeseries` — для графика по месяцам, а `rows` — для таблицы/ручной корректировки."
    ),
)
async def upload_transactions(file: UploadFile = File(...)) -> schemas.CSVUploadResponse:
    """
    Загрузка CSV и получение данных для фронта.

    Возвращает JSON следующего вида:
    - `meta`: количество строк в файле, сколько удалось обработать, тип/версия модели.
    - `summary.by_category`: список объектов `{category, count, amount}` — удобно для круговых/бар-чартов.
    - `summary.timeseries`: массив периодов `YYYY-MM` с суммой операций и разбиением по категориям.
    - `rows`: массив строк для табличного отображения; содержит исходные значения и `predicted_category`.
    - `metrics`: если в CSV была колонка `Category`, здесь будут `macro_f1` и `balanced_accuracy`.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="VALIDATION_ERROR: файл должен иметь расширение .csv",
        )
    raw_bytes = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"VALIDATION_ERROR: невозможно прочитать CSV ({exc})",
        )

    required_cols = {"Date", "Withdrawal", "Deposit", "Balance"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"VALIDATION_ERROR: отсутствуют колонки {missing}",
        )

    total_rows = len(df)
    df_clean = df.copy()
    df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
    for col in ["Withdrawal", "Deposit", "Balance"]:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_valid = df_clean.dropna(subset=["Date", "Withdrawal", "Deposit", "Balance"])
    processed_rows = len(df_valid)
    failed_rows = total_rows - processed_rows

    labels, probas = predict_dataframe(df_valid[["Date", "Withdrawal", "Deposit", "Balance"]])

    amount_series = df_valid["Withdrawal"].where(df_valid["Withdrawal"] > 0, df_valid["Deposit"]).fillna(0.0)
    df_enriched = df_valid.assign(predicted=labels, amount=amount_series, period=df_valid["Date"].dt.to_period("M").astype(str))

    category_summary = (
        df_enriched.groupby("predicted")
        .agg(count=("predicted", "size"), amount=("amount", "sum"))
        .reset_index()
    )
    summary_by_category = []
    for _, row in category_summary.iterrows():
        label = str(row["predicted"])
        try:
            category_enum = schemas.CategoryEnum(label)
        except ValueError:
            continue
        summary_by_category.append(
            schemas.CategorySummary(
                category=category_enum,
                count=int(row["count"]),
                amount=float(row["amount"]),
            )
        )

    timeseries_entries = []
    if not df_enriched.empty:
        pivot = (
            df_enriched.pivot_table(
                index="period",
                columns="predicted",
                values="amount",
                aggfunc="sum",
                fill_value=0.0,
            )
        )
        total_amount = pivot.sum(axis=1)
        for period, row in pivot.iterrows():
            timeseries_entries.append(
                schemas.TimeseriesEntry(
                    period=str(period),
                    total_amount=float(total_amount.loc[period]),
                    by_category={str(cat): float(row[cat]) for cat in pivot.columns},
                )
            )

    truth_series = df.loc[df_enriched.index, "Category"] if "Category" in df.columns else None
    has_truth = truth_series is not None and truth_series.notna().any()
    macro_f1 = None
    balanced_acc = None
    if has_truth:
        mask = truth_series.notna()
        y_true = truth_series[mask].astype(str)
        y_pred_truth = [labels[idx] for idx, flag in enumerate(mask.tolist()) if flag]
        if len(y_true) > 0 and len(y_pred_truth) == len(y_true):
            macro_f1 = f1_score(y_true, y_pred_truth, average="macro", zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true, y_pred_truth)

    rows = []
    actual_categories = truth_series if truth_series is not None else None
    for (idx, row), label, proba in zip(df_enriched.iterrows(), labels, probas):
        actual_value = None
        if actual_categories is not None:
            raw = actual_categories.loc[idx]
            if isinstance(raw, str):
                try:
                    actual_value = schemas.CategoryEnum(raw)
                except ValueError:
                    actual_value = None
        try:
            predicted_enum = schemas.CategoryEnum(label)
        except ValueError:
            continue
        rows.append(
            schemas.RowResult(
                index=int(idx),
                date=row["Date"],
                withdrawal=float(row["Withdrawal"]),
                deposit=float(row["Deposit"]),
                balance=float(row["Balance"]),
                predicted_category=predicted_enum,
                probabilities=proba,
                actual_category=actual_value,
            )
        )

    meta = schemas.MetaInfo(
        total_rows=total_rows,
        processed_rows=processed_rows,
        failed_rows=failed_rows,
        model_type=settings.model_type,
        model_version=Path(settings.model_path).name,
    )
    summary = schemas.SummaryBlock(
        by_category=summary_by_category,
        timeseries=timeseries_entries,
    )
    metrics = schemas.MetricsBlock(
        has_ground_truth=bool(has_truth),
        macro_f1=macro_f1,
        balanced_accuracy=balanced_acc,
    )

    return schemas.CSVUploadResponse(
        meta=meta,
        summary=summary,
        rows=rows,
        metrics=metrics,
    )
