"""Pydantic-схемы ответов API для ML."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class APIModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class CategoryEnum(str, Enum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"


class HealthResponse(APIModel):
    status: str = "ok"
    model_loaded: bool
    model_type: str
    model_path: str


class CategorySummary(APIModel):
    category: CategoryEnum
    count: int
    amount: float


class TimeseriesEntry(APIModel):
    period: str
    total_amount: float
    by_category: Dict[str, float]


class SummaryBlock(APIModel):
    by_category: List[CategorySummary]
    timeseries: List[TimeseriesEntry]


class RowResult(APIModel):
    index: int
    date: datetime
    withdrawal: float
    deposit: float
    balance: float
    predicted_category: CategoryEnum
    probabilities: Dict[str, float]
    actual_category: Optional[CategoryEnum] = None


class MetricsBlock(APIModel):
    has_ground_truth: bool
    macro_f1: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    accuracy: Optional[float] = None


class MetaInfo(APIModel):
    total_rows: int
    processed_rows: int
    failed_rows: int
    model_type: str
    model_version: str


class CSVUploadResponse(APIModel):
    meta: MetaInfo
    summary: SummaryBlock
    rows: List[RowResult]
    metrics: MetricsBlock


class ModelMetadata(APIModel):
    type: str
    version: str
    random_state: int


class ModelMetrics(APIModel):
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    precision_weighted: float
    recall_weighted: float
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    classification_report: str


class PerClassMetric(APIModel):
    category: str
    precision: float
    recall: float
    f1: float
    support: int


class ConfusionMatrixBlock(APIModel):
    labels: List[str]
    matrix: List[List[int]]


class FeatureImportanceItem(APIModel):
    feature: str
    importance: float


class ModelReport(APIModel):
    model: ModelMetadata
    metrics: ModelMetrics
    per_class: List[PerClassMetric]
    confusion_matrix: ConfusionMatrixBlock
    feature_importance: List[FeatureImportanceItem]
