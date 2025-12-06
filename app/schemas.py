"""Pydantic schemas for API responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class CategoryEnum(str, Enum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_type: str
    model_path: str


class CategorySummary(BaseModel):
    category: CategoryEnum
    count: int
    amount: float


class TimeseriesEntry(BaseModel):
    period: str
    total_amount: float
    by_category: Dict[str, float]


class SummaryBlock(BaseModel):
    by_category: List[CategorySummary]
    timeseries: List[TimeseriesEntry]


class RowResult(BaseModel):
    index: int
    date: datetime
    withdrawal: float
    deposit: float
    balance: float
    predicted_category: CategoryEnum
    probabilities: Dict[str, float]
    actual_category: Optional[CategoryEnum] = None


class MetricsBlock(BaseModel):
    has_ground_truth: bool
    macro_f1: Optional[float]
    balanced_accuracy: Optional[float]


class MetaInfo(BaseModel):
    total_rows: int
    processed_rows: int
    failed_rows: int
    model_type: str
    model_version: str


class CSVUploadResponse(BaseModel):
    meta: MetaInfo
    summary: SummaryBlock
    rows: List[RowResult]
    metrics: MetricsBlock
