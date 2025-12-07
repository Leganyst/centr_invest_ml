import json
import logging
from pathlib import Path

from dishka import FromDishka
from dishka.integrations.fastapi import DishkaRoute
from fastapi import APIRouter, HTTPException, status

from app.schemas import ml as ml_schemas
from app.settings.ml import ModelSettings

router = APIRouter(prefix="/ml", tags=["ml"], route_class=DishkaRoute)
logger = logging.getLogger(__name__)

REPORT_FILENAME = "transaction_classifier_report.json"


@router.get(
    "/report",
    response_model=ml_schemas.ModelReport,
    summary="Глобальные метрики обученной модели",
)
async def get_model_report(
    settings: FromDishka[ModelSettings],
) -> ml_schemas.ModelReport:
    """Возвращает json-отчёт, который сохраняется при обучении модели."""
    report_path = Path(settings.model_path).parent / REPORT_FILENAME
    if not report_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MODEL_REPORT_NOT_FOUND: обучите модель и убедитесь, что transaction_classifier_report.json лежит рядом с артефактом модели.",
        )
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        return ml_schemas.ModelReport.model_validate(payload)
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to read model report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"FAILED_TO_READ_REPORT: {exc}",
        )
