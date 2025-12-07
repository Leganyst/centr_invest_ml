import asyncio
import csv
import logging
from collections import defaultdict
from datetime import datetime
from io import StringIO
from uuid import UUID

from dishka import Provider, Scope, provide
from fastapi import UploadFile
from pydantic import TypeAdapter
from sqlalchemy import case, select, update, cast
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps.auth import CurrentUser
from app.models.enums import TransactionCategory
from app.models.transaction import Transaction
from app.schemas import ml as ml_schemas
from app.schemas.notifications import NotificationSchema
from app.schemas.transactions import TransactionCreateSchema, TransactionSchema
from app.services.filters import FilterType, PaginatedResponse, PaginatedSchema
from app.services.providers.protocols.category_classifier import ICategoryClassifier
from app.services.providers.protocols.notification_manager import INotificationManager
from app.settings.ml import ModelSettings

logger = logging.getLogger(__name__)


class TransactionRetrieveInteractor:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def all(
        self, filters: FilterType | None = None, page: PaginatedSchema | None = None
    ) -> PaginatedResponse[Transaction]:
        query = select(Transaction)
        if filters is not None:
            query = query.where(filters)
        return await PaginatedResponse.of(self.session, query, page=page)


class TransactionBulkCreateInteractor:
    def __init__(self, session: AsyncSession, current_user: CurrentUser):
        self.session = session
        self.current_user = current_user

    def _map(self, transaction: TransactionCreateSchema) -> Transaction:
        return Transaction(
            date=transaction.date,
            withdrawal=transaction.withdrawal,
            deposit=transaction.deposit,
            balance=transaction.balance,
            category=transaction.category,
            user_id=self.current_user.id,
        )

    async def create(
        self, transactions: list[TransactionCreateSchema]
    ) -> list[Transaction]:
        transactions_db = [self._map(transaction) for transaction in transactions]
        self.session.add_all(transactions_db)
        return transactions_db


class TransactionImporter:
    def __init__(self, create_interactor: TransactionBulkCreateInteractor):
        self.create_interactor = create_interactor

    async def __call__(self, file: UploadFile) -> list[Transaction]:
        logger.info("Importing transactions from %s", file.filename)
        content = await file.read()
        reader = csv.DictReader(StringIO(content.decode()), delimiter=",")
        transactions = TypeAdapter(list[TransactionCreateSchema]).validate_python(
            reader
        )
        return await self.create_interactor.create(transactions)


class TransactionBackgroundClassifier:
    def __init__(
        self,
        session: AsyncSession,
        retriever: TransactionRetrieveInteractor,
        classifier: ICategoryClassifier,
        notifications: INotificationManager,
    ):
        self.session = session
        self.retriever = retriever
        self.classifier = classifier
        self.notifications = notifications

    async def __call__(self):
        transactions = await self.retriever.all(
            Transaction.category.is_(None), PaginatedSchema(limit=50)
        )
        transactions_for_update = transactions.items
        if not transactions_for_update:
            return
        logger.info("Found %s transactions", len(transactions_for_update))
        updates: dict[UUID, TransactionCategory] = {}
        users_for_notifications: set[UUID] = set()
        for transaction in transactions_for_update:
            prediction = self.classifier.predict(
                TransactionSchema.model_validate(transaction)
            )
            updates[transaction.id] = prediction.category
            users_for_notifications.add(transaction.user_id)
        if not updates:
            return
        await self.session.execute(
            update(Transaction)
            .where(Transaction.id.in_(updates.keys()))
            .values(
                category=case(
                    *[
                        (
                            Transaction.id == key,
                            cast(value.name, Transaction.category.type),
                        )
                        for key, value in updates.items()
                    ]
                )
            )
        )
        if users_for_notifications:
            await asyncio.gather(
                *[
                    self.notifications.send(
                        user,
                        NotificationSchema(
                            user_id=user,
                            text="Была выполнена классификация ваших транзакций",
                            type="transaction-classifier",
                        ),
                    )
                    for user in users_for_notifications
                ]
            )


class TransactionAnalyticsInteractor:
    def __init__(
        self,
        session: AsyncSession,
        classifier: ICategoryClassifier,
        model_settings: ModelSettings,
    ):
        self.session = session
        self.classifier = classifier
        self.model_settings = model_settings

    async def for_user(self, user_id: UUID) -> ml_schemas.CSVUploadResponse:
        transactions = list(
            await self.session.scalars(
                select(Transaction)
                .where(Transaction.user_id == user_id)
                .order_by(Transaction.date, Transaction.created_at)
            )
        )
        total_rows = len(transactions)
        rows: list[ml_schemas.RowResult] = []

        summary_by_category: dict[TransactionCategory, dict[str, float]] = defaultdict(
            lambda: {"count": 0, "amount": 0.0}
        )
        timeseries_amounts: dict[str, dict[TransactionCategory, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        timeseries_totals: dict[str, float] = defaultdict(float)

        for transaction in transactions:
            try:
                tx_schema = TransactionSchema.model_validate(transaction)
                prediction = self.classifier.predict(tx_schema)
                predicted_category = prediction.category
                probabilities = {
                    category.value: float(probability)
                    for category, probability in prediction.probabilities.items()
                }
                category_value = transaction.category or predicted_category
                if transaction.category is None:
                    transaction.category = predicted_category
                category_enum = ml_schemas.CategoryEnum(category_value.value)
            except Exception:
                logger.exception(
                    "Failed to classify transaction %s for analytics",
                    getattr(transaction, "id", None),
                )
                continue

            amount = (
                transaction.withdrawal
                if transaction.withdrawal > 0
                else transaction.deposit
            )
            amount_value = float(amount or 0.0)

            summary_entry = summary_by_category[category_value]
            summary_entry["count"] += 1
            summary_entry["amount"] += amount_value

            period = transaction.date.strftime("%Y-%m")
            timeseries_totals[period] += amount_value
            timeseries_amounts[period][category_value] += amount_value

            row_index = len(rows)
            rows.append(
                ml_schemas.RowResult(
                    index=row_index,
                    date=datetime.combine(transaction.date, datetime.min.time()),
                    withdrawal=float(transaction.withdrawal),
                    deposit=float(transaction.deposit),
                    balance=float(transaction.balance),
                    predicted_category=category_enum,
                    probabilities=probabilities,
                    actual_category=None,
                )
            )

        summary = [
            ml_schemas.CategorySummary(
                category=ml_schemas.CategoryEnum(category.value),
                count=values["count"],
                amount=values["amount"],
            )
            for category, values in summary_by_category.items()
        ]

        timeseries_entries = [
            ml_schemas.TimeseriesEntry(
                period=period,
                total_amount=timeseries_totals[period],
                by_category={
                    category.value: amount for category, amount in by_category.items()
                },
            )
            for period, by_category in sorted(timeseries_amounts.items())
        ]

        meta = ml_schemas.MetaInfo(
            total_rows=total_rows,
            processed_rows=len(rows),
            failed_rows=total_rows - len(rows),
            model_type=self.model_settings.model_type,
            model_version=self.model_settings.model_path.name,
        )
        metrics = ml_schemas.MetricsBlock(
            has_ground_truth=False,
            macro_f1=None,
            balanced_accuracy=None,
            accuracy=None,
        )

        return ml_schemas.CSVUploadResponse(
            meta=meta,
            summary=ml_schemas.SummaryBlock(
                by_category=summary,
                timeseries=timeseries_entries,
            ),
            rows=rows,
            metrics=metrics,
        )


class TransactionServicesProvider(Provider):
    scope = Scope.REQUEST

    retrieve = provide(TransactionRetrieveInteractor)
    importer = provide(TransactionImporter)
    background_classifier = provide(TransactionBackgroundClassifier)
    analytics = provide(TransactionAnalyticsInteractor)
    bulk_create = provide(TransactionBulkCreateInteractor)
