import logging
from io import StringIO
from typing import Iterable
from uuid import UUID

from dishka import Provider, Scope, provide
from dishka.integrations.fastapi import inject
from fastapi import UploadFile
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import select, update, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.enums import TransactionCategory
from app.schemas.transactions import TransactionCreateSchema, TransactionSchema
from app.services.filters import FilterType, PaginatedSchema, apply_pagination
from app.models.transaction import Transaction
from app.deps.auth import CurrentUser
import csv

from app.services.providers.protocols.category_classifier import ICategoryClassifier

logger = logging.getLogger(__name__)


class TransactionRetrieveInteractor:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def all(self,
            filters: FilterType | None = None,
            page: PaginatedSchema | None = None) -> Iterable[Transaction]:
        query = select(Transaction)
        if filters:
            query = query.where(filters)
        if page:
            query = apply_pagination(
                query,
                page,
                default_ordering=[
                    Transaction.date.desc(),
                    Transaction.created_at.desc(),
                ]
            )
        return await self.session.scalars(query)


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
            user_id=self.current_user.id,
        )

    async def create(self, transactions: list[TransactionCreateSchema]) -> list[Transaction]:
        transactions_db = [self._map(transaction) for transaction in transactions]
        self.session.add_all(transactions_db)
        return transactions_db


class TransactionImporter:
    def __init__(self, create_interactor: TransactionBulkCreateInteractor):
        self.create_interactor = create_interactor

    async def __call__(self, file: UploadFile):
        logger.info(f"Importing transactions from %s", file.filename)
        content = await file.read()
        reader = csv.DictReader(StringIO(content.decode()), delimiter=",")
        transactions = TypeAdapter(list[TransactionCreateSchema]).validate_python(reader)
        await self.create_interactor.create(transactions)


class TransactionBackgroundClassifier:
    def __init__(self,
                 session: AsyncSession,
                 retriever: TransactionRetrieveInteractor,
                 classifier: ICategoryClassifier):
        self.session = session
        self.retriever = retriever
        self.classifier = classifier

    async def __call__(self):
        transactions_for_update = list(
            await self.retriever.all(
                Transaction.category.is_(None),
                PaginatedSchema(limit=10)
            )
        )
        if not transactions_for_update:
            return
        logger.info("Found %s transactions", len(transactions_for_update))
        updates: dict[UUID, TransactionCategory] = {}
        for transaction in transactions_for_update:
            category = self.classifier.predict(TransactionSchema.model_validate(transaction))
            updates[transaction.id] = category
        await self.session.execute(
            update(Transaction)
            .values(
                category=case(
                    *[
                        (Transaction.id == key, value)
                        for key, value in updates.items()
                    ]
                )
            )
        )


class TransactionServicesProvider(Provider):
    scope = Scope.REQUEST

    retrieve = provide(TransactionRetrieveInteractor)
    importer = provide(TransactionImporter)
    background_classifier = provide(TransactionBackgroundClassifier)
    bulk_create = provide(TransactionBulkCreateInteractor)
