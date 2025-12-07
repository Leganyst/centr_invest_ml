# Category Classifier

ML-модуль для классификации банковских транзакций по категориям. Использует CatBoost с feature engineering для достижения высокой точности на несбалансированных данных.

## Содержание

- [Описание](#описание)
- [Архитектура](#архитектура)
- [Установка](#установка)
- [Использование](#использование)
- [Обучение модели](#обучение-модели)
- [API](#api)
- [Конфигурация](#конфигурация)

## Описание

### Задача

Автоматическая категоризация банковских транзакций на основе:
- Даты операции
- Суммы списания (Withdrawal)
- Суммы пополнения (Deposit)
- Баланса после операции

### Категории

| Категория | Описание |
|-----------|----------|
| `Food` | Продукты питания, рестораны |
| `Misc` | Разное |
| `Rent` | Аренда, коммунальные платежи |
| `Salary` | Зарплата, доход |
| `Shopping` | Покупки |
| `Transport` | Транспорт |

### Метрики

Модель оптимизирована по **Macro F1** для корректной работы с несбалансированными классами:

- **Accuracy** — общая точность
- **Macro F1** — среднее F1 по классам (равный вес)
- **Balanced Accuracy** — точность с учётом дисбаланса классов
- **Per-class Precision/Recall/F1** — метрики по каждой категории

## Архитектура

### Pipeline

```
Raw Transaction
      ↓
Feature Engineering (_feature_builder)
      ↓
StandardScaler (нормализация)
      ↓
CatBoost Classifier
      ↓
Category + Probabilities
```

### Feature Engineering

Из сырых данных (Date, Withdrawal, Deposit, Balance) строятся 16 признаков:

| Группа | Признаки |
|--------|----------|
| **Исходные** | Withdrawal, Deposit, Balance |
| **Календарные** | day, month, weekday, is_weekend |
| **Суммы** | transaction_amount, is_withdrawal, is_deposit |
| **Бинирование** | amount_low (<100), amount_medium (100-1000), amount_high (>1000) |
| **Баланс** | balance_after_low (<100) |
| **Паттерны** | shopping_pattern, month_shopping_season (ноябрь-январь) |

### Модели

Поддерживаются 4 типа моделей:

| Тип | Описание |
|-----|----------|
| `simple` | Logistic Regression (baseline) |
| `advanced` | Random Forest (200 деревьев, max_depth=8) |
| `ensemble` | Voting (LogReg + RF, soft voting) |
| `catboost` | CatBoost (400 итераций, auto_class_weights) — **по умолчанию** |

## Установка

```bash
# Как часть основного проекта (workspace package)
uv sync

# Или отдельно
cd third-party/category-classifier
pip install -e .
```

### Зависимости

- Python 3.13+
- catboost >= 1.2.8
- scikit-learn >= 1.7.2
- pandas >= 2.3.3
- joblib == 1.5.2

## Использование

### Инференс

```python
from category_classifier import CategoryClassifierService
from category_classifier.schemas import Transaction

# Создание сервиса (загружает модель из resources/models/)
service = CategoryClassifierService()

# Предсказание категории
tx = Transaction(
    date="2023-01-15",
    withdrawal=150.0,
    deposit=0.0,
    balance=2500.0,
)

result = service.predict(tx)
print(f"Категория: {result.category}")
print(f"Вероятности: {result.proba}")
```

### Прямой доступ к классификатору

```python
from category_classifier.model import TransactionClassifier
from category_classifier.schemas import Transaction

classifier = TransactionClassifier()

tx = Transaction(
    date="2023-01-15",
    withdrawal=100.0,
    deposit=0.0,
    balance=1500.0,
)

# Только категория
category = classifier.predict_category(tx)

# Вероятности по всем категориям
probabilities = classifier.predict_proba(tx)
```

## Обучение модели

### Формат данных

CSV с колонками:

| Колонка | Тип | Описание |
|---------|-----|----------|
| Date | string | Дата транзакции (MM/DD/YYYY или DD/MM/YYYY) |
| Withdrawal | float | Сумма списания |
| Deposit | float | Сумма пополнения |
| Balance | float | Баланс после операции |
| Category | string | Категория (target) |

### Запуск обучения

```bash
# С параметрами по умолчанию
python -m category_classifier.train

# С указанием путей
python -m category_classifier.train \
  --data ./resources/data/ci_data.csv \
  --model ./resources/models/transaction_classifier.joblib
```

### Артефакты

После обучения создаются:

| Файл | Описание |
|------|----------|
| `transaction_classifier.joblib` | Сериализованный pipeline |
| `transaction_classifier_report.json` | Метрики, confusion matrix, feature importance |
| `train_split.csv` | Тренировочная выборка |
| `test_split.csv` | Тестовая выборка |

### Структура артефакта модели

```python
{
    "model": Pipeline,           # sklearn Pipeline (features → preprocess → clf)
    "feature_cols": [...],       # Список имён фичей
    "input_cols": [...],         # Входные колонки (Date, Withdrawal, ...)
    "scaler": StandardScaler,    # Обученный scaler
    "classifier": CatBoost,      # Обученный классификатор
}
```

### Структура JSON-отчёта

```json
{
  "model": {
    "type": "catboost",
    "version": "transaction_classifier.joblib",
    "random_state": 42
  },
  "metrics": {
    "accuracy": 0.85,
    "macro_f1": 0.78,
    "weighted_f1": 0.84,
    "balanced_accuracy": 0.76,
    "cv_macro_f1_mean": 0.77,
    "cv_macro_f1_std": 0.02
  },
  "per_class": [
    {"category": "Food", "precision": 0.82, "recall": 0.79, "f1": 0.80, "support": 150},
    ...
  ],
  "confusion_matrix": {
    "labels": ["Food", "Misc", "Rent", "Salary", "Shopping", "Transport"],
    "matrix": [[...], ...]
  },
  "feature_importance": [
    {"feature": "transaction_amount", "importance": 0.25},
    ...
  ]
}
```

## API

### Схемы

```python
from category_classifier.schemas import Category, Transaction, PredictionResponse

class Category(StrEnum):
    FOOD = "Food"
    MISC = "Misc"
    RENT = "Rent"
    SALARY = "Salary"
    SHOPPING = "Shopping"
    TRANSPORT = "Transport"

@dataclass
class Transaction:
    date: str           # ISO date string
    withdrawal: float
    deposit: float
    balance: float

@dataclass
class PredictionResponse:
    category: Category
    proba: dict[Category, float]
```

### CategoryClassifierService

```python
class CategoryClassifierService:
    def __init__(self, model_path: Path | str | None = None):
        """
        Args:
            model_path: Путь к .joblib файлу. 
                       Если None — используется MODEL_PATH из config.py.
        """
        ...

    def predict(self, tx: Transaction) -> PredictionResponse:
        """Предсказывает категорию и возвращает вероятности."""
        ...
```

### TransactionClassifier

```python
class TransactionClassifier:
    def __init__(self, model_path: Path | str | None = None):
        ...

    def predict_category(self, tx: Transaction) -> Category:
        """Возвращает наиболее вероятную категорию."""
        ...

    def predict_proba(self, tx: Transaction) -> dict[Category, float]:
        """Возвращает вероятности по всем категориям."""
        ...
```

## Конфигурация

Файл `config.py`:

```python
from category_classifier.utils import RESOURCES_DIR

DATA_PATH = RESOURCES_DIR / "data" / "ci_data.csv"
MODEL_PATH = RESOURCES_DIR / "models" / "transaction_classifier.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TYPE = "catboost"  # "simple" | "advanced" | "ensemble" | "catboost"
```

## Интеграция с основным приложением

Модуль интегрируется через провайдер в `app/services/providers/category_classifier.py`:

```python
from category_classifier import CategoryClassifierService
from category_classifier.schemas import Transaction, Category

class MlCategoryClassifier(ICategoryClassifier):
    def __init__(self):
        self.classifier = CategoryClassifierService()
    
    def predict(self, transaction: TransactionSchema) -> PredictionResult:
        tx = Transaction(
            balance=transaction.balance,
            deposit=transaction.deposit,
            withdrawal=transaction.withdrawal,
            date=transaction.date.isoformat(),
        )
        response = self.classifier.predict(tx)
        # Маппинг Category → TransactionCategory (enum из основного приложения)
        ...
```

## Структура директории

```
category-classifier/
├── category_classifier/
│   ├── __init__.py          # CategoryClassifierService
│   ├── config.py            # Конфигурация путей
│   ├── model.py             # TransactionClassifier
│   ├── schemas.py           # Category, Transaction, PredictionResponse
│   ├── train.py             # Скрипт обучения
│   └── utils.py             # Утилиты (RESOURCES_DIR)
├── resources/
│   ├── data/
│   │   └── ci_data.csv      # Тренировочные данные
│   └── models/
│       ├── transaction_classifier.joblib
│       └── transaction_classifier_report.json
├── pyproject.toml
└── README.md
```

## Лицензия

MIT

