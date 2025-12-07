# Forecasting

ML-модуль для прогнозирования месячных расходов пользователя. Использует временные ряды с лаговыми признаками и регрессионные модели (Ridge, RandomForest).

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

Прогнозирование суммарных расходов (total_expense = сумма Withdrawal) на следующий месяц по истории транзакций пользователя.

### Входные данные

DataFrame с транзакциями одного пользователя:

| Колонка | Описание |
|---------|----------|
| Date | Дата транзакции |
| Withdrawal | Расход (>0 для списаний) |
| Deposit | Пополнение (>0 для приходов) |
| Balance | Баланс после операции |
| Category | Категория транзакции |

### Метрики

- **MAE** — средняя абсолютная ошибка
- **RMSE** — корень из среднеквадратичной ошибки
- **MAPE** — средняя абсолютная процентная ошибка

### Baseline

Наивный прогноз: расход следующего месяца = расход текущего месяца.

## Архитектура

### Pipeline

```
Raw Transactions
        ↓
build_monthly_features()
   - Агрегация по месяцам
   - Календарные признаки
   - Категориальные доли
        ↓
add_lag_features()
   - Лаги (1, 2, 3 месяца)
   - Скользящее среднее
   - Темп роста
        ↓
train_test_split_by_month()
   - Временной split (последние N месяцев = test)
        ↓
Ridge / RandomForest Regressor
        ↓
Predicted total_expense
```

### Месячные признаки

| Признак | Описание |
|---------|----------|
| `total_expense` | Сумма Withdrawal за месяц (target) |
| `total_deposit` | Сумма Deposit за месяц |
| `avg_balance` | Средний баланс |
| `end_balance` | Баланс в конце месяца |
| `n_transactions` | Количество транзакций |
| `avg_tx_amount` | Средний размер расходной транзакции |
| `food_ratio` | Доля транзакций категории "Food" |
| `misc_ratio` | Доля транзакций категории "Misc" |
| `is_december` | Флаг декабря |
| `quarter` | Квартал (1-4) |
| `month_sin`, `month_cos` | Сезонность (циклическое кодирование) |

### Лаговые признаки

| Признак | Описание |
|---------|----------|
| `total_expense_lag1` | Расходы предыдущего месяца |
| `total_expense_lag2` | Расходы 2 месяца назад |
| `total_expense_lag3` | Расходы 3 месяца назад |
| `total_expense_ma3` | Скользящее среднее за 3 месяца |
| `total_expense_growth` | Темп роста vs прошлый месяц |
| `end_balance_lag1` | Баланс на конец прошлого месяца |
| `baseline_prev_month` | Наивный прогноз (= lag1) |

### Модели

| Модель | Описание |
|--------|----------|
| **Ridge** | Линейная регрессия с L2-регуляризацией (alpha=1.0) |
| **RandomForest** | Ансамбль деревьев (50 деревьев, max_depth=4) |

Выбирается модель с наименьшим RMSE на тестовой выборке.

## Установка

```bash
# Как часть основного проекта (workspace package)
uv sync

# Или отдельно
cd third-party/forecasting
pip install -e .
```

### Зависимости

- Python 3.13+
- scikit-learn >= 1.7.2
- pandas >= 2.3.3
- numpy >= 2.3.5
- joblib == 1.5.2

## Использование

### Прогноз на следующий месяц

```python
from forecasting.model import ExpenseForecastModel
import pandas as pd

# Загрузка истории транзакций пользователя
df_user = pd.read_csv("user_transactions.csv")

# Создание модели (загружает артефакт из resources/models/)
model = ExpenseForecastModel()

# Прогноз суммарных расходов на следующий месяц
next_month_expense = model.predict_next_month(df_user)
print(f"Прогноз расходов: {next_month_expense:.2f}")

# Метрики модели
print(model.metrics)
```

### Прогноз по временному ряду

```python
from forecasting.model import ExpenseForecastModel
import pandas as pd

model = ExpenseForecastModel()
df_user = pd.read_csv("user_transactions.csv")

# Получить DataFrame с прогнозами по всем месяцам
backtest_df = model.predict_series(df_user)
```

### Метрики обученной модели

```python
model = ExpenseForecastModel()

# Структура metrics:
# {
#   "baseline": {"mae": ..., "rmse": ..., "mape": ...},
#   "ridge": {"mae": ..., "rmse": ..., "mape": ...},
#   "random_forest": {"mae": ..., "rmse": ..., "mape": ...},
#   "best_model": "random_forest" | "ridge"
# }
print(model.metrics)
print(f"Лучшая модель: {model.metrics['best_model']}")
```

## Обучение модели

### Формат данных

CSV с колонками:

| Колонка | Тип | Описание |
|---------|-----|----------|
| Date | string | Дата транзакции |
| Withdrawal | float | Сумма списания |
| Deposit | float | Сумма пополнения |
| Balance | float | Баланс после операции |
| Category | string | Категория транзакции |

**Важно:** для корректного обучения нужны данные минимум за 6+ месяцев (3 месяца на лаги + 3 месяца на test).

### Запуск обучения

```bash
# С параметрами по умолчанию
python -m forecasting.train

# С указанием путей
python -m forecasting.train \
  --data ./resources/data/ci_data.csv \
  --monthly ./resources/data/monthly_features.csv \
  --model ./resources/models/expense_forecast.joblib \
  --test-months 3
```

### Параметры CLI

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--data` | Путь к CSV с транзакциями | `resources/data/ci_data.csv` |
| `--monthly` | Путь для сохранения месячных фичей | `resources/data/monthly_features.csv` |
| `--model` | Путь для сохранения модели | `resources/models/expense_forecast.joblib` |
| `--test-months` | Количество последних месяцев для теста | 3 |

### Артефакты

После обучения создаются:

| Файл | Описание |
|------|----------|
| `expense_forecast.joblib` | Сериализованная модель |
| `expense_forecast.forecast_report.json` | Метрики (baseline, ridge, rf) |
| `monthly_features.csv` | Агрегированные месячные признаки |

### Структура артефакта модели

```python
{
    "model": Ridge | RandomForestRegressor,  # Лучшая модель
    "feature_cols": [...],                    # Список признаков
    "metrics": {
        "baseline": {"mae": ..., "rmse": ..., "mape": ...},
        "ridge": {"mae": ..., "rmse": ..., "mape": ...},
        "random_forest": {"mae": ..., "rmse": ..., "mape": ...},
        "best_model": "random_forest"
    }
}
```

### Структура JSON-отчёта

```json
{
  "baseline": {
    "mae": 1234.56,
    "rmse": 1567.89,
    "mape": 15.5
  },
  "ridge": {
    "mae": 1100.00,
    "rmse": 1400.00,
    "mape": 12.3
  },
  "random_forest": {
    "mae": 950.00,
    "rmse": 1250.00,
    "mape": 10.5
  },
  "best_model": "random_forest"
}
```

## API

### ExpenseForecastModel

```python
class ExpenseForecastModel:
    def __init__(self, model_path: Path | str | None = None):
        """
        Args:
            model_path: Путь к .joblib файлу.
                       Если None — используется MODEL_PATH из config.py.
        """
        ...

    @property
    def feature_cols(self) -> list[str]:
        """Список имён признаков модели."""
        ...

    @property
    def metrics(self) -> dict[str, Any]:
        """Метрики на hold-out (baseline, ridge, rf, best_model)."""
        ...

    def predict_next_month(self, df_tx: pd.DataFrame) -> float:
        """
        Прогнозирует суммарные расходы на следующий месяц.
        
        Args:
            df_tx: DataFrame с историей транзакций (минимум 4-6 месяцев).
                   Колонки: Date, Withdrawal, Deposit, Balance, Category.
        
        Returns:
            Прогноз total_expense для следующего месяца.
        
        Raises:
            ValueError: Недостаточно данных для построения лагов.
        """
        ...

    def predict_series(self, df_tx: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает прогнозы по всем месяцам (для backtesting).
        
        Args:
            df_tx: DataFrame с транзакциями.
        
        Returns:
            DataFrame с колонками: year_month, actual, predicted.
        """
        ...
```

### Feature Engineering Functions

```python
from forecasting.features import (
    load_transactions,
    build_monthly_features,
    add_lag_features,
    train_test_split_by_month,
    build_feature_matrix,
)

# Загрузка и очистка транзакций
df = load_transactions("transactions.csv")

# Агрегация по месяцам
monthly = build_monthly_features(df)

# Добавление лагов
lagged = add_lag_features(monthly)

# Временной split
df_train, df_test = train_test_split_by_month(lagged, test_months=3)

# Подготовка X, y для sklearn
X_train, y_train, feature_cols = build_feature_matrix(df_train)
```

## Конфигурация

Файл `config.py`:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = BASE_DIR / "resources"

DATA_PATH = RESOURCES_DIR / "data" / "ci_data.csv"
MONTHLY_FEATURES_PATH = RESOURCES_DIR / "data" / "monthly_features.csv"
MODEL_PATH = RESOURCES_DIR / "models" / "expense_forecast.joblib"

RANDOM_STATE = 42
FORECAST_HORIZON = 1  # месяцев вперёд
```

## Интеграция с основным приложением

Модуль может быть интегрирован через провайдер аналогично category-classifier:

```python
from forecasting.model import ExpenseForecastModel

class ExpenseForecaster:
    def __init__(self):
        self.model = ExpenseForecastModel()
    
    def forecast_user_expenses(self, transactions_df: pd.DataFrame) -> float:
        """Прогноз расходов на следующий месяц."""
        return self.model.predict_next_month(transactions_df)
```

## Структура директории

```
forecasting/
├── forecasting/
│   ├── __init__.py
│   ├── config.py            # Конфигурация путей
│   ├── dto.py               # Data Transfer Objects
│   ├── features.py          # Feature engineering
│   ├── model.py             # ExpenseForecastModel
│   ├── schemas.py           # Схемы данных
│   └── train.py             # Скрипт обучения
├── resources/
│   ├── data/
│   │   ├── ci_data.csv      # Исходные транзакции
│   │   └── monthly_features.csv
│   └── models/
│       ├── expense_forecast.joblib
│       └── expense_forecast.forecast_report.json
├── pyproject.toml
├── uv.lock
└── README.md
```

## Примечания

### Требования к данным

- Минимум **6 месяцев** истории для обучения (3 на лаги + 3 на test)
- Минимум **4 месяца** истории для инференса (3 на лаги + 1 для прогноза)
- Транзакции должны быть отсортированы по дате

### Ограничения

- Модель обучена на агрегированных данных, не учитывает индивидуальные паттерны конкретного пользователя
- Сезонность кодируется только через месяц и квартал
- Не учитываются внешние факторы (праздники, инфляция)

### Возможные улучшения

- [ ] Добавить XGBoost/LightGBM регрессоры
- [ ] Учёт праздничных периодов
- [ ] Персонализация на данных конкретного пользователя
- [ ] Прогноз по категориям расходов

## Лицензия

MIT

