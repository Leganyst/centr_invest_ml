# API-справочник для фронтенда

Документ описывает, что именно возвращают ключевые ручки сервиса. Все ответы — JSON, если не оговорено иначе.

## 1. Healthcheck

**GET `/api/health`**

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "catboost",
  "model_path": "models/transaction_classifier.joblib"
}
```

Что рисовать:

- зелёный индикатор готовности (`model_loaded`);
- подпись “Модель: catboost (transaction_classifier.joblib)”.

## 2. Загрузка CSV и предсказание

**POST `/api/v1/transactions/upload`** (multipart/form-data, поле `file`).

Возвращает:

```json
{
  "meta": { ... },
  "summary": {
    "by_category": [ ... ],
    "timeseries": [ ... ]
  },
  "rows": [ ... ],
  "metrics": { ... }
}
```

### 2.1. `meta`

- `total_rows` — сколько строк в исходном CSV.
- `processed_rows` — сколько прошло очистку.
- `failed_rows` — сколько отброшено (проблемы с датой/числом).
- `model_type`, `model_version` — информация о модели.

Используйте для KPI-блока “Обработано X из Y строк”, “Модель: catboost v…”.

### 2.2. `summary.by_category`

Массив объектов:

```json
{
  "category": "Food",
  "count": 900,
  "amount": 45000.0
}
```

Рекомендуем:

- круговая диаграмма по `amount` или `count`;
- таблица “Категория / количество / сумма”.

### 2.3. `summary.timeseries`

```json
{
  "period": "2023-01",
  "total_amount": 50000.0,
  "by_category": {
    "Food": 8000.0,
    "Misc": 7000.0,
    ...
  }
}
```

Подходит для:

- линейного графика `total_amount` по месяцам;
- stacked-bar графика, где `by_category` — слои.

### 2.4. `rows`

Каждый элемент:

```json
{
  "index": 42,
  "date": "2023-01-03T00:00:00",
  "withdrawal": 100.0,
  "deposit": 0.0,
  "balance": 1500.0,
  "predicted_category": "Food",
  "probabilities": { "Food": 0.85, "Misc": 0.1, ... },
  "actual_category": "Food"
}
```

Использование:

- табличное представление транзакций;
- при клике можно раскрывать `probabilities`;
- если есть `actual_category`, подсвечивать совпадения/ошибки.

### 2.5. `metrics`

```json
{
  "has_ground_truth": true,
  "macro_f1": 0.71,
  "balanced_accuracy": 0.78,
  "accuracy": 0.85
}
```

Если в CSV не было колонки `Category`, `has_ground_truth=false` и остальные поля `null`. Можно показывать карточку “Качество по загруженному файлу”.

## 3. Отчёт о модели

**GET `/api/ml/report`**

```json
{
  "model": { "type": "catboost", "version": "transaction_classifier.joblib", "random_state": 42 },
  "metrics": {
    "accuracy": 0.86,
    "macro_f1": 0.62,
    "weighted_f1": 0.84,
    "balanced_accuracy": 0.69,
    "precision_weighted": 0.83,
    "recall_weighted": 0.79,
    "cv_macro_f1_mean": 0.58,
    "cv_macro_f1_std": 0.04,
    "classification_report": "precision recall f1-score ..."
  },
  "per_class": [
    { "category": "Food", "precision": 0.99, "recall": 0.92, "f1": 0.96, "support": 183 },
    ...
  ],
  "confusion_matrix": {
    "labels": ["Food", "Misc", ...],
    "matrix": [
      [169, 7, 0, ...],
      ...
    ]
  },
  "feature_importance": [
    { "feature": "Withdrawal", "importance": 0.32 },
    ...
  ]
}
```

Рекомендации:

- верхний KPI-блок — accuracy/macro F1/balanced accuracy.
- heatmap для `confusion_matrix`.
- таблица “Per-class metrics” для precision/recall/F1.
- вертикальный bar chart по `feature_importance` (10 фич).

## 4. Экспорт сплитов

**GET `/api/ml/export/train`**  
**GET `/api/ml/export/test`**

Возвращают CSV-файлы тех же колонок, что исходный датасет. Просто дайте пользователю ссылку «Скачать train/test».
