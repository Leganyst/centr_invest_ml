# Centr Invest ML

Репозиторий содержит полный цикл по классификации транзакций: подготовку/обучение модели, REST-API для инференса и вспомогательные скрипты. Основная цель — быстро поднять сервис для демо хакатона «Система интеллектуального анализа личных финансов».

## Структура проекта

- `ml/` — модуль с пайплайном обучения (`train.py`), обёрткой для инференса (`model.py`), конфигурацией (`config.py`) и генераторами данных (`data_generator.py` для semi-supervised и балансировки).
- `app/` — FastAPI-бэкенд с одним основным эндпоинтом загрузки CSV и healthcheck’ом.
- `scripts/` — утилиты (benchmarks, генераторы датасета).
- `models/` — готовые артефакты (`transaction_classifier.joblib`).
- `docker/` — конфиги инфраструктуры (например, `nginx.conf` для прокси).
- `Dockerfile`, `docker-compose.yml` — запуск сервиса в контейнерах.

## Подготовка окружения (локально)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Обучение модели

1. Подложите CSV в формате тренировочного набора (например, `ml/data/ci_data.csv`).
2. Запустите:

   ```bash
   python -m ml.train --data ./ml/data/ci_data.csv --model ./models/transaction_classifier.joblib
   ```

   Скрипт выведет отчёты по качеству, сохранит модель и структуру фичей в `models/transaction_classifier.joblib`.

3. Быстрая проверка инференса:

   ```bash
   python -m ml.model
   ```

   В консоли появится предсказание категории для тестовой транзакции.

## Запуск бэкенда (локально)

Бэкенд использует `app/main.py`. Для запуска:

```bash
uvicorn app.main:app --reload
```

По умолчанию сервис загружает модель из `ml.config.MODEL_PATH`. Можно переопределить переменными окружения `MODEL_PATH` и `MODEL_TYPE`.
В `ml/config.py` для `MODEL_TYPE` поддерживаются режимы `"simple"`, `"advanced"`, `"ensemble"`, `"catboost"` и `"neural"` (PyTorch-сеть поверх тех же фичей). Там же можно включить дополнительные шаги подготовки данных: semi-supervised label propagation (`USE_LABEL_PROPAGATION`, `UNLABELED_DATA_PATH`, `LABEL_PROPAGATION_CONFIDENCE`) и oversampling (`USE_OVERSAMPLING`, `OVERSAMPLING_MAX_PER_CLASS`), которые срабатывают только во время обучения.

### Эндпоинты

- `GET /api/health` — статус сервиса, путь к модели, тип модели.
- `POST /api/v1/transactions/upload` — принимает CSV (multipart/form-data → поле `file`), парсит колонки `Date`, `Withdrawal`, `Deposit`, `Balance`, прогоняет через модель и возвращает JSON:
  - `meta` — сколько строк обработано/отфильтровано и информация о модели.
  - `summary.by_category` — список для круговой диаграммы (категория, количество, сумма).
  - `summary.timeseries` — массив сагрегированных сумм по месяцам, пригодный для линейного графика (вложенный словарь `by_category` можно перекладывать в stacked-график).
  - `rows` — подробные строки (дата, суммы, предсказанная категория, вероятности, опционально реальная категория, если была колонка `Category`).
  - `metrics` — `macro_f1`, `balanced_accuracy`, `accuracy`, если в исходном CSV была колонка `Category`.
- `GET /api/ml/report` — глобальные метрики обученной модели (accuracy/macro F1/balanced accuracy, матрица ошибок, пер-класс статистика и топ фичей). Эти данные фронт может использовать для построения dashboards.
- `GET /api/ml/export/train` и `GET /api/ml/export/test` — скачивание CSV-сплитов, на которых модель обучалась/валидировалась.

## Запуск в Docker

### Через Dockerfile

```bash
docker build -t centr-invest-ml .
docker run --rm -p 8000:8000 -v $(pwd)/models:/app/models:ro centr-invest-ml
```

Том с моделями гарантирует, что артефакт из `models/` доступен контейнеру. При необходимости можно пробросить `MODEL_PATH`/`MODEL_TYPE` через `-e`.

### Через docker-compose

```bash
docker-compose up --build
```

В docker-compose разворачиваются два контейнера:

1. `backend` — FastAPI-приложение на Uvicorn (порт 8000 внутри сети compose, без публикации наружу).
2. `nginx` — обратный прокси, который принимает HTTP-запросы на `http://localhost:8080` и проксирует их в backend.

Модель автоматически монтируется из локальной папки `models`. Оба контейнера настроены на автоматический рестарт (`restart: unless-stopped`). При необходимости можно отредактировать `docker/nginx.conf`, чтобы менять правила маршрутизации или порты.

## Полезные заметки

- Для загрузки CSV бэкенд использует те же функции очистки дат и числовых колонок, что и тренинговый пайплайн (`ml.train._parse_date_series`). Это гарантирует согласованность с обучением.
- Если меняете версии библиотек (например, scikit-learn), переобучайте модель в том же окружении, иначе при загрузке появятся предупреждения о несовместимых версиях.
- В `app/main.py` добавлена расширенная документация к Swagger (описание полей ответа), что упрощает интеграцию фронтенда.
