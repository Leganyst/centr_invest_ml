FROM python:3.13-alpine AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /uvx /bin/

WORKDIR /opt
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project
ENV PATH="/opt/.venv/bin:$PATH"

FROM builder AS production

WORKDIR /app

RUN adduser -D app
USER app

COPY --chown=app app app/
COPY --chown=app resources resources/

ENTRYPOINT ["python", "-m", "app"]
