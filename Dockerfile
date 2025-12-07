FROM python:3.13-alpine AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.15 /uv /uvx /bin/
RUN apk add --virtual .builddeps g++ musl-dev


RUN adduser -D app
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY --chown=app third-party third-party/
RUN uv sync --locked --no-install-project --no-dev
ENV PATH="/app/.venv/bin:$PATH"

FROM builder AS production

USER app
COPY --chown=app app app/
COPY --chown=app docker/entrypoint.sh /opt/entrypoint.sh
COPY --chown=app alembic.ini .

ENTRYPOINT ["/bin/sh", "/opt/entrypoint.sh"]
CMD ["python", "-m", "app"]
