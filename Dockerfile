# ---------- Build stage ----------
FROM python:3.11-slim AS builder
WORKDIR /app

# Build tools for any dependency without a prebuilt wheel for this platform.
# Most of requirements.txt (psycopg2-binary, pymupdf, pillow, numpy, rapidfuzz)
# ships manylinux wheels and doesn't need this, but it's a cheap defensive
# fallback — never carried into the runtime image below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# PIP_EXTRA_ARGS is empty (fully verified, standard `pip install`) for every
# real build, Railway's included. It exists ONLY so a build running behind a
# network that intercepts/re-signs TLS (a corporate proxy, common on a dev
# laptop) can pass e.g. `--build-arg PIP_EXTRA_ARGS="--trusted-host
# pypi.org --trusted-host files.pythonhosted.org"` for a local test build —
# never something to set for a real deploy.
ARG PIP_EXTRA_ARGS=""
RUN pip install --no-cache-dir --prefix=/install ${PIP_EXTRA_ARGS} -r requirements.txt


# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime
WORKDIR /app

# Run as a non-root user.
RUN addgroup --system app && adduser --system --ingroup app app

# Installed dependencies only — no compiler, no pip cache, no build context.
COPY --from=builder /install /usr/local

# Application code. Deliberately selective (not `COPY . .`): keeps out
# tests/, .env*, docker-compose.yml, and anything else not needed to run.
# `scripts/` and `alembic/` are needed in the image for the same reason
# prod_app's own Dockerfile ships its provisioning scripts: migrations and
# the legacy/Indian-brand catalog seed are run via `docker run <image>
# python -m ...`, using this same image, not baked into image build time
# (this service's own database is provisioned as a deploy step, not a build
# step — see README "Setup & Installation").
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts ./scripts

RUN chown -R app:app /app
USER app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:' + os.environ.get('PORT','8000') + '/health', timeout=3)"

# Shell form (not exec-form JSON array) so $PORT is expanded — Railway
# injects PORT at runtime; ${PORT:-8000} keeps `docker run -p 8000:8000`
# working locally with no PORT set at all.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
