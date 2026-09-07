# --------------------------------------------------------------------------
# Stage 1: builder — install all Python dependencies into /install
# --------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --upgrade pip --quiet \
    && pip install --default-timeout=1000 --prefix=/install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------
# Stage 2: runner — lean runtime image, no training code or test suite
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runner

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only the runtime artefacts the service needs
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY config.yaml .
COPY .env.example .

# Ensure Python finds the src package
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
