# Multi-stage Dockerfile for Fibonacci Engine

# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml setup.py ./
COPY fibonacci_engine/ ./fibonacci_engine/
COPY README.md LICENSE ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/fib /usr/local/bin/fib

# Copy application code
COPY fibonacci_engine/ ./fibonacci_engine/
COPY README.md LICENSE ./

# Create directories for persistence and reports
RUN mkdir -p fibonacci_engine/persistence fibonacci_engine/reports

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["fib", "--help"]

# Labels
LABEL org.opencontainers.image.title="Fibonacci Engine"
LABEL org.opencontainers.image.description="Universal AI Optimization Engine"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Fibonacci Engine Team"
LABEL org.opencontainers.image.licenses="MIT"
