# CortexOS - Cognitive Architecture System
# Multi-stage build for optimized production image

FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CORTEXOS_ROOT=/app \
    DATA_DIR=/app/data \
    LOGS_DIR=/app/logs \
    CONFIG_DIR=/app/config \
    TEMP_DIR=/app/temp \
    CACHE_DIR=/app/cache

# Create non-root user
RUN useradd -m -u 1000 cortexos && \
    mkdir -p /app && \
    chown -R cortexos:cortexos /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/cortexos/.local

# Copy application code
COPY --chown=cortexos:cortexos . /app/

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/data \
    /app/logs \
    /app/config \
    /app/temp \
    /app/cache \
    /app/neural_data/phase1 \
    /app/neural_data/phase2 \
    /app/neural_data/phase3 \
    /app/neural_data/phase4 \
    /app/neural_data/phase5 \
    /app/neural_data/phase6 \
    /app/storage/contracts \
    /app/storage/cube_storage \
    /app/temp/batch_staging \
    /app/temp/ingestion_queue \
    /app/temp/output_queue \
    /app/temp/processing \
    /app/temp/stream_buffers \
    /app/backups \
    /app/recovery \
    /app/checkpoints \
    /app/archives \
    /app/metrics \
    /app/analytics \
    /app/reports \
    /app/alerts && \
    chown -R cortexos:cortexos /app && \
    chmod -R 755 /app

# Switch to non-root user
USER cortexos

# Update PATH to include user-installed packages
ENV PATH=/home/cortexos/.local/bin:$PATH

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Default command
CMD ["python3", "1_supervisor.py"]
