FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY wafer_defect_pipeline/__init__.py ./wafer_defect_pipeline/__init__.py

RUN python -m pip install --upgrade pip \
 && pip install -e ".[mlops]"

COPY wafer_defect_pipeline/ ./wafer_defect_pipeline/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

RUN useradd -m -u 1000 wafer && chown -R wafer:wafer /app
USER wafer

ENTRYPOINT ["python"]
CMD ["scripts/run_train_ddpm.py", "--help"]
