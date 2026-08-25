FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 cabcast
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY conf ./conf
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

COPY tests ./tests
RUN mkdir -p data/bronze data/silver data/gold data/external artifacts reports/figures \
    && chown -R cabcast:cabcast /app

USER cabcast
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "cabcast.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
