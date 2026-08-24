PYTHON ?= python
PKG := src/urbanflow

.DEFAULT_GOAL := help
.PHONY: help install data data-synthetic silver gold train report all serve test lint format docker clean

help:
	@echo "UrbanFlow"
	@echo ""
	@echo "  make install         install the package and dev dependencies"
	@echo "  make all             run the whole pipeline end to end"
	@echo "  make data            download real NYC TLC trip records"
	@echo "  make data-synthetic  generate the offline dataset instead"
	@echo "  make silver          clean and validate trips"
	@echo "  make gold            build the zone-hour demand panel"
	@echo "  make train           train models and back-test them"
	@echo "  make report          regenerate every figure and results file"
	@echo "  make serve           start the forecasting API on port 8000"
	@echo "  make test            run the test suite"
	@echo "  make lint            run ruff"
	@echo "  make docker          build the container image"
	@echo "  make clean           remove generated data and artifacts"

install:
	$(PYTHON) -m pip install -e ".[serve,dev]"

data:
	$(PYTHON) -m urbanflow.cli ingest --source remote

data-synthetic:
	$(PYTHON) -m urbanflow.cli ingest --source synthetic

silver:
	$(PYTHON) -m urbanflow.cli silver

gold:
	$(PYTHON) -m urbanflow.cli gold

train:
	$(PYTHON) -m urbanflow.cli train

report:
	$(PYTHON) -m urbanflow.cli report

all:
	$(PYTHON) -m urbanflow.cli all

serve:
	$(PYTHON) -m urbanflow.cli serve

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests

docker:
	docker build -t urbanflow:latest .

clean:
	rm -rf data/bronze/* data/silver/* data/gold/* data/external/* artifacts reports/figures/* reports/*.json reports/*.md reports/*.csv reports/*.parquet
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
