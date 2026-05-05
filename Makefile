.PHONY: install lint format test train predict clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	isort src tests
	black src tests

test:
	pytest --cov=pharma_side_effects --cov-report=term-missing

train:
	pharma-side-effects train

predict:
	pharma-side-effects predict --model-path models/random_forest.joblib --input data/processed/sample.csv --output data/processed/predictions.csv

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
