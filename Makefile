.PHONY: install lint typecheck test test-property test-integration test-e2e test-all clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/rezaa/

test:
	pytest tests/unit/ -v

test-property:
	pytest tests/property/ -v -m property

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

test-all:
	pytest tests/ -v --cov=src/rezaa --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
