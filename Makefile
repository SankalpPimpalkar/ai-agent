VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
UVICORN ?= $(if $(wildcard $(VENV)/bin/uvicorn),$(VENV)/bin/uvicorn,uvicorn)
ALEMBIC ?= $(if $(wildcard $(VENV)/bin/alembic),$(VENV)/bin/alembic,alembic)
PIP ?= $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip)

.PHONY: help dev start test test-verbose eval migrate migration rollback install sync-packages clean clean-runs

help:
	@echo "Available commands:"
	@echo "  make dev           - Run the development server with auto-reload"
	@echo "  make start         - Run the production server on port 8000"
	@echo "  make test          - Run unit test suite"
	@echo "  make test-verbose  - Run unit test suite with verbose output"
	@echo "  make eval          - Run evaluations suite"
	@echo "  make migrate       - Apply all pending database migrations"
	@echo "  make migration     - Generate a new Alembic migration"
	@echo "  make rollback      - Roll back the latest migration"
	@echo "  make install       - Install dependencies from requirements.txt"
	@echo "  make sync-packages - Freeze installed packages to requirements.txt"
	@echo "  make clean         - Remove Python bytecode and cache files"
	@echo "  make clean-runs    - Remove past evaluation run logs"

dev:
	$(UVICORN) src.agent.main:app --reload --reload-dir src

start:
	$(UVICORN) src.agent.main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m unittest discover tests

test-verbose:
	$(PYTHON) -m unittest discover -v tests

eval:
	$(PYTHON) -m evals.main

migrate:
	$(ALEMBIC) upgrade head

migration:
	$(ALEMBIC) revision --autogenerate

rollback:
	$(ALEMBIC) downgrade -1

install:
	$(PIP) install -r requirements.txt

sync-packages:
	$(PIP) freeze > requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete

clean-runs:
	rm -rf evals/runs/*.json
