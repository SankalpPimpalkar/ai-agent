VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
UVICORN ?= $(if $(wildcard $(VENV)/bin/uvicorn),$(VENV)/bin/uvicorn,uvicorn)
ALEMBIC ?= $(if $(wildcard $(VENV)/bin/alembic),$(VENV)/bin/alembic,alembic)
PIP ?= $(if $(wildcard $(VENV)/bin/pip),$(VENV)/bin/pip,pip)

.PHONY: help dev start test test-verbose migrate migration rollback db-reset db-clean install sync-packages clean

help:
	@echo "Available commands:"
	@echo "  make dev           - Run development server with auto-reload"
	@echo "  make start         - Run production server on port 8000"
	@echo "  make test          - Run unit test suite"
	@echo "  make test-verbose  - Run unit test suite with verbose output"
	@echo "  make migrate       - Apply all pending database migrations"
	@echo "  make migration     - Generate a new Alembic migration"
	@echo "  make rollback      - Roll back the latest migration"
	@echo "  make db-reset      - Clean app.db and recreate all tables and migrations from scratch"
	@echo "  make db-clean      - Clean app.db and re-apply existing migrations and checkpointers"
	@echo "  make install       - Install dependencies from requirements.txt"
	@echo "  make sync-packages - Freeze installed packages to requirements.txt"
	@echo "  make clean         - Remove Python bytecode and cache files"

dev:
	$(UVICORN) app.main:app --reload --reload-dir app

start:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m unittest discover tests

test-verbose:
	$(PYTHON) -m unittest discover -v tests

migrate:
	$(ALEMBIC) upgrade head

migration:
	$(ALEMBIC) revision --autogenerate

rollback:
	$(ALEMBIC) downgrade -1

db-reset:
	@echo "Cleaning app.db and recreating all migrations and tables from scratch..."
	rm -f data/app.db data/app.db-wal data/app.db-shm
	rm -rf migrations/versions/*.py migrations/versions/__pycache__
	$(ALEMBIC) revision --autogenerate -m "initial_schema"
	$(ALEMBIC) upgrade head
	$(PYTHON) -c "import asyncio; from app.agent.memory import init_checkpointer; asyncio.run(init_checkpointer())"
	@echo "Database and migrations successfully recreated!"

db-clean:
	@echo "Cleaning app.db and re-applying existing migrations and checkpointers..."
	rm -f data/app.db data/app.db-wal data/app.db-shm
	$(ALEMBIC) upgrade head
	$(PYTHON) -c "import asyncio; from app.agent.memory import init_checkpointer; asyncio.run(init_checkpointer())"
	@echo "Database successfully cleaned and re-initialized!"

install:
	$(PIP) install -r requirements.txt

sync-packages:
	$(PIP) freeze > requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
