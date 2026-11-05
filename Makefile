dev:
	uvicorn src.main:app --reload --reload-dir src

migration:
	alembic revision --autogenerate 

migrate:
	alembic upgrade head

rollback:
	alembic downgrade -1

sync_packages:
	pip freeze > requirements.txt