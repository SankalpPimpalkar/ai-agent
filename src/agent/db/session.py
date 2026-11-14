from typing import Optional
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from src.agent.settings import SETTINGS

SQLModel.metadata.naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def create_db_engine(database_url: Optional[str] = None) -> AsyncEngine:
    """Creates an async SQLAlchemy engine with injected or default database URL."""
    url = database_url or SETTINGS.DATABASE_URL
    return create_async_engine(url=url, echo=False)


engine = create_db_engine()


async def initdb(db_engine: Optional[AsyncEngine] = None):
    eng = db_engine or engine
    async with eng.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session(db_engine: Optional[AsyncEngine] = None):
    eng = db_engine or engine
    async_session = sessionmaker(bind=eng, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
