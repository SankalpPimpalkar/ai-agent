import unittest
from sqlmodel import select, SQLModel, Field
from sqlmodel.ext.asyncio.session import AsyncSession
from app.db import create_db_engine, get_session, init_db


class SampleModel(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str


class TestDatabaseSession(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.test_engine = create_db_engine("sqlite+aiosqlite:///:memory:")
        await init_db(self.test_engine)

    async def test_session_lifecycle(self):
        async for session in get_session(self.test_engine):
            item = SampleModel(name="test_item")
            session.add(item)
            await session.commit()
            await session.refresh(item)
            self.assertIsNotNone(item.id)

            statement = select(SampleModel).where(SampleModel.name == "test_item")
            result = await session.exec(statement)
            fetched = result.first()
            self.assertIsNotNone(fetched)
            self.assertEqual(fetched.name, "test_item")


if __name__ == "__main__":
    unittest.main()
