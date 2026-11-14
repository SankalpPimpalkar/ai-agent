import asyncio
from evals.runner import EvalRunner


async def main():
    runner = EvalRunner()
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
