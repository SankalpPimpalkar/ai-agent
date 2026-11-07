import yaml, asyncio
from pathlib import Path
from .utils import EVAL_LAYERS
from ollama import Message
from src.schemas import PromptVersion
from src.agent import run_agent
from src.utils import get_settings
from .schemas import Case, Run
from typing import List

EVALS_DIR = Path(__file__).resolve().parent

async def main():
    with open(EVALS_DIR / "test.yaml", encoding="utf-8") as f:
        cases = [Case(**c) for c in yaml.safe_load(f)]
    pv: PromptVersion = await get_settings()
    results, passed = [], 0

    for case in cases:
        messages: List[Message] = [Message(role="system", content=pv.system_prompt)]
        raw = await asyncio.to_thread(
            run_agent, 
            chat_history=messages, 
            user_input=case.prompt, 
            model=pv.model
        )
        run = Run(**raw)

        failures = {}
        for name, layer in EVAL_LAYERS.items():
            failure = layer(case, run)
            if failure:
                failures[name] = failure
        ok = not failures
        passed += ok
        print(f"{case.name} {'PASS' if ok else 'FAIL'}  {run.steps} steps  {' '.join(failures.keys())}")
        results.append({"case": case.name, "passed": ok, "failures": failures, **run.model_dump()})

if __name__ == "__main__":
    asyncio.run(main())