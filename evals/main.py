import json
import yaml
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from ollama import Message
from src.agent import run_agent
from src.utils import get_settings, init
from .schemas import Case, Run, CaseResult, EvalRunRecord
from .utils import EVAL_LAYERS

EVALS_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVALS_DIR / "runs"

async def main():
    await init()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVALS_DIR / "test.yaml", encoding="utf-8") as f:
        cases = [Case(**c) for c in yaml.safe_load(f)]

    pv = await get_settings()
    model = pv.model if pv else None
    sys_prompt = pv.system_prompt if pv else None
    tool_inst = pv.tool_instructions if pv else None
    mem_inst = pv.memory_instructions if pv else None
    thinking = (
        pv.settings.get("thinking", False)
        if (pv and isinstance(pv.settings, dict))
        else False
    )

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    case_results: List[CaseResult] = []
    passed = 0

    print(f"\n{'=' * 60}")
    print(f"Starting Evaluation Run: {run_id}")
    print(f"Model: {model} | Cases: {len(cases)}")
    print(f"{'=' * 60}\n")

    for case in cases:
        messages: List[Message] = (
            [Message(role="system", content=sys_prompt)] if sys_prompt else []
        )
        raw = await asyncio.to_thread(
            run_agent, 
            chat_history=messages, 
            user_input=case.prompt, 
            model=model,
            tool_use=True,
            thinking=thinking,
            system_prompt=sys_prompt,
            tool_instructions=tool_inst,
            memory_instructions=mem_inst,
        )
        run = Run(**raw)

        failures = {}
        for name, layer in EVAL_LAYERS.items():
            failure = layer(case, run)
            if failure:
                failures[name] = failure

        ok = not failures
        passed += int(ok)

        result_item = CaseResult(
            case=case.name,
            prompt=case.prompt,
            passed=ok,
            failures=failures,
            steps=run.steps,
            tokens=run.tokens,
            total_in=run.total_in or 0,
            total_out=run.total_out or 0,
            answer=run.answer,
            trace=run.trace,
            logs=run.logs,
        )
        case_results.append(result_item)

        status_tag = "PASS" if ok else "FAIL"
        print(f"[{status_tag}] {case.name} | Steps: {run.steps} | Tokens: {run.tokens}")
        print(f"  Prompt: {case.prompt}")
        if run.trace:
            print("  Trace:")
            for t in run.trace:
                action_part = f" (action={t.args.action})" if t.args.action else ""
                print(f"    - {t.name}{action_part}")
        print(f"  Answer: {run.answer}")
        if failures:
            for layer_name, reasons in failures.items():
                print(f"  Failures [{layer_name}]: {', '.join(reasons)}")
        print("-" * 60)

    # Compile complete run record
    eval_record = EvalRunRecord(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model=model or "default",
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=len(cases) - passed,
        cases=case_results,
    )

    run_file = RUNS_DIR / f"{run_id}.json"
    latest_file = RUNS_DIR / "latest.json"
    run_json = json.dumps(eval_record.model_dump(), indent=2)
    run_file.write_text(run_json, encoding="utf-8")
    latest_file.write_text(run_json, encoding="utf-8")

    print(f"\nEvaluation Complete: {passed}/{len(cases)} cases passed")
    print(f"Run logs saved to: {run_file.relative_to(Path.cwd())}")
    print(f"Latest run linked: {latest_file.relative_to(Path.cwd())}\n")

if __name__ == "__main__":
    asyncio.run(main())
