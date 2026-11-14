import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional
import yaml
from ollama import Message
from src.agent.core import run_agent
from src.agent.db.repositories import get_settings, init
from src.agent.settings import SETTINGS
from evals.layers import BaseEvalLayer, DEFAULT_EVAL_LAYERS
from evals.models import Case, CaseResult, EvalRunRecord, Run
from evals.reporter import EvalReporter
from evals.storage import EvalStorage

EVALS_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVALS_DIR / "runs"


class EvalRunner:
    def __init__(
        self,
        evals_dir: Optional[Path] = None,
        layers: Optional[List[BaseEvalLayer]] = None,
        reporter: Optional[EvalReporter] = None,
        storage: Optional[EvalStorage] = None,
        agent_runner: Optional[Callable] = None,
    ):
        self.evals_dir = evals_dir or EVALS_DIR
        self.layers = layers or list(DEFAULT_EVAL_LAYERS)
        self.reporter = reporter or EvalReporter()
        self.storage = storage or EvalStorage(self.evals_dir / "runs")
        self.agent_runner = agent_runner or run_agent

    def load_cases(self, test_file: Optional[Path] = None) -> List[Case]:
        filepath = test_file or (self.evals_dir / "cases" / "test.yaml")
        if not filepath.exists():
            filepath = self.evals_dir / "test.yaml"
        with open(filepath, encoding="utf-8") as f:
            return [Case(**c) for c in yaml.safe_load(f)]

    @staticmethod
    def clean_scratch_directories() -> None:
        paths = [
            SETTINGS.paths.tools_dir,
            SETTINGS.paths.memory_dir
        ]
        for d in paths:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        SETTINGS.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    async def run_all(self, test_file: Optional[Path] = None) -> EvalRunRecord:
        await init()
        cases = self.load_cases(test_file)

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

        self.reporter.print_run_header(run_id, model, len(cases))

        for case in cases:
            self.reporter.print_case_start(case)
            self.clean_scratch_directories()

            messages: List[Message] = (
                [Message(role="system", content=sys_prompt)] if sys_prompt else []
            )
            raw = await asyncio.to_thread(
                self.agent_runner,
                chat_history=messages,
                user_input=case.prompt,
                model=model,
                tool_use=True,
                thinking=thinking,
                system_prompt=sys_prompt,
                tool_instructions=tool_inst,
                memory_instructions=mem_inst,
                max_steps=case.max_steps,
                verbose=True,
            )
            run = Run(**raw)

            failures = {}
            for layer in self.layers:
                layer_failures = layer.check(case, run)
                if layer_failures:
                    failures[layer.name] = layer_failures

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
            self.reporter.print_case_result(case, run, ok, failures)

        eval_record = EvalRunRecord(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model or "default",
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=len(cases) - passed,
            cases=case_results,
        )

        run_file, latest_file = self.storage.save_run(eval_record)
        self.reporter.print_run_summary(passed, len(cases), run_file, latest_file)
        return eval_record
