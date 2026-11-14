from pathlib import Path
from typing import Dict, List, Optional
from evals.models import Case, Run


class EvalReporter:
    """
    Handles formatting and emitting evaluation output to the console.
    """

    @staticmethod
    def print_run_header(run_id: str, model: Optional[str], total_cases: int) -> None:
        print(f"\n{'=' * 60}")
        print(f"Starting Evaluation Run: {run_id}")
        print(f"Model: {model} | Cases: {total_cases}")
        print(f"{'=' * 60}\n")

    @staticmethod
    def print_case_start(case: Case) -> None:
        print("-" * 60)
        print(f"Case: {case.name}")
        print(f"Prompt: {case.prompt}\n")

    @staticmethod
    def print_case_result(
        case: Case, run: Run, ok: bool, failures: Dict[str, List[str]]
    ) -> None:
        status_tag = "PASS" if ok else "FAIL"
        print(f"[{status_tag}] {case.name} | Steps: {run.steps} | Tokens: {run.tokens}")
        if run.trace:
            print("Trace:")
            for t in run.trace:
                action_part = f" (action={t.args.action})" if t.args.action else ""
                print(f"    - {t.name}{action_part}")
        print(f"Answer: {run.answer}")
        if failures:
            for layer_name, reasons in failures.items():
                print(f"Failures [{layer_name}]: {', '.join(reasons)}")
        print("-" * 60)

    @staticmethod
    def print_run_summary(
        passed: int, total_cases: int, run_file: Path, latest_file: Path
    ) -> None:
        cwd = Path.cwd()
        print(f"\nEvaluation Complete: {passed}/{total_cases} cases passed")
        try:
            print(f"Run logs saved to: {run_file.relative_to(cwd)}")
            print(f"Latest run linked: {latest_file.relative_to(cwd)}\n")
        except ValueError:
            print(f"Run logs saved to: {run_file}")
            print(f"Latest run linked: {latest_file}\n")
