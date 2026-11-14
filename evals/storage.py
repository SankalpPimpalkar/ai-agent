import json
from pathlib import Path
from typing import Tuple
from evals.models import EvalRunRecord


class EvalStorage:
    """
    Manages persistence of evaluation records to disk.
    """

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, eval_record: EvalRunRecord) -> Tuple[Path, Path]:
        """
        Saves the run record as a timestamped JSON file and updates latest.json.
        Returns paths to (run_file, latest_file).
        """
        run_file = self.runs_dir / f"{eval_record.run_id}.json"
        latest_file = self.runs_dir / "latest.json"

        run_json = json.dumps(eval_record.model_dump(), indent=2)
        run_file.write_text(run_json, encoding="utf-8")
        latest_file.write_text(run_json, encoding="utf-8")

        return run_file, latest_file
