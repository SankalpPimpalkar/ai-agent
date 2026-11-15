import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from langsmith import Client
from eval.config import DEFAULT_YAML_PATH, LANGSMITH_DATASET_NAME

logger = logging.getLogger(__name__)


def load_yaml_dataset(yaml_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Loads and validates evaluation test cases from YAML.
    """
    path = Path(yaml_path or DEFAULT_YAML_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation YAML dataset file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, list):
        test_cases = data
    elif isinstance(data, dict):
        test_cases = data.get("test_cases", [])
    else:
        raise ValueError(f"Expected YAML file at {path} to contain a list or mapping.")

    if not isinstance(test_cases, list) or len(test_cases) == 0:
        raise ValueError(f"YAML dataset at {path} must contain a non-empty 'test_cases' list.")

    for i, tc in enumerate(test_cases):
        if "question" not in tc or "reference_answer" not in tc:
            raise ValueError(
                f"Test case index {i} (id: {tc.get('id')}) is missing required 'question' or 'reference_answer'."
            )

    return {
        "test_cases": test_cases,
    }


def sync_dataset_to_langsmith(
    test_cases: List[Dict[str, Any]],
    dataset_name: Optional[str] = None,
    client: Optional[Client] = None,
) -> Optional[str]:
    """
    Uploads or updates the test cases in a LangSmith dataset.
    Returns dataset_id if synced, or None if LangSmith is unconfigured or unreachable.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        logger.info(
            "LANGCHAIN_API_KEY not configured in environment. Skipping LangSmith cloud dataset synchronization."
        )
        return None

    target_name = dataset_name or LANGSMITH_DATASET_NAME
    try:
        smith_client = client or Client()

        # Check if dataset already exists
        dataset = None
        try:
            if smith_client.has_dataset(dataset_name=target_name):
                dataset = smith_client.read_dataset(dataset_name=target_name)
                logger.info(f"Found existing LangSmith dataset '{target_name}' (id: {dataset.id}).")
        except Exception:
            dataset = None

        if dataset is None:
            dataset = smith_client.create_dataset(
                dataset_name=target_name,
                description="RAG Agent evaluation dataset with ground truth references for correctness, relevance, and groundedness.",
            )
            logger.info(f"Created new LangSmith dataset '{target_name}' (id: {dataset.id}).")

            inputs = [{"question": tc["question"]} for tc in test_cases]
            outputs = [{"reference_answer": tc["reference_answer"]} for tc in test_cases]
            metadata = [
                {
                    "id": tc.get("id", f"tc_{i+1}"),
                    "expected_sources": tc.get("expected_sources", []),
                    "tags": tc.get("tags", []),
                }
                for i, tc in enumerate(test_cases)
            ]

            smith_client.create_examples(
                inputs=inputs,
                outputs=outputs,
                metadata=metadata,
                dataset_id=dataset.id,
            )
            logger.info(f"Uploaded {len(test_cases)} test case examples to LangSmith dataset '{target_name}'.")

        return str(dataset.id)

    except Exception as exc:
        logger.warning(
            f"Failed to synchronize dataset to LangSmith: {exc}. Proceeding with local evaluation."
        )
        return None
