import os
from pathlib import Path
from typing import Optional
from app.config import DATA_DIR, Settings

# Dedicated evaluation paths ensuring 100% isolation from production/development databases
EVAL_DATA_DIR = DATA_DIR / "eval"
EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

EVAL_DB_PATH = str(EVAL_DATA_DIR / "eval_app.db")
EVAL_DATABASE_URL = f"sqlite+aiosqlite:///{EVAL_DB_PATH}"

EVAL_CHROMA_DIR = str(EVAL_DATA_DIR / "eval_chroma")
EVAL_CHROMA_COLLECTION = "eval_rag_documents"

# LangSmith dataset and project configuration
LANGSMITH_DATASET_NAME = os.getenv("LANGSMITH_DATASET_NAME", "rag-agent-evaluation")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag-agent-eval")
DEFAULT_YAML_PATH = Path(__file__).resolve().parent / "test_cases.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Judge model configuration (allows a separate, stronger judge model if configured)
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL") or os.getenv("JUDGE_MODEL")
EVAL_JUDGE_BASE_URL = os.getenv("EVAL_JUDGE_BASE_URL")
EVAL_JUDGE_TIMEOUT = float(os.getenv("EVAL_JUDGE_TIMEOUT", "120.0"))
EVAL_JUDGE_RETRIES = int(os.getenv("EVAL_JUDGE_RETRIES", "2"))
EVAL_AGENT_RETRIES = int(os.getenv("EVAL_AGENT_RETRIES", "1"))


def get_judge_llm(timeout: Optional[float] = None):
    """Instantiates the evaluation judge LLM, using dedicated judge model if specified with reasoning=False."""
    from app.agent.llm import get_llm
    from app.config import settings

    model = EVAL_JUDGE_MODEL or settings.MODEL
    base_url = EVAL_JUDGE_BASE_URL or settings.OLLAMA_BASE_URL
    t_out = timeout if timeout is not None else EVAL_JUDGE_TIMEOUT
    return get_llm(
        model=model,
        base_url=base_url,
        temperature=0.0,
        timeout=t_out,
        reasoning=False,
    )


def get_eval_settings() -> Settings:
    """Returns application Settings configured with isolated evaluation databases."""
    return Settings(
        DATABASE_URL=EVAL_DATABASE_URL,
        CHECKPOINTER_DB_PATH=EVAL_DB_PATH,
        CHROMA_PERSIST_DIRECTORY=EVAL_CHROMA_DIR,
        CHROMA_COLLECTION_NAME=EVAL_CHROMA_COLLECTION,
        ENABLE_RAG=True,
        ENABLE_TOOLS=True,
        ENABLE_PERSISTENT_MEMORY=True,
    )
