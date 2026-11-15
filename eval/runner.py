import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from app.agent.builder import AgentRunner
from app.agent.llm import get_llm
from app.agent.memory import init_checkpointer
from app.db import create_db_engine, init_db
from app.rag.embeddings import get_embeddings
from app.rag.ingest import ingest_document
from app.rag.store import reset_vector_store, set_vector_store
from app.tools.local import LOCAL_TOOLS
from app.tools.rag import search_knowledge_base
from eval.config import (
    DEFAULT_YAML_PATH,
    EVAL_AGENT_RETRIES,
    EVAL_CHROMA_COLLECTION,
    EVAL_CHROMA_DIR,
    EVAL_DATABASE_URL,
    EVAL_DATA_DIR,
    EVAL_DB_PATH,
    EVAL_JUDGE_RETRIES,
    EVAL_JUDGE_TIMEOUT,
    LANGSMITH_DATASET_NAME,
    LANGSMITH_PROJECT,
    RESULTS_DIR,
    get_judge_llm,
)
from eval.dataset_loader import load_yaml_dataset, sync_dataset_to_langsmith
from eval.evaluators import (
    MetricResult,
    TurnEvaluationResult,
    create_langsmith_evaluators,
    evaluate_turn,
)

logger = logging.getLogger("rag_evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RagEvaluator:
    """
    RAG Evaluation harness that tests Correctness (fact-based verdicts), Answer Relevance,
    Groundedness, and Retrieval Relevance (recall@k & precision@k in code).
    Maintains strict database isolation and excludes error/judge_error cases from score averages.
    """

    def __init__(
        self,
        yaml_path: Optional[Path | str] = None,
        llm: Optional[BaseChatModel] = None,
        embeddings: Optional[Embeddings] = None,
        clean_on_start: bool = False,
        clean_on_finish: bool = False,
        db_path: Optional[str] = None,
        chroma_dir: Optional[str] = None,
    ):
        self.yaml_path = Path(yaml_path or DEFAULT_YAML_PATH)
        self.llm = llm
        self.embeddings = embeddings or get_embeddings()
        self.clean_on_start = clean_on_start
        self.clean_on_finish = clean_on_finish

        self.db_path = db_path or EVAL_DB_PATH
        self.db_url = f"sqlite+aiosqlite:///{self.db_path}" if db_path else EVAL_DATABASE_URL
        self.chroma_dir = chroma_dir or EVAL_CHROMA_DIR
        self.collection_name = EVAL_CHROMA_COLLECTION

        self.eval_engine = None
        self.eval_vector_store = None
        self._initialized = False

    async def setup(self) -> None:
        """Initializes isolated evaluation database and ChromaDB collection."""
        if self._initialized:
            return

        if self.clean_on_start:
            self._clean_eval_storage()

        os.makedirs(EVAL_DATA_DIR, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)

        self.eval_engine = create_db_engine(self.db_url)
        await init_db(self.eval_engine)
        await init_checkpointer(self.db_path)

        self.eval_vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.chroma_dir,
        )
        set_vector_store(self.eval_vector_store)
        self._initialized = True
        logger.info("Evaluation environment successfully initialized.")

    def _clean_eval_storage(self) -> None:
        """Safely cleans the isolated evaluation data directories."""
        if os.path.exists(self.chroma_dir):
            try:
                shutil.rmtree(self.chroma_dir)
            except Exception as e:
                logger.warning(f"Failed to remove eval chroma dir: {e}")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception as e:
                logger.warning(f"Failed to remove eval sqlite db: {e}")
        for wal in [f"{self.db_path}-wal", f"{self.db_path}-shm"]:
            if os.path.exists(wal):
                try:
                    os.remove(wal)
                except Exception:
                    pass

    async def teardown(self) -> None:
        """Resets active vector store and cleans up resources if requested."""
        reset_vector_store()
        if self.eval_engine:
            await self.eval_engine.dispose()
        if self.clean_on_finish:
            self._clean_eval_storage()
        self._initialized = False
        logger.info("Evaluation environment torn down.")

    async def run(
        self,
        custom_test_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes evaluation:
        1. Runs each case sequentially (low concurrency prevents GPU contention).
        2. Retries empty agent answers once; marks status='error' if still empty (never reaches judge).
        3. Evaluates retrieval relevance in code (recall@k & precision@k).
        4. Evaluates fact-based correctness, answer relevance, and groundedness with judge retries.
        5. Excludes error/judge_error cases from score averages, reporting 'X.XX over Y of Z cases'.
        """
        await self.setup()

        try:
            if custom_test_cases is not None:
                test_cases = custom_test_cases
            else:
                dataset = load_yaml_dataset(self.yaml_path)
                test_cases = dataset["test_cases"]

            logger.info(f"Loaded {len(test_cases)} evaluation test cases from {self.yaml_path}")

            # 1. Sync to LangSmith if API key is provided
            dataset_id = sync_dataset_to_langsmith(
                test_cases=test_cases,
                dataset_name=LANGSMITH_DATASET_NAME,
            )

            results: List[Dict[str, Any]] = []
            judge_llm = self.llm or get_judge_llm()

            # Build tools for the agent (local tools + search_knowledge_base connected to eval_store)
            eval_tools = list(LOCAL_TOOLS) + [search_knowledge_base]

            for idx, tc in enumerate(test_cases):
                case_id = tc.get("id", f"case_{idx+1}")
                question = tc["question"]
                reference_answer = tc["reference_answer"]
                key_facts = tc.get("key_facts", [])
                must_retrieve = tc.get("must_retrieve")
                acceptable_sections = tc.get("acceptable_sections")

                logger.info(f"Evaluating [{idx+1}/{len(test_cases)}] (ID: {case_id}): {question[:60]}...")

                agent_response = ""
                retrieved_chunks: List[Any] = []
                context_str = ""
                case_status = "success"
                error_message = None

                # 1. Execute agent turn with retry on empty response
                max_agent_attempts = 1 + EVAL_AGENT_RETRIES  # 1 initial + 1 retry = 2 attempts
                for attempt in range(max_agent_attempts):
                    try:
                        async with AsyncSqliteSaver.from_conn_string(self.db_path) as saver:
                            agent = AgentRunner(
                                llm=self.llm,
                                tools=eval_tools,
                                checkpointer=saver,
                                db_engine=self.eval_engine,
                            )
                            exec_result = await asyncio.wait_for(
                                agent.execute_detailed(
                                    message=question,
                                    session_id=f"eval-session-{case_id}-try{attempt}",
                                ),
                                timeout=180.0,
                            )

                        raw_resp = exec_result.get("response", "")
                        agent_response = raw_resp.strip() if raw_resp else ""
                        retrieved_chunks = exec_result.get("retrieved_contexts", [])

                        # Log raw model output for visibility
                        logger.info(
                            f"Case {case_id} [attempt {attempt+1}/{max_agent_attempts}] "
                            f"raw agent response: {repr(agent_response)[:300]}"
                        )

                        if agent_response:
                            break
                        else:
                            logger.warning(
                                f"Case {case_id} [attempt {attempt+1}/{max_agent_attempts}] "
                                f"produced empty response. Full exec_result: {exec_result}"
                            )
                    except asyncio.TimeoutError:
                        logger.warning(f"Case {case_id} [attempt {attempt+1}/{max_agent_attempts}] timed out after 180s.")
                        error_message = f"Agent execution timed out after 180s on attempt {attempt+1}."
                    except Exception as e:
                        logger.warning(f"Case {case_id} [attempt {attempt+1}/{max_agent_attempts}] failed: {e}", exc_info=True)
                        error_message = f"Agent execution error on attempt {attempt+1}: {e}"

                # 2. Check if agent response is still empty
                if not agent_response:
                    case_status = "error"
                    msg = error_message or "Agent produced empty response after retry."
                    logger.error(f"Case {case_id} marked as status='error'. Empty answer will not reach the judge.")
                    eval_metrics = TurnEvaluationResult(
                        correctness=MetricResult(metric="correctness", score=None, reasoning=msg, status="skipped"),
                        answer_relevance=MetricResult(metric="answer_relevance", score=None, reasoning=msg, status="skipped"),
                        groundedness=MetricResult(metric="groundedness", score=None, reasoning=msg, status="skipped"),
                        retrieval_relevance=MetricResult(metric="retrieval_relevance", score=None, reasoning=msg, status="skipped"),
                        average_score=None,
                        status="error",
                    )
                else:
                    # If the agent didn't invoke the search tool directly, query retriever
                    if not retrieved_chunks:
                        retriever = self.eval_vector_store.as_retriever(search_kwargs={"k": 3})
                        docs = await retriever.ainvoke(question)
                        retrieved_chunks = docs
                        context_str = "\n\n".join(d.page_content for d in docs)
                    else:
                        context_str = "\n\n".join(
                            d.page_content if hasattr(d, "page_content") else str(d)
                            for d in retrieved_chunks
                        )

                    # 3. Evaluate turn modularly (judge is only called for valid non-empty responses)
                    eval_metrics = await evaluate_turn(
                        question=question,
                        agent_response=agent_response,
                        reference_answer=reference_answer,
                        retrieved_context=context_str,
                        retrieved_chunks=retrieved_chunks,
                        key_facts=key_facts,
                        must_retrieve=must_retrieve,
                        acceptable_sections=acceptable_sections,
                        llm=judge_llm,
                        timeout=EVAL_JUDGE_TIMEOUT,
                        retries=EVAL_JUDGE_RETRIES,
                    )
                    case_status = eval_metrics.status

                results.append({
                    "id": case_id,
                    "status": case_status,
                    "error_message": error_message,
                    "question": question,
                    "reference_answer": reference_answer,
                    "agent_response": agent_response,
                    "retrieved_context": context_str,
                    "tags": tc.get("tags", []),
                    "metrics": eval_metrics.to_dict(),
                })

            # 4. Calculate overall averages — excluding cases with null scores or error status
            total = len(results)

            def calc_metric_avg(metric_name: str) -> tuple[Optional[float], int, str]:
                valid_cases = [
                    r for r in results
                    if r["status"] != "error"
                    and r["metrics"][metric_name]["score"] is not None
                ]
                count = len(valid_cases)
                if count == 0:
                    return None, 0, f"N/A (0 of {total} cases)"
                avg = round(sum(r["metrics"][metric_name]["score"] for r in valid_cases) / count, 3)
                display = f"{avg:.2f} over {count} of {total} cases"
                return avg, count, display

            avg_correctness, valid_corr, disp_corr = calc_metric_avg("correctness")
            avg_relevance, valid_rel, disp_rel = calc_metric_avg("answer_relevance")
            avg_groundedness, valid_gnd, disp_gnd = calc_metric_avg("groundedness")
            avg_retrieval, valid_ret, disp_ret = calc_metric_avg("retrieval_relevance")

            valid_overall_cases = [
                r for r in results
                if r["status"] not in ("error", "judge_error")
                and r["metrics"]["average_score"] is not None
            ]
            evaluated_count = len(valid_overall_cases)
            excluded_count = total - evaluated_count

            valid_avgs = [
                v for v in [avg_correctness, avg_relevance, avg_groundedness, avg_retrieval]
                if v is not None
            ]
            overall_score = round(sum(valid_avgs) / len(valid_avgs), 3) if valid_avgs else None
            overall_display = (
                f"{overall_score:.2f} over {evaluated_count} of {total} cases"
                if overall_score is not None
                else f"N/A (0 of {total} cases)"
            )
            if excluded_count > 0:
                overall_display += f" ({excluded_count} excluded due to error/judge_error)"

            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_name": LANGSMITH_DATASET_NAME,
                "dataset_id": dataset_id,
                "total_cases": total,
                "evaluated_cases": evaluated_count,
                "excluded_cases": excluded_count,
                "score_display": overall_display,
                "metrics_summary": {
                    "correctness": avg_correctness,
                    "answer_relevance": avg_relevance,
                    "groundedness": avg_groundedness,
                    "retrieval_relevance": avg_retrieval,
                    "overall_score": overall_score,
                    "total_cases": total,
                    "evaluated_cases": evaluated_count,
                    "excluded_cases": excluded_count,
                    "score_display": overall_display,
                    "metrics_display": {
                        "correctness": disp_corr,
                        "answer_relevance": disp_rel,
                        "groundedness": disp_gnd,
                        "retrieval_relevance": disp_ret,
                    },
                },
                "cases": results,
            }

            self._print_terminal_report(summary)
            self._save_reports(summary)

            # If LangSmith evaluation is configured, run evaluation benchmark through LangSmith
            if dataset_id and os.getenv("LANGCHAIN_API_KEY"):
                await self._run_langsmith_benchmark(dataset_name=LANGSMITH_DATASET_NAME)

            return summary

        finally:
            await self.teardown()

    async def _run_langsmith_benchmark(self, dataset_name: str) -> None:
        """Runs the LangSmith cloud evaluation benchmark using langsmith.evaluation.evaluate."""
        try:
            from langsmith.evaluation import evaluate

            eval_tools = list(LOCAL_TOOLS) + [search_knowledge_base]

            async def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
                question = inputs.get("question", "")
                async with AsyncSqliteSaver.from_conn_string(self.db_path) as saver:
                    agent = AgentRunner(
                        llm=self.llm,
                        tools=eval_tools,
                        checkpointer=saver,
                        db_engine=self.eval_engine,
                    )
                    res = await agent.execute_detailed(
                        message=question,
                        session_id=f"eval-ls-{datetime.now().timestamp()}",
                    )
                return {
                    "response": res.get("response", ""),
                    "retrieved_context": "\n\n".join(res.get("retrieved_contexts", [])),
                    "retrieved_chunks": res.get("retrieved_contexts", []),
                }

            judge_llm = self.llm or get_judge_llm()
            evaluators = create_langsmith_evaluators(llm=judge_llm)

            logger.info("Executing cloud LangSmith evaluation suite...")
            await evaluate(
                target,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=LANGSMITH_PROJECT,
                max_concurrency=1,
            )
            logger.info("LangSmith evaluation complete.")
        except Exception as e:
            logger.warning(f"LangSmith cloud evaluation could not run: {e}")

    def _print_terminal_report(self, summary: Dict[str, Any]) -> None:
        """Prints a human-readable summary table to stdout."""
        m_sum = summary["metrics_summary"]
        overall = m_sum.get("overall_score")
        overall_str = f"{overall * 100:.1f}%" if overall is not None else "N/A"

        print("\n" + "=" * 80)
        print(f"RAG EVALUATION BENCHMARK REPORT — {summary['dataset_name']}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Overall Score: {overall_str} — {summary.get('score_display', '')}")
        print("=" * 80)
        print(f"{'Case ID':<8} | {'Status':<11} | {'Correctness':<11} | {'Relevance':<9} | {'Grounded':<8} | {'Retrieval':<9} | {'Avg':<5}")
        print("-" * 80)
        for c in summary["cases"]:
            m = c["metrics"]
            cid = c["id"]
            stat = c.get("status", "success")

            def fmt(val):
                return f"{val:.2f}" if isinstance(val, (int, float)) else "N/A"

            corr = fmt(m["correctness"]["score"])
            rel = fmt(m["answer_relevance"]["score"])
            gnd = fmt(m["groundedness"]["score"])
            ret = fmt(m["retrieval_relevance"]["score"])
            avg = fmt(m.get("average_score"))
            print(f"{cid:<8} | {stat:<11} | {corr:<11} | {rel:<9} | {gnd:<8} | {ret:<9} | {avg:<5}")
        print("=" * 80 + "\n")

    def _save_reports(self, summary: Dict[str, Any]) -> None:
        """Saves evaluation results as both JSON and Markdown in eval/results/."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = RESULTS_DIR / "eval_report.json"
        md_path = RESULTS_DIR / "eval_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Markdown Report
        metrics = summary["metrics_summary"]
        overall = metrics.get("overall_score")
        overall_str = f"{overall * 100:.1f}%" if overall is not None else "N/A"

        def fmt_pct(val):
            return f"{val * 100:.1f}%" if isinstance(val, (int, float)) else "N/A"

        disp_map = metrics.get("metrics_display", {})

        md_lines = [
            "# RAG Evaluation Report",
            f"\n- **Generated At**: `{summary['timestamp']}`",
            f"- **Dataset**: `{summary['dataset_name']}`",
            f"- **Score Display**: `{summary.get('score_display', '')}`",
            f"- **Total Test Cases**: {summary['total_cases']}",
            f"- **Evaluated Cases**: {summary.get('evaluated_cases', summary['total_cases'])}",
            f"- **Excluded Cases**: {summary.get('excluded_cases', 0)}",
            f"- **Overall Score**: **{overall_str}**\n",
            "## Summary Metrics\n",
            "| Metric | Score | Display |",
            "| :--- | :--- | :--- |",
            f"| **Correctness** | {fmt_pct(metrics.get('correctness'))} | {disp_map.get('correctness', '')} |",
            f"| **Answer Relevance** | {fmt_pct(metrics.get('answer_relevance'))} | {disp_map.get('answer_relevance', '')} |",
            f"| **Groundedness** | {fmt_pct(metrics.get('groundedness'))} | {disp_map.get('groundedness', '')} |",
            f"| **Retrieval Relevance** | {fmt_pct(metrics.get('retrieval_relevance'))} | {disp_map.get('retrieval_relevance', '')} |",
            f"| **Composite Average** | **{overall_str}** | {summary.get('score_display', '')} |\n",
            "## Test Case Details\n",
        ]

        for c in summary["cases"]:
            m = c["metrics"]
            stat = c.get("status", "success")
            err = f" (Error: {c.get('error_message')})" if c.get("error_message") else ""

            def fmt_score(val):
                return f"`{val:.2f}`" if isinstance(val, (int, float)) else "`null`"

            md_lines.extend([
                f"### Test Case `{c['id']}` [{stat.upper()}]{err}",
                f"**Question:** {c['question']}\n",
                f"**Reference Answer:** {c['reference_answer']}\n",
                f"**Agent Response:** {c['agent_response'] or '[None]'}\n",
                f"- **Correctness:** {fmt_score(m['correctness']['score'])} — *{m['correctness']['reasoning']}*",
                f"- **Answer Relevance:** {fmt_score(m['answer_relevance']['score'])} — *{m['answer_relevance']['reasoning']}*",
                f"- **Groundedness:** {fmt_score(m['groundedness']['score'])} — *{m['groundedness']['reasoning']}*",
                f"- **Retrieval Relevance:** {fmt_score(m['retrieval_relevance']['score'])} — *{m['retrieval_relevance']['reasoning']}*",
                f"- **Case Average:** {fmt_score(m.get('average_score'))}\n",
                "---",
            ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Saved evaluation reports to {json_path} and {md_path}.")


async def main() -> None:
    evaluator = RagEvaluator()
    await evaluator.run()


if __name__ == "__main__":
    asyncio.run(main())
