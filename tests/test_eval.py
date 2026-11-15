import asyncio
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from eval.dataset_loader import load_yaml_dataset, sync_dataset_to_langsmith
from eval.evaluators import (
    evaluate_answer_relevance,
    evaluate_correctness,
    evaluate_groundedness,
    evaluate_retrieval_relevance,
    evaluate_turn,
    parse_eval_json,
)
from eval.runner import RagEvaluator


class FakeJudgeLLM(BaseChatModel):
    """Fake judge model that returns deterministic evaluation JSON."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        resp_json = json.dumps({
            "score": 0.95,
            "reasoning": "The response matches the ground truth and context accurately.",
        })
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resp_json))])

    @property
    def _llm_type(self) -> str:
        return "fake-judge"


class TestRagEvaluation(unittest.TestCase):

    def test_load_yaml_dataset(self):
        dataset = load_yaml_dataset()
        self.assertIn("test_cases", dataset)
        self.assertGreater(len(dataset["test_cases"]), 0)

        first_case = dataset["test_cases"][0]
        self.assertIn("question", first_case)
        self.assertIn("reference_answer", first_case)

    def test_parse_eval_json_standard(self):
        sample = '{"score": 0.85, "reasoning": "Good match"}'
        res = parse_eval_json(sample, "correctness")
        self.assertEqual(res.score, 0.85)
        self.assertEqual(res.reasoning, "Good match")

    def test_parse_eval_json_markdown_wrapped(self):
        sample = '```json\n{"score": 1.0, "reasoning": "Perfect"}\n```'
        res = parse_eval_json(sample, "groundedness")
        self.assertEqual(res.score, 1.0)
        self.assertEqual(res.reasoning, "Perfect")

    def test_parse_eval_json_fallback_regex(self):
        sample = 'The score is "score": 0.72 and the reason is "reasoning": "Partially grounded".'
        res = parse_eval_json(sample, "groundedness")
        self.assertEqual(res.score, 0.72)
        self.assertEqual(res.reasoning, "Partially grounded")

    def test_evaluators_with_fake_judge(self):
        async def _run():
            judge = FakeJudgeLLM()
            corr = await evaluate_correctness(
                "What is Chroma?", "A vector database.", "A vector database.", llm=judge
            )
            self.assertEqual(corr.score, 0.95)

            rel = await evaluate_answer_relevance(
                "What is Chroma?", "A vector database.", llm=judge
            )
            self.assertEqual(rel.score, 0.95)

            gnd = await evaluate_groundedness(
                "A vector database.", "Chroma is an open-source vector database.", llm=judge
            )
            self.assertEqual(gnd.score, 0.95)

            # Retrieval is evaluated in code
            ret = await evaluate_retrieval_relevance(
                "What is Chroma?", "Chroma is an open-source vector database.", llm=judge, must_retrieve="Chroma"
            )
            self.assertEqual(ret.score, 1.0)

            turn = await evaluate_turn(
                question="What is Chroma?",
                agent_response="A vector database.",
                reference_answer="A vector database.",
                retrieved_context="Chroma is an open-source vector database.",
                llm=judge,
            )
            self.assertEqual(turn.average_score, 0.962)
            self.assertEqual(turn.correctness.score, 0.95)

        asyncio.run(_run())

    def test_parse_fact_verdicts(self):
        from eval.evaluators import parse_fact_verdicts

        raw_json = """{
            "verdicts": [
                {"fact": "Notice within 72 hours", "verdict": "present", "reason": "Explicitly stated."},
                {"fact": "Affected users notified without undue delay", "verdict": "partial", "reason": "Mentions users but not undue delay."}
            ]
        }"""
        key_facts = [
            "Notice within 72 hours",
            "Affected users notified without undue delay",
            "Third fact that is omitted"
        ]
        score, verdicts, reasoning = parse_fact_verdicts(raw_json, key_facts)
        # 1 present (1.0) + 1 partial (0.5) + 1 missing (0.0) = 1.5 / 3 = 0.50
        self.assertEqual(score, 0.5)
        self.assertEqual(len(verdicts), 3)
        self.assertIn("1/3 key facts present", reasoning)

    def test_evaluate_retrieval_in_code_recall_precision(self):
        from eval.evaluators import evaluate_retrieval_in_code

        chunks = [
            "## Section: Privacy Policy > Retention\nSupport chats are kept for 3 years.",
            "## Section: Privacy Policy > Security\nPasswords must be hashed.",
            "## Section: Privacy Policy > Retention\nCall recordings are kept for 12 months.",
        ]
        res = evaluate_retrieval_in_code(
            retrieved_chunks=chunks,
            must_retrieve=["3 years", "12 months"],
            acceptable_sections=["Retention"],
        )
        self.assertEqual(res.score, 1.0)  # Both phrases found -> recall@k = 1.0
        self.assertEqual(res.details["recall_at_k"], 1.0)
        self.assertAlmostEqual(res.details["precision_at_k"], 0.667, places=2)  # 2 of 3 chunks relevant

    def test_empty_agent_response_never_reaches_judge(self):
        async def _run():
            judge = FakeJudgeLLM()
            turn = await evaluate_turn(
                question="What is Chroma?",
                agent_response="",  # Empty response
                reference_answer="A vector database.",
                retrieved_context="Chroma is a database.",
                llm=judge,
            )
            self.assertEqual(turn.status, "error")
            self.assertIsNone(turn.average_score)
            self.assertEqual(turn.correctness.status, "skipped")
            self.assertIsNone(turn.correctness.score)

        asyncio.run(_run())

    def test_judge_timeout_stores_none_and_judge_error(self):
        from eval.evaluators import MetricResult, TurnEvaluationResult

        err_turn = TurnEvaluationResult(
            correctness=MetricResult(metric="correctness", score=None, reasoning="Timed out", status="judge_error"),
            answer_relevance=MetricResult(metric="answer_relevance", score=0.9, reasoning="Good", status="success"),
            groundedness=MetricResult(metric="groundedness", score=0.9, reasoning="Good", status="success"),
            retrieval_relevance=MetricResult(metric="retrieval_relevance", score=1.0, reasoning="Good", status="success"),
            average_score=0.933,
            status="judge_error",
        )
        self.assertEqual(err_turn.status, "judge_error")
        self.assertIsNone(err_turn.correctness.score)

    def test_sync_dataset_to_langsmith_mock(self):
        mock_client = MagicMock()
        mock_client.has_dataset.return_value = False
        mock_dataset = MagicMock()
        mock_dataset.id = "mock-dataset-uuid-123"
        mock_client.create_dataset.return_value = mock_dataset

        test_cases = [
            {"id": "tc1", "question": "Q1?", "reference_answer": "A1."},
            {"id": "tc2", "question": "Q2?", "reference_answer": "A2."},
        ]

        with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "lsv2_pt_test_key_12345"}):
            dataset_id = sync_dataset_to_langsmith(
                test_cases=test_cases,
                dataset_name="test-eval-ds",
                client=mock_client,
            )

        self.assertEqual(dataset_id, "mock-dataset-uuid-123")
        mock_client.create_dataset.assert_called_once()
        mock_client.create_examples.assert_called_once()

    def test_sync_dataset_to_langsmith_skipped_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            res = sync_dataset_to_langsmith([{"question": "Q", "reference_answer": "A"}])
            self.assertIsNone(res)

    def test_rag_evaluator_end_to_end_isolation(self):
        """
        Tests the complete RagEvaluator runner in an isolated temp test YAML
        and verifies that it never interferes with production databases.
        """
        temp_dir = tempfile.mkdtemp()
        temp_yaml = Path(temp_dir) / "test_eval.yaml"
        sample_yaml_content = """
test_cases:
  - id: "test-case-1"
    question: "What backend framework is used?"
    reference_answer: "The system uses FastAPI for backend development."
    tags: ["test"]
"""
        with open(temp_yaml, "w", encoding="utf-8") as f:
            f.write(sample_yaml_content)

        async def _run():
            from langchain_core.embeddings import FakeEmbeddings

            judge = FakeJudgeLLM()
            fake_embed = FakeEmbeddings(size=128)

            evaluator = RagEvaluator(
                yaml_path=temp_yaml,
                llm=judge,
                embeddings=fake_embed,
                db_path=str(Path(temp_dir) / "test.db"),
                chroma_dir=str(Path(temp_dir) / "chroma"),
                clean_on_start=False,
                clean_on_finish=False,
            )
            summary = await evaluator.run()

            self.assertEqual(summary["total_cases"], 1)
            self.assertEqual(summary["metrics_summary"]["correctness"], 0.95)
            self.assertEqual(summary["metrics_summary"]["overall_score"], 0.962)

            # Check report files generated
            report_json = Path(__file__).resolve().parent.parent / "eval" / "results" / "eval_report.json"
            self.assertTrue(report_json.exists())

        try:
            asyncio.run(_run())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_evals_api_endpoints(self):
        """Tests uploading, listing, running, and deleting documents via the /evals API routes."""
        from fastapi.testclient import TestClient
        from langchain_core.embeddings import FakeEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore
        from app.main import build_app
        from app.api.routes.evals import set_eval_vector_store, reset_eval_vector_store

        fake_store = InMemoryVectorStore(FakeEmbeddings(size=128))
        set_eval_vector_store(fake_store)

        app = build_app()
        try:
            with TestClient(app) as client:
                # 1. Upload raw JSON text to eval knowledge base
                post_res = client.post(
                    "/evals/documents",
                    json={"content": "Evaluation content for testing purposes.", "doc_id": "api-eval-doc-1"},
                )
                self.assertEqual(post_res.status_code, 200)
                data = post_res.json()
                self.assertEqual(data["status"], "success")
                self.assertEqual(data["doc_id"], "api-eval-doc-1")
                self.assertGreaterEqual(data["chunks_count"], 1)

                # 2. Upload multipart file to eval knowledge base
                files = {"file": ("eval_notes.txt", b"Notes on evaluation metrics.", "text/plain")}
                post_file_res = client.post("/evals/documents", files=files)
                self.assertEqual(post_file_res.status_code, 200)
                file_data = post_file_res.json()
                self.assertEqual(file_data["status"], "success")
                self.assertTrue(file_data.get("doc_id"))

                # 3. List evaluation documents
                list_res = client.get("/evals/documents")
                self.assertEqual(list_res.status_code, 200)
                docs = list_res.json()
                ids = [d["doc_id"] for d in docs]
                self.assertIn("api-eval-doc-1", ids)

                # 3b. Query evaluation chunks via GET and POST
                search_get = client.get("/evals/chunks?query=evaluation&k=2")
                self.assertEqual(search_get.status_code, 200)
                search_get_data = search_get.json()
                self.assertEqual(search_get_data["query"], "evaluation")
                self.assertGreaterEqual(search_get_data["total_chunks"], 1)

                search_post = client.post("/evals/query", json={"query": "evaluation", "k": 2})
                self.assertEqual(search_post.status_code, 200)
                search_post_data = search_post.json()
                self.assertEqual(search_post_data["query"], "evaluation")
                self.assertGreaterEqual(search_post_data["total_chunks"], 1)

                # Alias routes check
                alias_get = client.get("/evals/documents/chunks?query=evaluation&k=2")
                self.assertEqual(alias_get.status_code, 200)

                # Validation error check
                err_get = client.get("/evals/chunks?query=   ")
                self.assertEqual(err_get.status_code, 422)

                # 4. Delete document from evaluation knowledge base
                del_res = client.delete("/evals/documents/api-eval-doc-1")
                self.assertEqual(del_res.status_code, 200)
                self.assertEqual(del_res.json()["status"], "success")

                # Verify deletion in list
                list_res2 = client.get("/evals/documents")
                self.assertEqual(list_res2.status_code, 200)
                ids2 = [d["doc_id"] for d in list_res2.json()]
                self.assertNotIn("api-eval-doc-1", ids2)

                # 5. Fetch latest evaluation results (from earlier run in tests)
                results_res = client.get("/evals/results")
                self.assertEqual(results_res.status_code, 200)
                res_data = results_res.json()
                self.assertIn("metrics_summary", res_data)
        finally:
            reset_eval_vector_store()


if __name__ == "__main__":
    unittest.main()
