import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.agent.llm import get_llm
from eval.config import get_judge_llm

logger = logging.getLogger("evaluators")


class MetricResult(BaseModel):
    metric: str
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reasoning: str
    status: str = "success"  # "success" | "judge_error" | "skipped" | "error"
    details: Optional[Dict[str, Any]] = None


class TurnEvaluationResult(BaseModel):
    correctness: MetricResult
    answer_relevance: MetricResult
    groundedness: MetricResult
    retrieval_relevance: MetricResult
    average_score: Optional[float] = None
    status: str = "success"  # "success" | "judge_error" | "error" | "partial_error"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correctness": self.correctness.model_dump(),
            "answer_relevance": self.answer_relevance.model_dump(),
            "groundedness": self.groundedness.model_dump(),
            "retrieval_relevance": self.retrieval_relevance.model_dump(),
            "average_score": self.average_score,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Judge Prompts (Each metric is strictly focused on its own evaluation criteria)
# ---------------------------------------------------------------------------

FACT_VERDICT_SYSTEM_PROMPT = """You are a strict, objective evaluation judge. Grade whether each Key Fact is stated in the Agent Response.
For each fact in the provided list, determine one of three verdicts:
- "present": The fact is explicitly and accurately stated in the Agent Response (full credit).
- "partial": The fact is partially mentioned, but misses key qualifications, numbers, or details (half credit).
- "missing": The fact is omitted, incorrect, or contradicted in the Agent Response (no credit).

Respond strictly with a JSON object in this exact schema:
{
  "verdicts": [
    {"fact": "<exact key fact>", "verdict": "present"|"partial"|"missing", "reason": "<1 concise sentence>"}
  ]
}
"""

CORRECTNESS_SYSTEM_PROMPT = """You are a strict evaluation judge grading the CORRECTNESS of an AI assistant's response against a ground truth reference answer.
Compare the Agent Response to the Reference Answer:
1.0: Factually accurate, covers all critical claims in the reference answer.
0.5: Partially correct, but contains notable omissions or slight inaccuracies.
0.0: Factually incorrect or contradicts the reference answer.

Respond strictly with a JSON object in this exact schema:
{"score": <float between 0.0 and 1.0>, "reasoning": "<1-2 concise sentences explaining factual accuracy>"}
"""

ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are an evaluation judge grading the ANSWER RELEVANCE of an AI assistant's response to the user's question.
Focus ONLY on this question: Does the Agent Response directly and completely address what the User asked in the Question?
Do NOT evaluate factual truthfulness against external facts or context here. Only grade whether the response addresses the prompt directly without evasion or irrelevant filler.
1.0: Directly, completely, and clearly answers what was asked.
0.5: Partially addresses the question or drifts into tangential topics.
0.0: Irrelevant, completely off-topic, evasive, or fails to address the question.

Respond strictly with a JSON object in this exact schema:
{"score": <float between 0.0 and 1.0>, "reasoning": "<1-2 concise sentences explaining relevance>"}
"""

GROUNDEDNESS_SYSTEM_PROMPT = """You are a strict evaluation judge grading the GROUNDEDNESS (faithfulness) of an AI assistant's response based on the retrieved context.
Focus ONLY on this question: Is every factual claim made in the Agent Response directly supported by the Retrieved Context?
1.0: Every factual claim is strictly supported by and traceable to the retrieved context (no hallucinations). If no context was retrieved and the assistant correctly states it lacks the information, score 1.0.
0.5: The main claims are supported, but the response includes minor unsupported extrapolations or unverified claims.
0.0: Major hallucinations, fabrications, or claims contradicting the retrieved context.

Respond strictly with a JSON object in this exact schema:
{"score": <float between 0.0 and 1.0>, "reasoning": "<1-2 concise sentences identifying any unsupported claims or confirming full grounding>"}
"""


# ---------------------------------------------------------------------------
# Helper: Judge Invocation with Retries
# ---------------------------------------------------------------------------

async def invoke_judge_with_retry(
    judge: BaseChatModel,
    messages: List[Any],
    timeout: float = 120.0,
    retries: int = 2,
    metric_name: str = "metric",
) -> tuple[Optional[str], Optional[str]]:
    """
    Invokes the judge LLM with retries on timeout or errors.
    Returns (raw_content, error_message). If all retries fail, returns (None, error_msg).
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            reply = await asyncio.wait_for(
                judge.ainvoke(messages),
                timeout=timeout,
            )
            content = reply.content if hasattr(reply, "content") else str(reply)
            return content, None
        except asyncio.TimeoutError:
            last_err = f"Timed out after {timeout}s (attempt {attempt + 1}/{retries + 1})"
            logger.warning(f"Judge call for '{metric_name}' {last_err}")
            if attempt < retries:
                await asyncio.sleep(1.0)
        except Exception as e:
            last_err = f"Failed with error: {e} (attempt {attempt + 1}/{retries + 1})"
            logger.warning(f"Judge call for '{metric_name}' {last_err}")
            if attempt < retries:
                await asyncio.sleep(1.0)

    return None, last_err


# ---------------------------------------------------------------------------
# JSON Parsers
# ---------------------------------------------------------------------------

def parse_eval_json(raw_text: str, default_metric: str) -> MetricResult:
    """Safely extracts JSON with 'score' and 'reasoning' from LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
        score = float(data.get("score", 0.0))
        reasoning = str(data.get("reasoning", "No explanation provided."))
        return MetricResult(
            metric=default_metric,
            score=max(0.0, min(1.0, score)),
            reasoning=reasoning,
            status="success",
        )
    except Exception:
        # Fallback regex extraction
        score_match = re.search(r'["\']?score["\']?\s*:\s*([0-9.]+)', cleaned)
        reasoning_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\']([^"\']+)["\']', cleaned)

        score = float(score_match.group(1)) if score_match else 0.5
        reasoning = reasoning_match.group(1) if reasoning_match else cleaned[:300]
        return MetricResult(
            metric=default_metric,
            score=max(0.0, min(1.0, score)),
            reasoning=reasoning,
            status="success",
        )


def parse_fact_verdicts(
    raw_text: str,
    key_facts: List[str],
) -> tuple[float, List[Dict[str, Any]], str]:
    """
    Parses per-fact verdicts from the judge output and deterministically computes
    the correctness score in code: present = 1.0, partial = 0.5, missing = 0.0.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        m = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
            except Exception:
                data = {}
        else:
            data = {}

    if isinstance(data, dict):
        verdicts = data.get("verdicts", [])
    elif isinstance(data, list):
        verdicts = data
    else:
        verdicts = []

    val_map = {"present": 1.0, "partial": 0.5, "missing": 0.0}
    verdict_by_fact: Dict[str, tuple[str, str]] = {}
    for v in verdicts:
        if isinstance(v, dict):
            f = str(v.get("fact", "")).strip().lower()
            verd = str(v.get("verdict", "missing")).strip().lower()
            reason = str(v.get("reason", ""))
            verdict_by_fact[f] = (verd if verd in val_map else "missing", reason)

    scored_verdicts: List[Dict[str, Any]] = []
    total_val = 0.0

    for fact in key_facts:
        f_norm = fact.strip().lower()
        match_verd = "missing"
        match_reason = "Fact not addressed in verdict"

        # Match exact or substring in fact keys
        for vf, (vd, r) in verdict_by_fact.items():
            if vf in f_norm or f_norm in vf or any(w in vf for w in f_norm.split()[:3]):
                match_verd = vd
                match_reason = r
                break

        pts = val_map.get(match_verd, 0.0)
        total_val += pts
        scored_verdicts.append({
            "fact": fact,
            "verdict": match_verd,
            "score": pts,
            "reason": match_reason,
        })

    score = round(total_val / len(key_facts), 3) if key_facts else 0.0
    present_cnt = sum(1 for v in scored_verdicts if v["verdict"] == "present")
    partial_cnt = sum(1 for v in scored_verdicts if v["verdict"] == "partial")
    missing_cnt = sum(1 for v in scored_verdicts if v["verdict"] == "missing")
    reasoning = (
        f"{present_cnt}/{len(key_facts)} key facts present, "
        f"{partial_cnt} partial, {missing_cnt} missing."
    )
    return score, scored_verdicts, reasoning


# ---------------------------------------------------------------------------
# Metric Evaluator 1: Deterministic Retrieval in Code (Recall@k & Precision@k)
# ---------------------------------------------------------------------------

def evaluate_retrieval_in_code(
    retrieved_chunks: List[Any],
    must_retrieve: Optional[List[str] | str] = None,
    acceptable_sections: Optional[List[str]] = None,
) -> MetricResult:
    """
    Evaluates retrieval relevance strictly in code without an LLM.
    - Computes recall@k: whether key phrases from the expected passage appear in retrieved chunks.
    - Computes precision@k: fraction of retrieved chunks belonging to acceptable sections.
    """
    chunk_texts = []
    chunk_sections = []
    for c in retrieved_chunks:
        if hasattr(c, "page_content"):
            chunk_texts.append(c.page_content)
            chunk_sections.append(str(getattr(c, "metadata", {}).get("section", "")))
        elif isinstance(c, dict):
            chunk_texts.append(str(c.get("content", c.get("text", ""))))
            chunk_sections.append(str(c.get("section", "")))
        else:
            chunk_texts.append(str(c))
            chunk_sections.append("")

    combined_text = "\n\n".join(chunk_texts).lower()

    if isinstance(must_retrieve, str):
        phrases = [must_retrieve] if must_retrieve.strip() else []
    elif isinstance(must_retrieve, list):
        phrases = [p for p in must_retrieve if str(p).strip()]
    else:
        phrases = []

    if phrases:
        found = [p for p in phrases if p.lower() in combined_text]
        missing = [p for p in phrases if p.lower() not in combined_text]
        recall_at_k = round(len(found) / len(phrases), 3)
    else:
        found = []
        missing = []
        # If no must_retrieve specified (e.g. unanswerable / negative test case)
        recall_at_k = 1.0

    total_chunks = len(chunk_texts)
    sections_list = [s.strip().lower() for s in (acceptable_sections or []) if s.strip()]
    if sections_list and total_chunks > 0:
        relevant_chunks_count = 0
        for text, sec in zip(chunk_texts, chunk_sections):
            text_lower = text.lower()
            sec_lower = sec.lower()
            if any(s in sec_lower or s in text_lower for s in sections_list):
                relevant_chunks_count += 1
        precision_at_k = round(relevant_chunks_count / total_chunks, 3)
    else:
        precision_at_k = 1.0 if total_chunks > 0 else 0.0

    score = recall_at_k
    found_count = len(found)
    total_phrases = len(phrases)
    rel_chunks = int(round(precision_at_k * total_chunks))

    reasoning = (
        f"recall@k={recall_at_k} ({found_count}/{total_phrases} key phrases found), "
        f"precision@k={precision_at_k} ({rel_chunks}/{total_chunks} chunks from relevant sections)"
    )

    return MetricResult(
        metric="retrieval_relevance",
        score=score,
        reasoning=reasoning,
        status="success",
        details={
            "recall_at_k": recall_at_k,
            "precision_at_k": precision_at_k,
            "must_retrieve_phrases": phrases,
            "found_phrases": found,
            "missing_phrases": missing,
            "acceptable_sections": sections_list,
            "total_chunks_retrieved": total_chunks,
        },
    )


async def evaluate_retrieval_relevance(
    question: str,
    retrieved_context: str,
    llm: Optional[BaseChatModel] = None,
    timeout: float = 120.0,
    **kwargs,
) -> MetricResult:
    """Async wrapper for retrieval relevance evaluation (evaluated deterministically in code)."""
    return evaluate_retrieval_in_code(
        retrieved_chunks=[retrieved_context],
        must_retrieve=kwargs.get("must_retrieve"),
        acceptable_sections=kwargs.get("acceptable_sections"),
    )


# ---------------------------------------------------------------------------
# Metric Evaluator 2: Correctness (Fact-Based Verdicts or Reference Comparison)
# ---------------------------------------------------------------------------

async def evaluate_correctness(
    question: str,
    agent_response: str,
    reference_answer: str,
    key_facts: Optional[List[str]] = None,
    llm: Optional[BaseChatModel] = None,
    timeout: float = 120.0,
    retries: int = 2,
) -> MetricResult:
    """
    Evaluates factual correctness.
    If key_facts is provided, the judge outputs per-fact verdicts and the score
    is computed deterministically in code.
    If key_facts is not provided, grades factual alignment with the reference answer.
    """
    judge = llm or get_judge_llm(timeout=timeout)

    # Fact-based correctness
    if key_facts and len(key_facts) > 0:
        facts_list_str = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(key_facts))
        user_prompt = f"""[User Question]: {question}
[Agent Response]:
{agent_response}

[Key Facts to Check]:
{facts_list_str}
"""
        content, err = await invoke_judge_with_retry(
            judge,
            [
                SystemMessage(content=FACT_VERDICT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ],
            timeout=timeout,
            retries=retries,
            metric_name="correctness",
        )
        if content is None:
            return MetricResult(
                metric="correctness",
                score=None,
                reasoning=f"Judge failed after {retries + 1} attempts: {err}",
                status="judge_error",
            )

        score, verdicts, reasoning = parse_fact_verdicts(content, key_facts)
        return MetricResult(
            metric="correctness",
            score=score,
            reasoning=reasoning,
            status="success",
            details={"verdicts": verdicts},
        )

    # Fallback reference-based correctness
    user_prompt = f"""[Question]: {question}
[Reference Answer]: {reference_answer}
[Agent Response]: {agent_response}
"""
    content, err = await invoke_judge_with_retry(
        judge,
        [
            SystemMessage(content=CORRECTNESS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        timeout=timeout,
        retries=retries,
        metric_name="correctness",
    )
    if content is None:
        return MetricResult(
            metric="correctness",
            score=None,
            reasoning=f"Judge failed after {retries + 1} attempts: {err}",
            status="judge_error",
        )

    return parse_eval_json(content, "correctness")


# ---------------------------------------------------------------------------
# Metric Evaluator 3: Answer Relevance
# ---------------------------------------------------------------------------

async def evaluate_answer_relevance(
    question: str,
    agent_response: str,
    llm: Optional[BaseChatModel] = None,
    timeout: float = 120.0,
    retries: int = 2,
) -> MetricResult:
    """
    Evaluates whether the agent response directly and completely addresses the user question.
    """
    judge = llm or get_judge_llm(timeout=timeout)
    user_prompt = f"""[Question]: {question}
[Agent Response]: {agent_response}
"""
    content, err = await invoke_judge_with_retry(
        judge,
        [
            SystemMessage(content=ANSWER_RELEVANCE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        timeout=timeout,
        retries=retries,
        metric_name="answer_relevance",
    )
    if content is None:
        return MetricResult(
            metric="answer_relevance",
            score=None,
            reasoning=f"Judge failed after {retries + 1} attempts: {err}",
            status="judge_error",
        )

    return parse_eval_json(content, "answer_relevance")


# ---------------------------------------------------------------------------
# Metric Evaluator 4: Groundedness (Faithfulness)
# ---------------------------------------------------------------------------

async def evaluate_groundedness(
    agent_response: str,
    retrieved_context: str,
    llm: Optional[BaseChatModel] = None,
    timeout: float = 120.0,
    retries: int = 2,
) -> MetricResult:
    """
    Evaluates whether all factual claims in the response are supported by retrieved context.
    """
    judge = llm or get_judge_llm(timeout=timeout)
    context_str = retrieved_context.strip() if retrieved_context else "[No context retrieved]"
    if len(context_str) > 3000:
        context_str = context_str[:3000] + "... [context truncated]"

    trimmed_response = agent_response.strip()
    if len(trimmed_response) > 2000:
        trimmed_response = trimmed_response[:2000] + "... [response truncated]"

    user_prompt = f"""[Retrieved Context]:
{context_str}

[Agent Response]:
{trimmed_response}
"""
    content, err = await invoke_judge_with_retry(
        judge,
        [
            SystemMessage(content=GROUNDEDNESS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ],
        timeout=timeout,
        retries=retries,
        metric_name="groundedness",
    )
    if content is None:
        return MetricResult(
            metric="groundedness",
            score=None,
            reasoning=f"Judge failed after {retries + 1} attempts: {err}",
            status="judge_error",
        )

    return parse_eval_json(content, "groundedness")


# ---------------------------------------------------------------------------
# Modular Turn Evaluation
# ---------------------------------------------------------------------------

async def evaluate_turn(
    question: str,
    agent_response: str,
    reference_answer: str,
    retrieved_context: str,
    retrieved_chunks: Optional[List[Any]] = None,
    key_facts: Optional[List[str]] = None,
    must_retrieve: Optional[List[str] | str] = None,
    acceptable_sections: Optional[List[str]] = None,
    llm: Optional[BaseChatModel] = None,
    timeout: float = 120.0,
    retries: int = 2,
) -> TurnEvaluationResult:
    """
    Performs modular, independent evaluation across all 4 RAG metrics:
    1. Empty check: If response is empty, returns skipped status without calling judge.
    2. Retrieval relevance: Evaluated deterministically in code (recall@k & precision@k).
    3. Correctness: Fact-based verdict scoring in code, or focused reference grading.
    4. Answer relevance: Dedicated single-metric judge call.
    5. Groundedness: Dedicated single-metric judge call.
    """
    # 1. Guard against empty response: an empty answer must never reach the judge
    if not agent_response or not agent_response.strip():
        msg = "Empty agent response - skipped judge evaluation."
        return TurnEvaluationResult(
            correctness=MetricResult(metric="correctness", score=None, reasoning=msg, status="skipped"),
            answer_relevance=MetricResult(metric="answer_relevance", score=None, reasoning=msg, status="skipped"),
            groundedness=MetricResult(metric="groundedness", score=None, reasoning=msg, status="skipped"),
            retrieval_relevance=MetricResult(metric="retrieval_relevance", score=None, reasoning=msg, status="skipped"),
            average_score=None,
            status="error",
        )

    # 2. Retrieval Relevance (deterministic in code)
    chunks = retrieved_chunks if retrieved_chunks is not None else [retrieved_context]
    ret_metric = evaluate_retrieval_in_code(
        retrieved_chunks=chunks,
        must_retrieve=must_retrieve,
        acceptable_sections=acceptable_sections,
    )

    # 3. Correctness (fact-based or reference-based)
    corr_metric = await evaluate_correctness(
        question=question,
        agent_response=agent_response,
        reference_answer=reference_answer,
        key_facts=key_facts,
        llm=llm,
        timeout=timeout,
        retries=retries,
    )

    # 4. Answer Relevance
    rel_metric = await evaluate_answer_relevance(
        question=question,
        agent_response=agent_response,
        llm=llm,
        timeout=timeout,
        retries=retries,
    )

    # 5. Groundedness
    gnd_metric = await evaluate_groundedness(
        agent_response=agent_response,
        retrieved_context=retrieved_context,
        llm=llm,
        timeout=timeout,
        retries=retries,
    )

    # Compute average over valid non-None scores
    valid_scores = [
        m.score
        for m in [corr_metric, rel_metric, gnd_metric, ret_metric]
        if m.score is not None
    ]
    avg = round(sum(valid_scores) / len(valid_scores), 3) if valid_scores else None

    # Determine status and case average
    if any(m.status == "judge_error" for m in [corr_metric, rel_metric, gnd_metric, ret_metric]):
        turn_status = "judge_error"
        avg = None  # Do not calculate an incomplete or misleading average if a metric failed
    else:
        turn_status = "success"

    return TurnEvaluationResult(
        correctness=corr_metric,
        answer_relevance=rel_metric,
        groundedness=gnd_metric,
        retrieval_relevance=ret_metric,
        average_score=avg,
        status=turn_status,
    )


# ---------------------------------------------------------------------------
# LangSmith Evaluator Adapters
# ---------------------------------------------------------------------------

def create_langsmith_evaluators(llm: Optional[BaseChatModel] = None) -> List[Any]:
    """
    Creates LangSmith-compatible evaluator functions for use with
    langsmith.evaluation.evaluate(target, data=..., evaluators=[...]).
    """
    import asyncio

    def correctness_eval(run: Any, example: Any) -> Dict[str, Any]:
        question = example.inputs.get("question", "")
        reference = example.outputs.get("reference_answer", "")
        key_facts = example.outputs.get("key_facts", [])
        response = run.outputs.get("response", "") if run.outputs else ""
        res = asyncio.run(evaluate_correctness(question, response, reference, key_facts=key_facts, llm=llm))
        return {"key": "correctness", "score": res.score, "comment": res.reasoning}

    def relevance_eval(run: Any, example: Any) -> Dict[str, Any]:
        question = example.inputs.get("question", "")
        response = run.outputs.get("response", "") if run.outputs else ""
        res = asyncio.run(evaluate_answer_relevance(question, response, llm=llm))
        return {"key": "answer_relevance", "score": res.score, "comment": res.reasoning}

    def groundedness_eval(run: Any, example: Any) -> Dict[str, Any]:
        response = run.outputs.get("response", "") if run.outputs else ""
        context = run.outputs.get("retrieved_context", "") if run.outputs else ""
        res = asyncio.run(evaluate_groundedness(response, context, llm=llm))
        return {"key": "groundedness", "score": res.score, "comment": res.reasoning}

    def retrieval_eval(run: Any, example: Any) -> Dict[str, Any]:
        must_retrieve = example.outputs.get("must_retrieve")
        acceptable_sections = example.outputs.get("acceptable_sections")
        chunks = run.outputs.get("retrieved_chunks", [run.outputs.get("retrieved_context", "")]) if run.outputs else []
        res = evaluate_retrieval_in_code(chunks, must_retrieve=must_retrieve, acceptable_sections=acceptable_sections)
        return {"key": "retrieval_relevance", "score": res.score, "comment": res.reasoning}

    return [correctness_eval, relevance_eval, groundedness_eval, retrieval_eval]
