import re
from typing import Optional
from fastapi import HTTPException, status

PROHIBITED_INSTRUCTION_PATTERNS = [
    # Jailbreak / prompt injection / system prompt override
    r"\b(ignore|disregard|forget|overwrite|replace|cancel)\s+(all\s+)?(previous|prior|system|above|base)\s+(instructions|prompts|rules|directives|constraints)\b",
    r"\boverride\s+(the\s+)?(system(\s+prompt)?|safety|security|base|core)\s*(prompt|rules|constraints|instructions|directives)?\b",
    r"\b(ignore|disregard|forget|overwrite|replace)\s+(the\s+)?(system\s+prompt|system-prompt|system\s+rules|system\s+instructions)\b",
    r"\b(do\s+not|never)\s+follow\s+(the\s+)?(system(\s+prompt)?|safety|security|base|core)\s*(prompt|rules|instructions)?\b",
    r"\bbypass\s+(system|safety|security|filters|guardrails|constraints|rules)\b",
    r"\bdisable\s+(safety|tools|guardrails|filters|restrictions|security)\b",
    r"\byour\s+new\s+(system\s+prompt|instructions|rules)\s+(is|are)\b",
    r"\b(system\s+prompt|system-prompt|system\s+instructions?)\s*:\b",
    r"\byou\s+are\s+now\s+(in\s+)?(dan|developer\s+mode|jailbreak|unrestricted|god\s*mode)\b",
    r"\b(dan\s+mode|jailbreak\s+mode|developer\s+mode\s+enabled)\b",
    r"\bact\s+as\s+(an\s+)?(unrestricted|jailbroken|evil)\b",
    r"\byou\s+are\s+no\s+longer\s+(an?\s+)?(ai|assistant|bound)\b",
    r"\bpretend\s+(to\s+be\s+unrestricted|you\s+have\s+no\s+(rules|restrictions|limits|safety))\b",
    r"\bdo\s+anything\s+now\b",
    # Prompt / credential exfiltration
    r"\breveal\s+(your\s+)?(system|internal|hidden)\s+(prompt|instructions|rules|text)\b",
    r"\bprint\s+(your\s+)?(system\s+prompt|system-prompt|system\s+instructions)\b",
    r"\bshow\s+(me\s+)?(your\s+)?(system|internal)\s+(prompt|instructions)\b",
    r"\bcat\s+(system-prompt|\.env|secrets)\b",
    r"\b(dump|exfiltrate|leak)\s+(database|db|secrets|passwords|api[_-]?keys|tokens)\b",
    # Destructive operations
    r"\brm\s+-rf\b",
    r"\bdrop\s+(table|database)\b",
    r"\bdelete\s+from\s+[a-z0-9_]+\b",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PROHIBITED_INSTRUCTION_PATTERNS]


def validate_user_instructions(instructions: Optional[str]) -> Optional[str]:
    """
    Validates user-provided instructions to prevent prompt injections, jailbreaks,
    system prompt overrides, and malicious directives.
    Raises HTTPException(422) if suspicious or prohibited directives are detected.
    """
    if not instructions or not instructions.strip():
        return None

    cleaned = instructions.strip()
    normalized = " ".join(cleaned.split())

    for pattern in COMPILED_PATTERNS:
        if pattern.search(normalized):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=(
                    "Instruction rejected: Potential prompt injection or system override detected. "
                    "User instructions cannot override the system prompt, core capabilities, or security directives."
                ),
            )

    return cleaned
