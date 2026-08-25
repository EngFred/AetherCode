import json
import re
import threading
from typing import Any, Dict, List, Optional

from core.tool_loop import (
    CommandApprovalCallback,
    LogCallback,
    ToolExecutor,
    ToolLoopResult,
    is_cancelled,
)

# Cerebras uses the exact same OpenAI-compatible wire protocol as Groq, so
# we re-use run_groq_tool_loop directly.  The separate module exists purely
# to give Cerebras its own error-classification and explanation layer,
# keeping the agent code clean and the two providers independently testable.
from core.groq_tool_loop import run_groq_tool_loop as _run_openai_compat_tool_loop


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_ERROR_CODE_RE = re.compile(r"Error code:\s*(\d{3})")
_RETRY_AFTER_RE = re.compile(r"try again in (?:([0-9]+)m)?([0-9.]+)s")
_TPD_RE = re.compile(r"tokens per day|daily.{0,20}limit|daily.{0,20}quota", re.IGNORECASE)


def _format_wait_time(raw_err: str) -> str:
    match = _RETRY_AFTER_RE.search(raw_err)
    if not match:
        return "a short while"
    minutes_part, seconds_part = match.groups()
    minutes = int(minutes_part) if minutes_part else 0
    seconds = int(float(seconds_part))
    return f"about {minutes}m {seconds}s" if minutes else f"about {seconds}s"


def is_cerebras_quota_or_rate_limit(raw_err: str) -> bool:
    """
    Returns True when the Cerebras error is a 429 rate-limit / quota error.
    Cerebras follows the same HTTP 429 convention as Groq/OpenAI.
    """
    lowered = raw_err.lower()
    status_match = _ERROR_CODE_RE.search(raw_err)
    has_429 = (status_match is not None and status_match.group(1) == "429") or \
              "429" in raw_err
    is_limit = any(kw in lowered for kw in (
        "rate_limit_exceeded", "rate limit", "quota", "too many requests",
    ))
    return has_429 and is_limit


def explain_cerebras_error(raw_err: str) -> str:
    """Returns a friendly, user-facing explanation of a Cerebras API error."""
    status_match = _ERROR_CODE_RE.search(raw_err)
    status = status_match.group(1) if status_match else None
    lowered = raw_err.lower()

    if status == "429" or is_cerebras_quota_or_rate_limit(raw_err):
        wait_str = _format_wait_time(raw_err)
        if _TPD_RE.search(raw_err):
            return (
                f"Today's Cerebras token quota has been used up. "
                f"It resets daily — try again in {wait_str}, or tomorrow if the "
                f"daily cap has been fully reached."
            )
        return f"Cerebras' rate limit was hit. It should free up in {wait_str}."

    if status == "413" or "context_length_exceeded" in lowered:
        return (
            "The conversation grew too large for Cerebras' per-request limit. "
            "Try narrowing the request to a specific file or smaller change."
        )

    if "timeout" in lowered:
        return "The request to Cerebras timed out."

    return "Cerebras returned an error while processing the task."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_cerebras_tool_loop(
    provider: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    log_callback: LogCallback,
    execute_tool: ToolExecutor,
    command_approval_callback: CommandApprovalCallback,
    cancel_event: Optional[threading.Event] = None,
) -> ToolLoopResult:
    """
    Runs a full tool-calling loop against the Cerebras API.

    Internally delegates to run_groq_tool_loop because the two APIs are
    wire-identical (both OpenAI-compatible).  The provider object passed in
    must expose `.client` (an openai.OpenAI instance) and `.model_name`.
    """
    return _run_openai_compat_tool_loop(
        provider=provider,
        messages=messages,
        tools=tools,
        log_callback=log_callback,
        execute_tool=execute_tool,
        command_approval_callback=command_approval_callback,
        cancel_event=cancel_event,
    )
