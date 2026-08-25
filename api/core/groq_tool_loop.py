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
    log_tool_result,
)


# Maximum total chars across all messages in an active tool-loop before older
# tool results are trimmed.  Reduced from 24000 → 18000 to save ~25% on
# mid-loop payload sizes across all three executor providers (Groq, Cerebras,
# and the Gemini fallback which shares the same message list on hand-off).
GROQ_LOOP_CHAR_BUDGET = 18000


def shrink_tool_history_if_needed(messages: list):
    total = sum(len(m.get("content") or "") for m in messages)
    if total <= GROQ_LOOP_CHAR_BUDGET:
        return

    for m in messages:
        if total <= GROQ_LOOP_CHAR_BUDGET:
            break
        if m.get("role") == "tool" and len(m.get("content") or "") > 200:
            original_len = len(m["content"])
            total -= original_len
            m["content"] = (
                f"[Earlier tool output from this turn trimmed to stay within budget "
                f"- was {original_len} chars. Call the tool again if you need it.]"
            )
            total += len(m["content"])


_ERROR_CODE_RE = re.compile(r"Error code:\s*(\d{3})")
_RETRY_AFTER_RE = re.compile(r"try again in (?:([0-9]+)m)?([0-9.]+)s")
_TPD_RE = re.compile(r"tokens per day \(TPD\)")


def _groq_search_region(raw_err: str) -> str:
    return raw_err.split("failed_generation", 1)[0]


def _format_wait_time(raw_err: str) -> str:
    match = _RETRY_AFTER_RE.search(raw_err)
    if not match:
        return "a short while"
    minutes_part, seconds_part = match.groups()
    minutes = int(minutes_part) if minutes_part else 0
    seconds = int(float(seconds_part))
    return f"about {minutes}m {seconds}s" if minutes else f"about {seconds}s"


def is_groq_quota_or_rate_limit(raw_err: str) -> bool:
    status_match = _ERROR_CODE_RE.search(raw_err)
    search_region = _groq_search_region(raw_err)
    return (
        status_match is not None
        and status_match.group(1) == "429"
        and "rate_limit_exceeded" in search_region
    )


def explain_groq_error(raw_err: str) -> str:
    status_match = _ERROR_CODE_RE.search(raw_err)
    status = status_match.group(1) if status_match else None
    search_region = _groq_search_region(raw_err)

    if "tool_use_failed" in search_region:
        return (
            "The model generated an invalid tool call - this usually happens when it "
            "tries to write or rewrite a very large file in a single step and the "
            "output gets malformed along the way. Try asking for a smaller, more "
            "targeted change instead of a full-file rewrite."
        )

    if status == "429" and "rate_limit_exceeded" in search_region:
        wait_str = _format_wait_time(raw_err)
        if _TPD_RE.search(raw_err):
            return (
                f"Today's Groq token quota is nearly used up from everything run in this "
                f"session so far - not from this request being too big on its own. Groq "
                f"says it should free up in {wait_str}. You can wait it out, or upgrade to "
                f"Dev Tier at the link Groq provided if you need to keep working sooner."
            )
        return f"Groq's rate limit was hit. It should free up in {wait_str}."

    if status == "413" or "context_length_exceeded" in search_region:
        return (
            "The conversation for this task grew too large for the model's "
            "per-request limit - usually caused by several broad searches "
            "or large file reads happening back to back in one turn."
        )

    if "timeout" in raw_err.lower():
        return "The request to the model timed out."

    return "The underlying model API returned an error."


def run_groq_tool_loop(
    provider: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    log_callback: LogCallback,
    execute_tool: ToolExecutor,
    command_approval_callback: CommandApprovalCallback,
    cancel_event: Optional[threading.Event] = None,
) -> ToolLoopResult:
    max_turns = 10
    for _ in range(max_turns):
        if is_cancelled(cancel_event):
            return ToolLoopResult(final_answer="", messages=messages)

        shrink_tool_history_if_needed(messages)

        try:
            response_msg = provider.client.chat.completions.create(
                model=provider.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            ).choices[0].message
        except Exception as e:
            if is_cancelled(cancel_event):
                return ToolLoopResult(final_answer="", messages=messages)

            raw_err = str(e)
            log_callback("system", f"❌ Groq API Error: {raw_err}")
            return ToolLoopResult(provider_error=raw_err, messages=messages)

        messages.append(response_msg.model_dump(exclude_none=True))

        if response_msg.tool_calls:
            for tool_call in response_msg.tool_calls:
                if is_cancelled(cancel_event):
                    return ToolLoopResult(final_answer="", messages=messages)

                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except Exception:
                    fn_args = {}

                tool_result = execute_tool(fn_name, fn_args, command_approval_callback)
                log_tool_result(log_callback, fn_name, fn_args, tool_result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })
        else:
            final_answer = response_msg.content or "Task completed successfully."
            if is_cancelled(cancel_event):
                return ToolLoopResult(final_answer="", messages=messages)
            log_callback("ai", final_answer)
            return ToolLoopResult(final_answer=final_answer, messages=messages)

    return ToolLoopResult(
        final_answer="Agent reached maximum tool turns without concluding.",
        messages=messages,
    )
