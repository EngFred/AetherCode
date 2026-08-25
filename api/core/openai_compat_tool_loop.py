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


OPENAI_COMPAT_LOOP_CHAR_BUDGET = 18000


def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strips provider-specific proprietary fields (such as Groq's 'reasoning' or
    DeepSeek's 'thought') so messages comply strictly with standard OpenAI schema
    accepted across all providers (Mistral, OpenRouter, Groq).
    """
    clean_messages = []
    for m in messages:
        role = m.get("role")
        clean_m: Dict[str, Any] = {"role": role}

        if "content" in m:
            clean_m["content"] = m["content"]

        if "tool_calls" in m and m["tool_calls"] is not None:
            clean_m["tool_calls"] = m["tool_calls"]

        if "tool_call_id" in m and m["tool_call_id"] is not None:
            clean_m["tool_call_id"] = m["tool_call_id"]

        if "name" in m and m["name"] is not None:
            clean_m["name"] = m["name"]

        clean_messages.append(clean_m)
    return clean_messages


def shrink_tool_history_if_needed(messages: list, budget: int = OPENAI_COMPAT_LOOP_CHAR_BUDGET):
    total = sum(len(m.get("content") or "") for m in messages)
    if total <= budget:
        return

    for m in messages:
        if total <= budget:
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


def _format_wait_time(raw_err: str) -> str:
    match = _RETRY_AFTER_RE.search(raw_err)
    if not match:
        return "a short while"
    minutes_part, seconds_part = match.groups()
    minutes = int(minutes_part) if minutes_part else 0
    seconds = int(float(seconds_part))
    return f"about {minutes}m {seconds}s" if minutes else f"about {seconds}s"


def is_quota_or_rate_limit(raw_err: str) -> bool:
    """
    Returns True when an error from an OpenAI-compatible provider (Groq, Mistral, OpenRouter)
    is a 429 / 402 / resource_exhausted rate-limit or quota error.
    """
    lowered = raw_err.lower()
    status_match = _ERROR_CODE_RE.search(raw_err)
    status = status_match.group(1) if status_match else None

    if status in ("429", "402"):
        return True
    if "429" in raw_err or "402" in raw_err:
        return True

    keywords = (
        "rate_limit_exceeded",
        "rate_limit",
        "rate limit",
        "quota",
        "too many requests",
        "resource_exhausted",
        "payment_required",
        "tpd",
        "tokens per day",
        "credits exhausted",
    )
    return any(kw in lowered for kw in keywords)


def explain_provider_error(provider_name: str, raw_err: str) -> str:
    """Generates a clean, human-friendly explanation for a provider error."""
    status_match = _ERROR_CODE_RE.search(raw_err)
    status = status_match.group(1) if status_match else None
    lowered = raw_err.lower()

    if status == "429" or is_quota_or_rate_limit(raw_err):
        wait_str = _format_wait_time(raw_err)
        return (
            f"{provider_name}'s free rate/quota limit was reached. "
            f"It typically clears in {wait_str} or on the next daily reset."
        )

    if status == "404" or "model_not_found" in lowered or "not found" in lowered:
        return f"The configured model on {provider_name} was not found or is currently unavailable."

    if status == "413" or "context_length_exceeded" in lowered:
        return (
            f"The conversation grew too large for {provider_name}'s context window. "
            "Narrowing the request to a specific file helps."
        )

    if "tool_use_failed" in lowered:
        return (
            f"{provider_name} generated an invalid tool call while trying to make a change. "
            "Try requesting smaller, more targeted edits."
        )

    if "timeout" in lowered:
        return f"The request to {provider_name} timed out."

    return f"{provider_name} returned an error while processing the task."


def run_openai_compat_tool_loop(
    provider: Any,
    provider_name: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    log_callback: LogCallback,
    execute_tool: ToolExecutor,
    command_approval_callback: CommandApprovalCallback,
    cancel_event: Optional[threading.Event] = None,
) -> ToolLoopResult:
    """
    Executes a multi-turn tool calling loop against any OpenAI-compatible provider
    (Groq, Mistral, OpenRouter).
    """
    max_turns = 10
    for _ in range(max_turns):
        if is_cancelled(cancel_event):
            return ToolLoopResult(final_answer="", messages=messages)

        shrink_tool_history_if_needed(messages)
        clean_messages = _sanitize_messages(messages)

        try:
            response_msg = provider.client.chat.completions.create(
                model=provider.model_name,
                messages=clean_messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            ).choices[0].message
        except Exception as e:
            if is_cancelled(cancel_event):
                return ToolLoopResult(final_answer="", messages=messages)

            raw_err = str(e)
            log_callback("system", f"❌ {provider_name} API Error: {raw_err}")
            return ToolLoopResult(provider_error=raw_err, messages=messages)

        # Store clean dictionary in messages
        assistant_dict: Dict[str, Any] = {
            "role": "assistant",
            "content": response_msg.content,
        }
        if response_msg.tool_calls:
            assistant_dict["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc
                for tc in response_msg.tool_calls
            ]
        messages.append(assistant_dict)

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
        final_answer=f"{provider_name} reached maximum tool turns without concluding.",
        messages=messages,
    )
