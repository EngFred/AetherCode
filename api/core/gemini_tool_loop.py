import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from core.tool_loop import (
    CommandApprovalCallback,
    LogCallback,
    ToolExecutor,
    ToolLoopResult,
    is_cancelled,
    log_tool_result,
)


def openai_tools_to_gemini_tool(tools: List[Dict[str, Any]]) -> types.Tool:
    declarations = []
    for tool in tools:
        function = tool.get("function", {})
        declarations.append(types.FunctionDeclaration(
            name=function.get("name", ""),
            description=function.get("description", ""),
            parameters_json_schema=function.get("parameters", {"type": "object"}),
        ))
    return types.Tool(function_declarations=declarations)


def openai_messages_to_gemini_contents(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[types.Content]]:
    system_parts: List[str] = []
    contents: List[types.Content] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            tool_name = _tool_name_for_result(message, messages)
            tool_call_id = message.get("tool_call_id") or "unknown"
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=(
                        f"Earlier Groq tool result already completed in this same task:\n"
                        f"Tool: {tool_name}\n"
                        f"Call id: {tool_call_id}\n"
                        f"Result:\n{content}"
                    ),
                )],
            ))
            continue

        if role == "assistant":
            text_parts: List[str] = []
            if content:
                text_parts.append(content)
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {})
                text_parts.append(
                    "Earlier Groq requested this tool call in the same task; "
                    "do not repeat it unless the result is insufficient:\n"
                    f"Tool: {function.get('name', '')}\n"
                    f"Call id: {tool_call.get('id', 'unknown')}\n"
                    f"Arguments: {json.dumps(_json_args(function.get('arguments')), sort_keys=True)}"
                )
            if text_parts:
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="\n\n".join(text_parts))],
                ))
            continue

        if role == "user":
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=content)],
            ))

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _tool_name_for_result(
    tool_result_message: Dict[str, Any],
    messages: List[Dict[str, Any]],
) -> str:
    tool_call_id = tool_result_message.get("tool_call_id")
    if not tool_call_id:
        return "unknown_tool"
    for message in messages:
        for tool_call in message.get("tool_calls", []) or []:
            if tool_call.get("id") == tool_call_id:
                return tool_call.get("function", {}).get("name", "unknown_tool")
    return "unknown_tool"


def _json_args(raw_args: Any) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if not raw_args:
        return {}
    try:
        parsed = json.loads(raw_args)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _content_text(content: Optional[types.Content]) -> str:
    if not content or not content.parts:
        return ""
    text_parts = [part.text for part in content.parts if getattr(part, "text", None)]
    return "\n".join(text_parts)


def _function_calls(content: Optional[types.Content]) -> List[types.FunctionCall]:
    if not content or not content.parts:
        return []
    return [
        part.function_call
        for part in content.parts
        if getattr(part, "function_call", None) is not None
    ]


def _candidate_content(response: Any) -> Optional[types.Content]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    return getattr(candidates[0], "content", None)


def append_model_content_preserving_signatures(
    contents: List[types.Content],
    model_content: Optional[types.Content],
):
    if model_content is not None:
        contents.append(model_content)


def explain_gemini_error(raw_err: str) -> str:
    lowered = raw_err.lower()
    if "429" in raw_err or "rate" in lowered or "quota" in lowered:
        return "Gemini also reported a quota or rate-limit error."
    if "timeout" in lowered:
        return "The request to Gemini timed out."
    return "Gemini returned an error while continuing the task."


def run_gemini_tool_loop(
    provider: Any,
    groq_messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    log_callback: LogCallback,
    execute_tool: ToolExecutor,
    command_approval_callback: CommandApprovalCallback,
    cancel_event: Optional[threading.Event] = None,
) -> ToolLoopResult:
    system_instruction, contents = openai_messages_to_gemini_contents(groq_messages)
    gemini_tool = openai_tools_to_gemini_tool(tools)

    max_turns = 10
    for _ in range(max_turns):
        if is_cancelled(cancel_event):
            return ToolLoopResult(final_answer="", messages=groq_messages)

        try:
            response = provider.client.models.generate_content(
                model=provider.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    tools=[gemini_tool],
                ),

            )
        except Exception as e:
            if is_cancelled(cancel_event):
                return ToolLoopResult(final_answer="", messages=groq_messages)
            raw_err = str(e)
            log_callback("system", f"❌ Gemini API Error: {raw_err}")
            return ToolLoopResult(provider_error=raw_err, messages=groq_messages)

        model_content = _candidate_content(response)
        append_model_content_preserving_signatures(contents, model_content)

        calls = _function_calls(model_content)
        if calls:
            response_parts: List[types.Part] = []
            for function_call in calls:
                if is_cancelled(cancel_event):
                    return ToolLoopResult(final_answer="", messages=groq_messages)

                fn_name = function_call.name or ""
                fn_args = dict(function_call.args or {})
                tool_result = execute_tool(fn_name, fn_args, command_approval_callback)
                log_tool_result(log_callback, fn_name, fn_args, tool_result)
                response_parts.append(types.Part(function_response=types.FunctionResponse(
                    id=function_call.id,
                    name=fn_name,
                    response={"result": tool_result},
                )))

            contents.append(types.Content(role="user", parts=response_parts))
            continue

        final_answer = _content_text(model_content) or "Task completed successfully."
        if is_cancelled(cancel_event):
            return ToolLoopResult(final_answer="", messages=groq_messages)
        log_callback("ai", final_answer)
        return ToolLoopResult(final_answer=final_answer, messages=groq_messages)

    return ToolLoopResult(
        final_answer="Agent reached maximum tool turns without concluding.",
        messages=groq_messages,
    )
