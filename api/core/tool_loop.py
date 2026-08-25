import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


ToolExecutor = Callable[[str, Dict[str, Any], Optional[Callable[[str], bool]]], str]
LogCallback = Callable[[str, str], None]
CommandApprovalCallback = Optional[Callable[[str], bool]]


@dataclass
class ToolLoopResult:
    final_answer: str = ""
    provider_error: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None

    @property
    def ok(self) -> bool:
        return self.provider_error is None


def is_cancelled(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def tool_call_label(name: str, args: Dict[str, Any]) -> str:
    if name == "read_file":
        return f"Read {args.get('relative_path', 'file')}"
    if name == "write_file":
        return f"Wrote {args.get('relative_path', 'file')}"
    if name == "delete_file":
        return f"Deleted {args.get('relative_path', 'file')}"
    if name == "list_project_files":
        return "Listed project files"
    if name == "run_command":
        return f"Ran: {args.get('command', '')}"
    if name == "push_changes":
        return f"Pushed changes: {args.get('commit_message', 'Auto-commit by AetherAgent')}"
    return name


def log_tool_result(log_callback: LogCallback, tool_name: str, args: Dict[str, Any], result: str):
    log_callback("tool", json.dumps({
        "tool": tool_name,
        "label": tool_call_label(tool_name, args),
        "result": result,
    }))
