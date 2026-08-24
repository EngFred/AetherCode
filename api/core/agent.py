import json
from typing import Callable, Dict, Any, Optional, List, Tuple
from tools.file_manager import SafeFileManager
from tools.tree_builder import generate_directory_tree
from tools.path_utils import find_referenced_files
from providers.gemini_provider import GeminiAnalyzerProvider
from providers.groq_provider import GroqExecutorProvider
from core.prompt_builder import PromptBuilder


class AetherAgent:
    def __init__(self, root_dir: str, gemini_key: Optional[str] = None, groq_key: Optional[str] = None):
        self.root_dir = root_dir
        self.file_manager = SafeFileManager(root_dir)
        self.gemini_provider = GeminiAnalyzerProvider(api_key=gemini_key)
        self.groq_provider = GroqExecutorProvider(api_key=groq_key)
        # One AetherAgent now lives for the whole WebSocket session (see api.py),
        # so caching the tree here is safe and saves a full filesystem walk on
        # every follow-up message.
        self._tree_cache: Optional[str] = None

    def get_tool_definitions(self) -> list[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file relative to project root.",
                    "parameters": {"type": "object", "properties": {"relative_path": {"type": "string"}}, "required": ["relative_path"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite content of a file relative to project root.",
                    "parameters": {"type": "object", "properties": {"relative_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["relative_path", "content"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file relative to project root.",
                    "parameters": {"type": "object", "properties": {"relative_path": {"type": "string"}}, "required": ["relative_path"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a terminal command (e.g. 'flutter analyze') inside project directory.",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
                }
            }
        ]

    def _execute_tool(self, tool_name: str, args: Dict[str, Any], command_approval_callback: Optional[Callable[[str], bool]] = None) -> str:
        rel_path = args.get("relative_path", "")
        if tool_name == "read_file":
            return self.file_manager.read_file(rel_path)
        elif tool_name == "write_file":
            return self.file_manager.write_file(rel_path, args.get("content", ""))
        elif tool_name == "delete_file":
            return self.file_manager.delete_file(rel_path)
        elif tool_name == "run_command":
            cmd = args.get("command", "")
            if command_approval_callback and not command_approval_callback(cmd):
                return "Execution Denied: User rejected terminal command execution."
            return self.file_manager.run_command(cmd)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def undo(self) -> str:
        return self.file_manager.undo_last_change()

    def _get_directory_tree(self) -> str:
        if self._tree_cache is None:
            self._tree_cache = generate_directory_tree(self.root_dir)
        return self._tree_cache

    def _run_groq_tool_loop(
        self,
        messages: list,
        tools: list,
        log_callback: Callable[[str, str], None],
        command_approval_callback: Optional[Callable[[str], bool]],
    ) -> str:
        max_turns = 10
        for _ in range(max_turns):
            try:
                response_msg = self.groq_provider.client.chat.completions.create(
                    model=self.groq_provider.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                ).choices[0].message
            except Exception as e:
                err = f"Groq API Error: {str(e)}"
                log_callback("system", f"❌ {err}")
                return err

            # The SDK returns a pydantic model, not a dict. Appending it raw
            # breaks the next request in this loop (and any retry) — the
            # OpenAI-compatible client needs plain dicts in `messages`.
            messages.append(response_msg.model_dump(exclude_none=True))

            if response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        fn_args = {}

                    log_callback("system", f"🛠️ Agent Tool Call -> {fn_name}({fn_args})")
                    tool_result = self._execute_tool(fn_name, fn_args, command_approval_callback)
                    log_callback("system", f"   Result: {tool_result}")

                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
            else:
                final_answer = response_msg.content or "Task completed successfully."
                log_callback("ai", final_answer)
                return final_answer

        return "Agent reached maximum tool turns without concluding."

    def run(
        self,
        user_prompt: str,
        log_callback: Callable[[str, str], None],
        command_approval_callback: Optional[Callable[[str], bool]] = None,
        is_general_chat: bool = False,
        execution_mode: str = "auto",
    ) -> str:

        # --- GENERAL CHAT BYPASS: no project selected → always Groq-only,
        # regardless of the mode the user picked. ---
        if is_general_chat:
            log_callback("system", "⚡ General Chat Mode: Bypassing file scan...")
            messages = [
                {"role": "system", "content": "You are AetherAgent, an AI coding assistant. The user is in general chat mode. No project folder is active."},
                {"role": "user", "content": user_prompt}
            ]
            try:
                response_msg = self.groq_provider.client.chat.completions.create(
                    model=self.groq_provider.model_name,
                    messages=messages,
                    temperature=0.6
                ).choices[0].message
                final_answer = response_msg.content or "No response generated."
                log_callback("ai", final_answer)
                return final_answer
            except Exception as e:
                err = f"Groq API Error: {str(e)}"
                log_callback("system", f"❌ {err}")
                return err

        tools = self.get_tool_definitions()

        # --- Smart routing ---
        referenced_files: List[Tuple[str, str]] = []
        if execution_mode in ("auto", "direct"):
            referenced_files = find_referenced_files(user_prompt, self.file_manager)

        use_deep_scan = execution_mode == "deep" or (execution_mode == "auto" and not referenced_files)

        if use_deep_scan:
            log_callback("system", "🔍 Phase 1: Scanning project directory with Gemini...")
            dir_tree = self._get_directory_tree()

            gemini_prompt = PromptBuilder.build_gemini_analysis_prompt(user_prompt, dir_tree)
            gemini_analysis = self.gemini_provider.generate_response(gemini_prompt)

            log_callback("ai", f"📋 **Gemini Diagnosis:**\n\n{gemini_analysis}")
            log_callback("system", "⚡ Phase 2: Handing over to Groq for tool execution...")

            user_content = f"User Request: {user_prompt}\n\nGemini Diagnosis:\n{gemini_analysis}"
        else:
            reason = "Instant mode is selected" if execution_mode == "direct" else "explicit file reference(s) were detected in your message"
            log_callback("system", f"⚡ Skipping directory scan — {reason}. Handing straight to Groq...")

            if referenced_files:
                context_blocks = "\n\n".join(f"--- {path} ---\n{content}" for path, content in referenced_files)
                user_content = (
                    f"User Request: {user_prompt}\n\n"
                    f"Referenced file(s), pre-loaded — call read_file again only if you need "
                    f"something not shown here (e.g. an import):\n{context_blocks}"
                )
            else:
                user_content = f"User Request: {user_prompt}"

        messages = [
            {"role": "system", "content": PromptBuilder.get_groq_system_instruction()},
            {"role": "user", "content": user_content}
        ]

        return self._run_groq_tool_loop(messages, tools, log_callback, command_approval_callback)