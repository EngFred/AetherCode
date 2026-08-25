import json
from typing import Callable, Dict, Any, Optional, List, Tuple
from tools.file_manager import SafeFileManager
from tools.tree_builder import generate_directory_tree, get_filtered_file_list
from tools.path_utils import find_referenced_files
from tools.chitchat_utils import is_chitchat
from providers.gemini_provider import GeminiAnalyzerProvider
from providers.groq_provider import GroqExecutorProvider
from core.prompt_builder import PromptBuilder
import config


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

        # --- Session memory ---
        # Persists for as long as this instance lives (i.e. one working
        # directory / one connection — see api.py). Cleared automatically on
        # a project switch, or on demand via reset_history().
        self.chat_history: List[Dict[str, str]] = []
        # Filenames (not content) touched this session, most-recent-last.
        # Lets a vague follow-up ("fix the bug in it") get resolved without
        # re-sending file content or re-running a full Gemini scan.
        self._recent_files: List[str] = []

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
                    "name": "list_project_files",
                    "description": (
                        "List every relevant file path in the project, already filtered to "
                        "exclude build output, dependency folders, and other generated noise "
                        "(see config.IGNORED_DIRECTORIES). Use this instead of run_command with "
                        "'ls -R', 'find', or similar — those walk generated directories "
                        "unfiltered and can return excessive output on a real project."
                    ),
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": (
                        "Run a terminal command (e.g. 'flutter analyze', 'npm test') inside "
                        "the project directory. Do not use this for listing project files — "
                        "use list_project_files instead."
                    ),
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
                }
            }
        ]

    def _track_recent_file(self, relative_path: str):
        if relative_path in self._recent_files:
            self._recent_files.remove(relative_path)
        self._recent_files.append(relative_path)
        if len(self._recent_files) > config.MAX_RECENT_FILES_TRACKED:
            self._recent_files.pop(0)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any], command_approval_callback: Optional[Callable[[str], bool]] = None) -> str:
        rel_path = args.get("relative_path", "")
        if tool_name in ("read_file", "write_file", "delete_file") and rel_path:
            self._track_recent_file(rel_path)

        if tool_name == "read_file":
            return self.file_manager.read_file(rel_path)
        elif tool_name == "write_file":
            return self.file_manager.write_file(rel_path, args.get("content", ""))
        elif tool_name == "delete_file":
            return self.file_manager.delete_file(rel_path)
        elif tool_name == "list_project_files":
            files = get_filtered_file_list(self.root_dir)
            listing = "\n".join(files) if files else "(no files found in project root)"
            return self.file_manager.truncate_output(listing, "list_project_files")
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

    def _get_history_window(self) -> List[Dict[str, str]]:
        """
        Bounded slice of chat_history to prepend to the next call — bounded
        by turn count AND a rough character budget, so a long session never
        silently balloons the request (and your daily token quota) forever.
        Oldest turns drop first, in (user, assistant) pairs.
        """
        if not self.chat_history:
            return []

        window = self.chat_history[-(config.MAX_CHAT_HISTORY_TURNS * 2):]

        total_chars = sum(len(m["content"]) for m in window)
        while len(window) > 2 and total_chars > config.MAX_CHAT_HISTORY_CHARS:
            removed_pair = window[:2]
            window = window[2:]
            total_chars -= sum(len(m["content"]) for m in removed_pair)

        return window

    def _append_turn(self, user_prompt: str, assistant_answer: str):
        """Persists one (user, assistant) exchange. Infra errors are skipped
        so a Groq/Gemini hiccup doesn't end up in the model's own memory."""
        if assistant_answer.startswith("Groq API Error:"):
            return

        self.chat_history.append({"role": "user", "content": user_prompt})
        self.chat_history.append({"role": "assistant", "content": assistant_answer})

        # Belt-and-braces ceiling above the normal window size, so the raw
        # list itself can't grow unbounded across a very long-lived connection.
        hard_cap = config.MAX_CHAT_HISTORY_TURNS * 4
        if len(self.chat_history) > hard_cap:
            self.chat_history = self.chat_history[-hard_cap:]

    def reset_history(self) -> str:
        """Forgets the conversation (e.g. user clicked 'New Chat'). File
        backups / undo stack are untouched — this only clears memory."""
        self.chat_history = []
        self._recent_files = []
        return "Conversation memory cleared."

    def _run_lightweight_chat(self, user_prompt: str, log_callback: Callable[[str, str], None], system_content: str) -> str:
        """
        Single-call, no-tools Groq path. Used for both general chat (no
        project linked) and chit-chat inside a linked project (see
        is_chitchat) — the cheapest possible turn, reserved for messages
        that plainly don't need the directory scan or the tool loop.
        """
        messages = [
            {"role": "system", "content": system_content},
            *self._get_history_window(),
            {"role": "user", "content": user_prompt},
        ]
        try:
            response_msg = self.groq_provider.client.chat.completions.create(
                model=self.groq_provider.model_name,
                messages=messages,
                temperature=0.6
            ).choices[0].message
            final_answer = response_msg.content or "No response generated."
            log_callback("ai", final_answer)
            self._append_turn(user_prompt, final_answer)
            return final_answer
        except Exception as e:
            err = f"Groq API Error: {str(e)}"
            log_callback("system", f"❌ {err}")
            return err

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
            return self._run_lightweight_chat(
                user_prompt,
                log_callback,
                (
                    "You are AetherAgent, a helpful AI coding assistant chatting with the "
                    "user in general chat mode (no project folder is currently linked). "
                    "This is a continuous conversation — use the earlier turns below to "
                    "understand context, references like 'it' or 'that', and anything the "
                    "user already told you, without asking them to repeat themselves."
                ),
            )

        # --- CHIT-CHAT BYPASS: a project IS linked, but this specific turn
        # is plainly small talk (greeting, thanks, etc.) with no code
        # signal in it. Without this, every "hey" / "how are you" on a
        # linked project used to trigger a full Gemini directory scan AND
        # a Groq tool-loop call just to produce a one-line reply. Kept
        # deliberately conservative by is_chitchat: any file-extension-
        # looking token or task verb anywhere in the prompt disqualifies
        # it, so a real request is never misrouted here. ---
        if is_chitchat(user_prompt):
            log_callback("system", "⚡ Small talk detected — skipping scan and tool loop...")
            return self._run_lightweight_chat(
                user_prompt,
                log_callback,
                (
                    "You are AetherAgent, a helpful AI coding assistant. A project "
                    "directory is linked for this session, but the user's current "
                    "message is just small talk and doesn't require touching the "
                    "project — respond naturally and briefly, without mentioning "
                    "files, scans, or tools. This is a continuous conversation — use "
                    "the earlier turns below for context."
                ),
            )

        tools = self.get_tool_definitions()

        # --- Smart routing ---
        referenced_files: List[Tuple[str, str]] = []
        if execution_mode in ("auto", "direct"):
            referenced_files = find_referenced_files(user_prompt, self.file_manager)
            for path, _ in referenced_files:
                self._track_recent_file(path)

        # In Auto mode, skip the full Gemini scan not just when the prompt
        # names a file, but also when we already have file context from
        # earlier in this session — a bare "now fix the bug in it" shouldn't
        # trigger a whole new project scan when we just read that file.
        use_deep_scan = execution_mode == "deep" or (
            execution_mode == "auto" and not referenced_files and not self._recent_files
        )

        if use_deep_scan:
            log_callback("system", "🔍 Phase 1: Scanning project directory with Gemini...")
            dir_tree = self._get_directory_tree()

            gemini_prompt = PromptBuilder.build_gemini_analysis_prompt(user_prompt, dir_tree)
            gemini_analysis = self.gemini_provider.generate_response(gemini_prompt)

            if gemini_analysis.startswith("Gemini Provider Error:"):
                # Infra failure, not a real diagnosis. Keep the user-facing
                # message clean — the raw error (stack/JSON) goes to the
                # server console only, for your own debugging, never into
                # the chat UI. Groq falls back to its own read_file /
                # run_command tools to figure out the project.
                log_callback("system", "⚠️ Gemini unavailable — proceeding with Groq only.")
                print(f"[AetherAgent] Gemini scan failed: {gemini_analysis}")
                user_content = f"User Request: {user_prompt}"
            else:
                log_callback("thinking", gemini_analysis)
                user_content = f"User Request: {user_prompt}\n\nDiagnosis:\n{gemini_analysis}"

            log_callback("system", "⚡ Phase 2: Handing over to Groq for tool execution...")
        else:
            if execution_mode == "direct":
                reason = "Instant mode is selected"
            elif referenced_files:
                reason = "explicit file reference(s) were detected in your message"
            else:
                reason = "using file context from earlier in this session"
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

        if self._recent_files:
            recent_list = ", ".join(self._recent_files)
            user_content += (
                f"\n\n(Files touched earlier this session, most recent last: {recent_list}. "
                f"If the request refers to one of these without naming it, call read_file on "
                f"the right one rather than asking the user which file they mean.)"
            )

        messages = [
            {"role": "system", "content": PromptBuilder.get_groq_system_instruction()},
            *self._get_history_window(),
            {"role": "user", "content": user_content}
        ]

        final_answer = self._run_groq_tool_loop(messages, tools, log_callback, command_approval_callback)
        self._append_turn(user_prompt, final_answer)
        return final_answer

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