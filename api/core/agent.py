import threading
from typing import Callable, Dict, Any, Optional, List, Tuple
from tools.file_manager import SafeFileManager
from tools.tree_builder import generate_directory_tree, get_filtered_file_list
from tools.path_utils import find_referenced_files
from tools.chitchat_utils import is_chitchat
from providers.gemini_provider import GeminiAnalyzerProvider
from providers.groq_provider import GroqExecutorProvider
from core.gemini_tool_loop import explain_gemini_error, run_gemini_tool_loop
from core.groq_tool_loop import explain_groq_error, is_groq_quota_or_rate_limit, run_groq_tool_loop
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
        # True when the MOST RECENT turn was routed through the
        # chit-chat bypass below. Lets a reply that continues an
        # already-established small-talk exchange ("im good and you?")
        # skip the scan too, even though it doesn't match one of the
        # fixed opening phrases in chitchat_utils — see is_chitchat's
        # is_continuation param.
        self._last_turn_was_chitchat: bool = False

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
            },
                        {
                "type": "function",
                "function": {
                    "name": "push_changes",
                    "description": (
                        "Stage all changes, commit them, and push to the remote git "
                        "repository in one step. Use this instead of running 'git add', "
                        "'git commit', and 'git push' individually via run_command. "
                        "This tool does not show you the diff itself — if you want "
                        "commit_message to actually describe what changed (rather than "
                        "a generic message), first call run_command with "
                        "'git diff --stat' (or 'git status --porcelain' for a quicker, "
                        "file-level view) and base commit_message on that output before "
                        "calling this tool."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commit_message": {
                                "type": "string",
                                "description": "Commit message to use. Optional — defaults to 'Auto-commit by AetherAgent' if omitted."
                            }
                        },
                        "required": []
                    }
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
        elif tool_name == "push_changes":
            commit_msg = args.get("commit_message", "Auto-commit by AetherAgent")
            return self.file_manager.push_changes(commit_msg)
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
        self._last_turn_was_chitchat = False
        return "Conversation memory cleared."

    def _run_lightweight_chat(
        self,
        user_prompt: str,
        log_callback: Callable[[str, str], None],
        system_content: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
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

            if cancel_event is not None and cancel_event.is_set():
                # This turn's timeout already fired while the call was in
                # flight — the frontend has already been told the task is
                # over, so don't send a late reply for it or save it.
                return ""

            log_callback("ai", final_answer)
            self._append_turn(user_prompt, final_answer)
            return final_answer
        except Exception as e:
            err = f"Groq API Error: {str(e)}"
            if cancel_event is None or not cancel_event.is_set():
                log_callback("system", f"❌ {err}")
            return err

    def run(
        self,
        user_prompt: str,
        log_callback: Callable[[str, str], None],
        command_approval_callback: Optional[Callable[[str], bool]] = None,
        is_general_chat: bool = False,
        execution_mode: str = "auto",
        cancel_event: Optional[threading.Event] = None,
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
                cancel_event=cancel_event,
            )

        # --- CHIT-CHAT BYPASS: a project IS linked, but this specific turn
        # is plainly small talk (greeting, thanks, etc.) with no code
        # signal in it. Without this, every "hey" / "how are you" on a
        # linked project used to trigger a full Gemini directory scan AND
        # a Groq tool-loop call just to produce a one-line reply. Kept
        # deliberately conservative by is_chitchat: any file-extension-
        # looking token or task verb anywhere in the prompt disqualifies
        # it, so a real request is never misrouted here. ---
        if is_chitchat(user_prompt, is_continuation=self._last_turn_was_chitchat):
            log_callback("system", "⚡ Small talk detected — skipping scan and tool loop...")
            self._last_turn_was_chitchat = True
            return self._run_lightweight_chat(
                user_prompt,
                log_callback,
                (
                    "You are AetherAgent, a helpful AI coding assistant. A project "
                    "directory is linked for this session, but the user's current "
                    "message is just small talk and doesn't require touching the "
                    "project — respond naturally and briefly, without mentioning "
                    "files, scans, or tools. You cannot inspect files, change files, "
                    "run commands, commit, or push from this no-tools chat path, so "
                    "never claim you did any of those things. This is a continuous conversation — use "
                    "the earlier turns below for context."
                ),
                cancel_event=cancel_event,
            )

        self._last_turn_was_chitchat = False

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
            {"role": "system", "content": PromptBuilder.get_executor_system_instruction()},
            *self._get_history_window(),
            {"role": "user", "content": user_content}
        ]

        final_answer = self._run_executor_tool_loop(
            messages, tools, log_callback, command_approval_callback, cancel_event=cancel_event
        )

        if cancel_event is not None and cancel_event.is_set():
            # This turn's 180s ceiling fired while the tool loop was still
            # running. The frontend was already told the task ended, so
            # nothing further is sent — and nothing is saved to
            # chat_history: a cancelled, incomplete exchange would just be
            # confusing context for whatever the next real turn is.
            return final_answer

        self._append_turn(user_prompt, final_answer)
        return final_answer

    def _run_executor_tool_loop(
        self,
        messages: list,
        tools: list,
        log_callback: Callable[[str, str], None],
        command_approval_callback: Optional[Callable[[str], bool]],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        groq_result = run_groq_tool_loop(
            self.groq_provider,
            messages,
            tools,
            log_callback,
            self._execute_tool,
            command_approval_callback,
            cancel_event=cancel_event,
        )
        if groq_result.ok:
            return groq_result.final_answer

        raw_err = groq_result.provider_error or ""
        if cancel_event is not None and cancel_event.is_set():
            return ""

        if is_groq_quota_or_rate_limit(raw_err):
            log_callback(
                "system",
                "⚠️ Groq hit a quota/rate limit mid-task. Continuing the same task with Gemini...",
            )
            gemini_result = run_gemini_tool_loop(
                self.gemini_provider,
                groq_result.messages or messages,
                tools,
                log_callback,
                self._execute_tool,
                command_approval_callback,
                cancel_event=cancel_event,
            )
            if gemini_result.ok:
                return gemini_result.final_answer

            if cancel_event is not None and cancel_event.is_set():
                return ""

            friendly = (
                "⚠️ I switched from Groq to Gemini after Groq hit a quota/rate limit, "
                "but Gemini could not finish the continuation. "
                f"{explain_gemini_error(gemini_result.provider_error or '')} "
                "Any changes already completed remain on disk; try again after the "
                "provider limit clears, or narrow the request if it touches a lot of code."
            )
            log_callback("ai", friendly)
            return friendly

        friendly = (
            "⚠️ I hit an error partway through this task and had to stop. "
            f"{explain_groq_error(raw_err)} "
            "Any changes already completed remain on disk. You can try again - if the "
            "request touches a lot of code, narrowing it to the specific file or values "
            "usually helps."
        )
        log_callback("ai", friendly)
        return friendly
