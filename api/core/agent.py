import threading
from typing import Callable, Dict, Any, Optional, List, Tuple
from tools.file_manager import SafeFileManager
from tools.tree_builder import generate_directory_tree, get_filtered_file_list
from tools.path_utils import find_referenced_files
from tools.chitchat_utils import is_chitchat
from providers.gemini_provider import GeminiAnalyzerProvider
from providers.groq_provider import GroqExecutorProvider
from providers.cerebras_provider import CerebrasExecutorProvider
from core.gemini_tool_loop import explain_gemini_error, run_gemini_tool_loop
from core.groq_tool_loop import explain_groq_error, is_groq_quota_or_rate_limit, run_groq_tool_loop
from core.cerebras_tool_loop import (
    explain_cerebras_error,
    is_cerebras_quota_or_rate_limit,
    run_cerebras_tool_loop,
)
from core.prompt_builder import PromptBuilder
import config


class AetherAgent:
    def __init__(
        self,
        root_dir: str,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
        cerebras_key: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.file_manager = SafeFileManager(root_dir)
        self.gemini_provider = GeminiAnalyzerProvider(api_key=gemini_key)
        self.groq_provider = GroqExecutorProvider(api_key=groq_key)
        self.cerebras_provider = CerebrasExecutorProvider(api_key=cerebras_key)

        # One AetherAgent lives for the whole WebSocket session (see api.py),
        # so caching the tree here saves a full filesystem walk on every
        # follow-up message.
        self._tree_cache: Optional[str] = None

        # --- Session memory ---
        # Persists for as long as this instance lives (one working directory /
        # one connection).  Cleared on project switch or via reset_history().
        self.chat_history: List[Dict[str, str]] = []

        # Filenames (not content) touched this session, most-recent-last.
        # Lets a vague follow-up ("fix the bug in it") get resolved without
        # re-sending file content or re-running a full Gemini scan.
        self._recent_files: List[str] = []

        # True when the MOST RECENT turn was routed through the chit-chat
        # bypass.  Lets a reply continuing a small-talk exchange skip the
        # scan too, even though it doesn't match a fixed opening phrase.
        self._last_turn_was_chitchat: bool = False

    # -----------------------------------------------------------------------
    # Tool definitions
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

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
        by turn count AND a rough character budget so a long session never
        silently balloons the request (and daily token quota) forever.
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
        """
        Persists one (user, assistant) exchange.
        Infra / provider errors are skipped so a failed call doesn't
        pollute the model's own memory.
        """
        # Skip error strings from any provider so partial failures are
        # never replayed as if they were successful AI responses.
        skip_prefixes = (
            "Groq API Error:",
            "Cerebras API Error:",
            "Gemini Provider Error:",
        )
        if any(assistant_answer.startswith(p) for p in skip_prefixes):
            return

        self.chat_history.append({"role": "user", "content": user_prompt})
        self.chat_history.append({"role": "assistant", "content": assistant_answer})

        # Belt-and-braces ceiling above the normal window so the raw list
        # can't grow unbounded across a very long-lived connection.
        hard_cap = config.MAX_CHAT_HISTORY_TURNS * 4
        if len(self.chat_history) > hard_cap:
            self.chat_history = self.chat_history[-hard_cap:]

    def reset_history(self) -> str:
        """Forgets the conversation (e.g. user clicked 'New Chat').
        File backups / undo stack are untouched — only memory is cleared."""
        self.chat_history = []
        self._recent_files = []
        self._last_turn_was_chitchat = False
        return "Conversation memory cleared."

    # -----------------------------------------------------------------------
    # Lightweight-chat path (no tools, no scan)
    # -----------------------------------------------------------------------

    def _run_lightweight_chat(
        self,
        user_prompt: str,
        log_callback: Callable[[str, str], None],
        system_content: str,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Single-call, no-tools path.  Used for general chat (no project
        linked) and chit-chat inside a linked project.

        Fallback chain: Groq → Cerebras.
        Gemini is intentionally excluded here (its tool-loop interface is
        different and would be overkill for a plain-text reply).
        """
        messages = [
            {"role": "system", "content": system_content},
            *self._get_history_window(),
            {"role": "user", "content": user_prompt},
        ]

        # --- Primary: Groq ---
        try:
            response_msg = self.groq_provider.client.chat.completions.create(
                model=self.groq_provider.model_name,
                messages=messages,
                temperature=0.6,
            ).choices[0].message
            final_answer = response_msg.content or "No response generated."

            if cancel_event is not None and cancel_event.is_set():
                return ""

            log_callback("ai", final_answer)
            self._append_turn(user_prompt, final_answer)
            return final_answer

        except Exception as e:
            raw_err = str(e)
            if cancel_event is not None and cancel_event.is_set():
                return ""

            # If Groq is rate-limited, try Cerebras before giving up.
            if is_groq_quota_or_rate_limit(raw_err):
                log_callback(
                    "system",
                    "⚠️ Groq quota reached — switching to Cerebras for this reply...",
                )
                return self._run_lightweight_chat_cerebras(
                    user_prompt, messages, log_callback, cancel_event
                )

            # Non-quota Groq error — surface it cleanly.
            err = f"Groq API Error: {raw_err}"
            log_callback("system", f"❌ {err}")
            return err

    def _run_lightweight_chat_cerebras(
        self,
        user_prompt: str,
        messages: list,
        log_callback: Callable[[str, str], None],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Cerebras fallback for the lightweight (no-tools) chat path.
        Called only when Groq hits a quota/rate-limit.
        """
        try:
            response_msg = self.cerebras_provider.client.chat.completions.create(
                model=self.cerebras_provider.model_name,
                messages=messages,
                temperature=0.6,
            ).choices[0].message
            final_answer = response_msg.content or "No response generated."

            if cancel_event is not None and cancel_event.is_set():
                return ""

            log_callback("ai", final_answer)
            self._append_turn(user_prompt, final_answer)
            return final_answer

        except Exception as e:
            raw_err = str(e)
            if cancel_event is not None and cancel_event.is_set():
                return ""

            err_msg = (
                "⚠️ Both Groq and Cerebras are currently rate-limited and couldn't "
                "respond. Please wait a few minutes and try again."
            )
            log_callback("system", f"❌ Cerebras API Error: {raw_err}")
            log_callback("ai", err_msg)
            return err_msg

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        user_prompt: str,
        log_callback: Callable[[str, str], None],
        command_approval_callback: Optional[Callable[[str], bool]] = None,
        is_general_chat: bool = False,
        execution_mode: str = "auto",
        cancel_event: Optional[threading.Event] = None,
    ) -> str:

        # --- GENERAL CHAT BYPASS: no project selected → lightweight path ---
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

        # --- CHIT-CHAT BYPASS: project linked but message is plainly small talk ---
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
                    "never claim you did any of those things. This is a continuous "
                    "conversation — use the earlier turns below for context."
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

        # Skip the full Gemini scan not just when the prompt names a file,
        # but also when we already have file context from earlier this session.
        use_deep_scan = execution_mode == "deep" or (
            execution_mode == "auto" and not referenced_files and not self._recent_files
        )

        if use_deep_scan:
            log_callback("system", "🔍 Phase 1: Scanning project directory with Gemini...")
            dir_tree = self._get_directory_tree()

            gemini_prompt = PromptBuilder.build_gemini_analysis_prompt(user_prompt, dir_tree)
            gemini_analysis = self.gemini_provider.generate_response(gemini_prompt)

            if gemini_analysis.startswith("Gemini Provider Error:"):
                # Infra failure, not a real diagnosis.  Groq will use its own
                # tools to figure out the project.
                log_callback("system", "⚠️ Gemini unavailable — proceeding with executor chain only.")
                print(f"[AetherAgent] Gemini scan failed: {gemini_analysis}")
                user_content = f"User Request: {user_prompt}"
            else:
                log_callback("thinking", gemini_analysis)
                user_content = f"User Request: {user_prompt}\n\nDiagnosis:\n{gemini_analysis}"

            log_callback("system", "⚡ Phase 2: Handing over to executor chain...")
        else:
            if execution_mode == "direct":
                reason = "Instant mode is selected"
            elif referenced_files:
                reason = "explicit file reference(s) were detected in your message"
            else:
                reason = "using file context from earlier in this session"
            log_callback("system", f"⚡ Skipping directory scan — {reason}. Handing straight to executor chain...")

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
            # The turn's 180s ceiling fired while the tool loop was still
            # running.  Nothing further is sent or saved — a cancelled,
            # incomplete exchange would be confusing context for the next turn.
            return final_answer

        self._append_turn(user_prompt, final_answer)
        return final_answer

    # -----------------------------------------------------------------------
    # 3-stage executor fallback chain
    # -----------------------------------------------------------------------

    def _run_executor_tool_loop(
        self,
        messages: list,
        tools: list,
        log_callback: Callable[[str, str], None],
        command_approval_callback: Optional[Callable[[str], bool]],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """
        Runs the executor chain: Groq → Cerebras → Gemini.

        Each stage:
          1. Attempts a full tool-calling loop with full conversation context.
          2. On success — returns immediately (the other providers are untouched).
          3. On quota / rate-limit 429 — logs a status message and hands the
             accumulated message history to the next stage so it picks up
             exactly where the previous model left off.
          4. On any other error (malformed tool call, timeout, etc.) — surfaces
             a friendly message immediately without continuing the chain, since
             retrying a non-quota failure on a different provider rarely helps.

        User-visible status messages at each hand-off make the fallback
        transparent without being alarming.
        """

        # ── Stage 1: Groq ───────────────────────────────────────────────────
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

        if cancel_event is not None and cancel_event.is_set():
            return ""

        groq_err = groq_result.provider_error or ""

        if not is_groq_quota_or_rate_limit(groq_err):
            # Non-quota Groq failure — surface it directly, no chain attempt.
            friendly = (
                "⚠️ I hit an error partway through this task and had to stop. "
                f"{explain_groq_error(groq_err)} "
                "Any changes already completed remain on disk. You can try again — if the "
                "request touches a lot of code, narrowing it to the specific file or values "
                "usually helps."
            )
            log_callback("ai", friendly)
            return friendly

        # ── Stage 2: Cerebras ───────────────────────────────────────────────
        log_callback(
            "system",
            "⚠️ Groq's quota was reached mid-task. Continuing seamlessly with Cerebras...",
        )

        # Pass the full accumulated message list so Cerebras sees every tool
        # call and result that Groq already performed.
        cerebras_messages = groq_result.messages or messages

        cerebras_result = run_cerebras_tool_loop(
            self.cerebras_provider,
            cerebras_messages,
            tools,
            log_callback,
            self._execute_tool,
            command_approval_callback,
            cancel_event=cancel_event,
        )

        if cerebras_result.ok:
            return cerebras_result.final_answer

        if cancel_event is not None and cancel_event.is_set():
            return ""

        cerebras_err = cerebras_result.provider_error or ""

        if not is_cerebras_quota_or_rate_limit(cerebras_err):
            # Non-quota Cerebras failure — surface it, no Gemini attempt.
            friendly = (
                "⚠️ Groq's quota was reached mid-task, and Cerebras (the fallback) "
                "also encountered an error. "
                f"{explain_cerebras_error(cerebras_err)} "
                "Any changes already completed remain on disk. Try again shortly, or "
                "narrow the request to a smaller change."
            )
            log_callback("ai", friendly)
            return friendly

        # ── Stage 3: Gemini ─────────────────────────────────────────────────
        log_callback(
            "system",
            "⚠️ Cerebras quota also reached. Passing the task to Gemini to finish...",
        )

        # Pass Cerebras' accumulated messages so Gemini has the full picture.
        gemini_messages = cerebras_result.messages or cerebras_messages

        gemini_result = run_gemini_tool_loop(
            self.gemini_provider,
            gemini_messages,
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

        # All three providers exhausted or failed.
        gemini_err = gemini_result.provider_error or ""
        friendly = (
            "⚠️ All three AI providers (Groq, Cerebras, and Gemini) have either hit "
            "their daily quota or returned an error for this task. "
            f"{explain_gemini_error(gemini_err)} "
            "Any changes already completed remain on disk. "
            "Free-tier limits typically reset within 24 hours — try again later, "
            "or break the request into smaller steps to make it fit within the "
            "remaining quota of one provider."
        )
        log_callback("ai", friendly)
        return friendly
