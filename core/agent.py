import json
from typing import Callable, Dict, Any, Optional
from tools.file_manager import SafeFileManager
from tools.tree_builder import generate_directory_tree
from providers.gemini_provider import GeminiAnalyzerProvider
from providers.groq_provider import GroqExecutorProvider
from core.prompt_builder import PromptBuilder

class AetherAgent:
    """
    Orchestrates the multi-model AI pipeline:
    Gemini scans directory tree -> Groq executes local disk tool calls & commands.
    """

    def __init__(self, root_dir: str, gemini_key: Optional[str] = None, groq_key: Optional[str] = None):
        self.root_dir = root_dir
        self.file_manager = SafeFileManager(root_dir)
        self.gemini_provider = GeminiAnalyzerProvider(api_key=gemini_key)
        self.groq_provider = GroqExecutorProvider(api_key=groq_key)

    def get_tool_definitions(self) -> list[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file relative to project root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {"type": "string"}
                        },
                        "required": ["relative_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Create or overwrite content of a file relative to project root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["relative_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_file",
                    "description": "Delete a file relative to project root.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {"type": "string"}
                        },
                        "required": ["relative_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a terminal command (e.g. 'flutter analyze', 'npm install', 'pytest') inside project directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ]

    def _execute_tool(
        self, 
        tool_name: str, 
        args: Dict[str, Any],
        command_approval_callback: Optional[Callable[[str], bool]] = None
    ) -> str:
        rel_path = args.get("relative_path", "")

        if tool_name == "read_file":
            return self.file_manager.read_file(rel_path)
        elif tool_name == "write_file":
            return self.file_manager.write_file(rel_path, args.get("content", ""))
        elif tool_name == "delete_file":
            return self.file_manager.delete_file(rel_path)
        elif tool_name == "run_command":
            cmd = args.get("command", "")
            if command_approval_callback:
                approved = command_approval_callback(cmd)
                if not approved:
                    return "Execution Denied: User rejected terminal command execution."
            return self.file_manager.run_command(cmd)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def undo(self) -> str:
        """Exposes the undo operation to the user UI."""
        return self.file_manager.undo_last_change()

    def run(
        self, 
        user_prompt: str, 
        log_callback: Callable[[str], None] = print,
        command_approval_callback: Optional[Callable[[str], bool]] = None
    ) -> str:
        log_callback("🔍 Phase 1: Scanning project directory with Gemini...")
        dir_tree = generate_directory_tree(self.root_dir)

        # Step 1: Gemini Analysis
        gemini_prompt = PromptBuilder.build_gemini_analysis_prompt(user_prompt, dir_tree)
        gemini_analysis = self.gemini_provider.generate_response(gemini_prompt)

        log_callback(f"\n📋 Gemini Diagnosis:\n{gemini_analysis}\n")
        log_callback("⚡ Phase 2: Handing over to Groq for tool execution...")

        # Step 2: Groq Tool Execution Loop
        tools = self.get_tool_definitions()
        system_instruction = PromptBuilder.get_groq_system_instruction()
        combined_prompt = f"User Request: {user_prompt}\n\nGemini Diagnosis:\n{gemini_analysis}"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": combined_prompt}
        ]

        max_turns = 10
        for turn in range(max_turns):
            try:
                response_msg = self.groq_provider.client.chat.completions.create(
                    model=self.groq_provider.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1
                ).choices[0].message
            except Exception as e:
                err_msg = f"Groq API Error: {str(e)}"
                log_callback(f"❌ {err_msg}")
                return err_msg

            messages.append(response_msg)

            if response_msg.tool_calls:
                for tool_call in response_msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        fn_args = {}

                    log_callback(f"🛠️ Agent Tool Call -> {fn_name}({fn_args})")
                    tool_result = self._execute_tool(fn_name, fn_args, command_approval_callback)
                    log_callback(f"   Result: {tool_result}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
            else:
                final_answer = response_msg.content or "Task completed successfully."
                log_callback(f"\n✅ Task Finished:\n{final_answer}")
                return final_answer

        return "Agent reached maximum tool turns without concluding."