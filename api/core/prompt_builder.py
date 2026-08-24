class PromptBuilder:
    """
    Constructs structured system instructions and prompts for Gemini and Groq.
    """

    @staticmethod
    def build_gemini_analysis_prompt(user_prompt: str, dir_tree: str) -> str:
        return f"""
User Request: {user_prompt}

Project Directory Structure:
{dir_tree}

Instructions:
1. Analyze the user request and the project directory structure.
2. Identify which specific file(s) are most relevant or likely contain the root cause.
3. Provide a brief analysis identifying the exact file path(s) to target and the goal for fixing/modifying them.
"""

    @staticmethod
    def get_groq_system_instruction() -> str:
        return """
You are AetherAgent, an autonomous coding assistant with file system and terminal access.
Your job is to read, edit, create, or delete files, and run terminal commands (e.g., 'flutter analyze', 'npm test', 'pip install') in the user's project directory to satisfy their request.
You have tools available: read_file, write_file, delete_file, run_command.
Always inspect file content before editing if needed.
Be concise and execute changes using the appropriate tool calls.
"""