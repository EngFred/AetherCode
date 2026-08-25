import unittest

from google.genai import types

from core.gemini_tool_loop import (
    append_model_content_preserving_signatures,
    openai_messages_to_gemini_contents,
    openai_tools_to_gemini_tool,
)
from core.groq_tool_loop import is_groq_quota_or_rate_limit


class GroqErrorClassificationTests(unittest.TestCase):
    def test_only_429_rate_limit_errors_fallback(self):
        err = "Error code: 429 - {'error': {'code': 'rate_limit_exceeded'}}"
        self.assertTrue(is_groq_quota_or_rate_limit(err))

    def test_context_length_does_not_fallback(self):
        err = "Error code: 413 - {'error': {'code': 'context_length_exceeded'}}"
        self.assertFalse(is_groq_quota_or_rate_limit(err))

    def test_tool_generation_failure_does_not_fallback(self):
        err = "Error code: 400 - tool_use_failed failed_generation 429 rate_limit_exceeded"
        self.assertFalse(is_groq_quota_or_rate_limit(err))


class GeminiTranslationTests(unittest.TestCase):
    def test_openai_tool_exchange_becomes_text_context_not_unsigned_function_parts(self):
        messages = [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"relative_path": "api/core/agent.py"}',
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        ]

        system_instruction, contents = openai_messages_to_gemini_contents(messages)

        self.assertEqual(system_instruction, "system rules")
        self.assertEqual([content.role for content in contents], ["user", "model", "user"])
        self.assertIsNone(contents[1].parts[0].function_call)
        self.assertIsNone(contents[2].parts[0].function_response)
        self.assertIn("Earlier Groq requested this tool call", contents[1].parts[0].text)
        self.assertIn("read_file", contents[1].parts[0].text)
        self.assertIn('"relative_path": "api/core/agent.py"', contents[1].parts[0].text)
        self.assertIn("Earlier Groq tool result already completed", contents[2].parts[0].text)
        self.assertIn("file contents", contents[2].parts[0].text)

    def test_openai_tool_schema_becomes_gemini_declaration(self):
        gemini_tool = openai_tools_to_gemini_tool([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"relative_path": {"type": "string"}},
                    "required": ["relative_path"],
                },
            },
        }])

        declaration = gemini_tool.function_declarations[0]
        self.assertEqual(declaration.name, "read_file")
        self.assertEqual(declaration.parameters_json_schema["required"], ["relative_path"])

    def test_gemini_model_content_is_appended_without_losing_thought_signature(self):
        contents = []
        model_content = types.Content(role="model", parts=[
            types.Part(
                function_call=types.FunctionCall(
                    id="gemini_call_1",
                    name="list_project_files",
                    args={},
                ),
                thought_signature=b"signature-bytes",
            )
        ])

        append_model_content_preserving_signatures(contents, model_content)

        self.assertIs(contents[0], model_content)
        self.assertEqual(contents[0].parts[0].thought_signature, b"signature-bytes")


if __name__ == "__main__":
    unittest.main()
