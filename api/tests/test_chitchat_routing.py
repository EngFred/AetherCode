import unittest

from tools.chitchat_utils import is_chitchat


class ChitChatRoutingTests(unittest.TestCase):
    def test_git_requests_are_not_chitchat_even_as_continuations(self):
        prompts = [
            "i want you to push the changes or updates in code to git",
            "stage everything and push, git the right commit message",
            "commit these changes",
            "what branch am i on?",
            "show git status",
        ]

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertFalse(is_chitchat(prompt, is_continuation=True))

    def test_plain_continuation_still_counts_as_chitchat(self):
        self.assertTrue(is_chitchat("im good and you?", is_continuation=True))


if __name__ == "__main__":
    unittest.main()
