import re
import config

# Any request text gets normalized (trimmed, lowercased, apostrophes
# stripped, trailing punctuation stripped, whitespace collapsed) before
# being checked against this set.
_CHITCHAT_PHRASES = {
    "hi", "hey", "hello", "yo", "sup", "howdy", "hiya", "heya",
    "hi there", "hey there", "hello there",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "how are you doing today",
    "hows it going", "how is it going",
    "whats up", "sup man", "how you doing", "how you doing today",
    "thanks", "thank you", "thx", "ty", "cheers", "thanks a lot", "much appreciated",
    "ok", "okay", "cool", "nice", "great", "awesome", "got it",
    "sounds good", "perfect", "nice one", "makes sense",
    "bye", "goodbye", "see ya", "see you", "later",
    # Common conversational replies:
    "im good", "i am good", "im fine", "i am fine", "im ok", "i am ok",
    "doing good", "doing well", "doing fine", "doing great",
    "all good", "pretty good", "not bad", "not much", "nothing much",
    "good you", "good and you", "im good you", "im good and you",
    "you too", "same here", "same to you", "no worries", "no problem",
}

# Any of these tokens anywhere in the message immediately disqualifies it from
# being chit-chat — even if it opens with a greeting like "hey can you fix..."
_TASK_OR_PROJECT_SIGNAL_PATTERN = re.compile(
    r"\.[a-zA-Z0-9]{1,6}\b"  # looks like a filename/extension (e.g. .py, .ts, .dart)
    r"|\b(fix|bug|error|implement|refactor|add|create|delete|remove|"
    r"update|run|test|build|debug|install|deploy|write|read|analyze|"
    r"scan|generate|check|review|change|edit|git|stage|commit|push|"
    r"pull|merge|branch|checkout|status|diff|stash|rebase|tag|publish|"
    r"project|code|files?|folders?|dirs?|directory|contents?|list|show|"
    r"inspect)\b",
    re.IGNORECASE,
)

_TRAILING_PUNCT_RE = re.compile(r"[?!.,\s]+$")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _TRAILING_PUNCT_RE.sub("", text)
    text = text.replace("'", "")
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def is_chitchat(prompt: str, is_continuation: bool = False) -> bool:
    """
    Returns True ONLY for pure small talk / greetings / acknowledgements.
    
    Any request containing project keywords, file inquiries, task verbs,
    or questions immediately routes through to the full agent pipeline.
    """
    raw = prompt.strip()
    if not raw or len(raw) > config.MAX_CHITCHAT_CHARS:
        return False

    # Disqualify if it has any coding or project task signal (e.g. "hey fix main.py")
    if _TASK_OR_PROJECT_SIGNAL_PATTERN.search(raw):
        return False

    norm = _normalize(raw)
    return norm in _CHITCHAT_PHRASES
