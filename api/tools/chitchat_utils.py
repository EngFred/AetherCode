import re
import config

# Set of purely conversational words. Any short phrase composed entirely
# of these words (e.g. "okay good", "sounds good", "alright thanks bro", "got it")
# is recognized as small talk / acknowledgement and bypasses the Gemini scan.
_CONVERSATIONAL_WORDS = {
    "hi", "hey", "hello", "yo", "sup", "howdy", "hiya", "heya",
    "good", "morning", "afternoon", "evening", "night", "day",
    "how", "are", "you", "doing", "today", "hows", "is", "it", "going",
    "whats", "up", "thanks", "thank", "thx", "ty", "cheers", "much", "appreciated",
    "ok", "okay", "cool", "nice", "great", "awesome", "got", "perfect", "one",
    "makes", "sense", "sounds", "looks", "bye", "goodbye", "see", "ya", "later",
    "im", "i", "am", "fine", "not", "bad", "all", "pretty", "nothing",
    "too", "same", "here", "no", "worries", "problem", "yeah", "yes", "yep", "yup",
    "sure", "alright", "job", "work", "done", "well", "dude", "bro", "man", "mate",
    "so", "far", "then", "indeed", "understood", "right"
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
    r"inspect|why|where|how to|help me)\b",
    re.IGNORECASE,
)


def is_chitchat(prompt: str, is_continuation: bool = False) -> bool:
    """
    Returns True for small talk, greetings, feedback, or acknowledgements
    (e.g. "hey", "okay good", "sounds good", "thanks man", "got it").
    
    Any request containing project keywords, file inquiries, or task verbs
    immediately returns False and routes to the full agent pipeline.
    """
    raw = prompt.strip()
    if not raw or len(raw) > config.MAX_CHITCHAT_CHARS:
        return False

    # Disqualify if it has any coding or project task signal
    if _TASK_OR_PROJECT_SIGNAL_PATTERN.search(raw):
        return False

    # Extract individual words
    words = re.findall(r"[a-zA-Z0-9]+", raw.lower())
    if not words:
        return False

    # If it is a short message (<= 5 words) composed purely of conversational words
    if len(words) <= 5 and all(w in _CONVERSATIONAL_WORDS for w in words):
        return True

    return False
