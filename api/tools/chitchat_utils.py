import re

import config

# Any request text gets normalized (trimmed, lowercased, apostrophes
# stripped, trailing punctuation stripped, whitespace collapsed) before
# being checked against this set. Add new greetings/acknowledgements here
# in their normalized form — no apostrophes, no trailing punctuation.
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
}

# Any of these tokens anywhere in the raw message disqualifies it from
# being chit-chat, regardless of how the message opens — a greeting can
# be the first few words of a real task ("hey, can you fix login.py"),
# and that must never get routed through the no-tools bypass.
_CODE_SIGNAL_PATTERN = re.compile(
    r"\.[a-zA-Z0-9]{1,6}\b"  # looks like a filename/extension
    r"|\b(fix|bug|error|implement|refactor|add|create|delete|remove|"
    r"update|run|test|build|debug|install|deploy|write|read|analyze|"
    r"scan|generate|check|review|change|edit)\b",
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
    True only for turns that plainly need no project context at all —
    greetings, thanks, acknowledgements. Used by AetherAgent.run() to skip
    the Gemini directory scan + full Groq tool loop for a linked project
    when the message is obviously small talk, instead of running the
    whole pipeline just to produce "Hey! How can I help?".

    Deliberately conservative in both directions:
    - length-capped and (for a NEW small-talk exchange) exact-phrase-
      matched, so it only fires on messages that plainly ARE one of
      these things, not ones that merely start with one;
    - any file-extension-looking token or task verb anywhere in the
      message disqualifies it outright, always — this check runs
      regardless of is_continuation.

    is_continuation: pass True when the immediately preceding turn in
    this session was ALSO routed through this bypass. A reply that's
    plainly continuing an already-established small-talk exchange
    (e.g. "im good and you?" replying to the agent's own "how's your
    day going?") won't ever match something from the fixed
    _CHITCHAT_PHRASES list, and no fixed list ever could — there's no
    way to enumerate every phrasing of a conversational reply. Once a
    continuation, it only needs to still pass the length cap and the
    code-signal check to qualify; exact-phrase matching is reserved for
    OPENING a chit-chat exchange, where nothing else yet confirms intent.

    A false negative here just costs an extra scan (safe). A false
    positive would silently skip a real request (not safe) — so the bar
    to return True is high on purpose.
    """
    raw = prompt.strip()
    if not raw or len(raw) > config.MAX_CHITCHAT_CHARS:
        return False
    if _CODE_SIGNAL_PATTERN.search(raw):
        return False
    if is_continuation:
        return True
    return _normalize(raw) in _CHITCHAT_PHRASES