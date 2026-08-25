import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "PLACEHOLDER_GROQ_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PLACEHOLDER_GEMINI_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "PLACEHOLDER_CEREBRAS_KEY")

# --- MODEL DEFINITIONS ---
GROQ_EXECUTOR_MODEL = "openai/gpt-oss-120b"
GEMINI_ANALYZER_MODEL = "gemini-3.5-flash"
# llama-3.3-70b is the strongest free tool-calling model on Cerebras.
# It is fully OpenAI-compatible and resets at 1M tokens / day.
CEREBRAS_EXECUTOR_MODEL = "llama-3.3-70b"

# --- CORS ---
# Comma-separated list in the env var, e.g. "http://localhost:3000,app://."
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# --- FILE SYSTEM SAFETY CONFIGURATION ---
IGNORED_DIRECTORIES = {
    "node_modules",
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "coverage"
}

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
    ".json", ".md", ".env", ".yaml", ".yml", ".sql", ".sh",
    # Flutter/Dart projects — missing .dart was the direct cause of every
    # write_file() call in a Flutter project failing with "Writing files
    # with extension '.dart' is not permitted," no matter what the user
    # approved. The rest of this group covers the native Android/iOS
    # config files a Flutter project typically also needs edited
    # (build.gradle, AndroidManifest.xml, Info.plist, localization .arb
    # files) — trim this list down if you don't want the agent touching
    # native platform config.
    ".dart", ".gradle", ".kt", ".kts", ".swift", ".xml", ".plist", ".arb",
}

# --- SMART ROUTING (Auto mode) ---
MAX_REFERENCED_FILES = 5
MAX_REFERENCED_FILE_CHARS = 12000

# --- CHIT-CHAT DETECTION (Auto mode routing) ---
# Upper bound on message length for the chit-chat shortcut in
# tools/chitchat_utils.is_chitchat. Kept short and conservative — this is
# only meant to catch plain greetings/acknowledgements ("hey", "thanks"),
# never a real request that happens to open politely. Anything longer
# than this just falls through to normal Auto-mode routing, which is
# always correct — just not free.
MAX_CHITCHAT_CHARS = 60

# --- TOOL OUTPUT SAFETY ---
# Hard cap (characters) on any single tool result — read_file,
# run_command, list_project_files — before it's appended to the Groq
# message list. This is the fix for an unfiltered command (e.g. `ls -R`
# walking into a huge build/ tree) or a large file read single-handedly
# blowing past Groq's context window on the very next call in the tool
# loop: previously that surfaced as a 400 context_length_exceeded error
# that burned the whole turn's tokens for nothing. Kept >=
# MAX_REFERENCED_FILE_CHARS so this outer cap never double-truncates a
# file that find_referenced_files() already intentionally pre-loads at
# that budget. See tools/file_manager.SafeFileManager.truncate_output.
MAX_TOOL_OUTPUT_CHARS = 12000

# --- CONVERSATION MEMORY ---
# Keeps follow-up messages continuous while keeping every call token-frugal.
# History is the biggest silent token drain across all three providers.
# Reduced from 12 turns / 16k chars → 8 turns / 12k chars; the window still
# covers the vast majority of real sessions, but burns ~20 % fewer tokens
# per call on long conversations.
MAX_CHAT_HISTORY_TURNS = 8        # how many past (user, assistant) exchanges to resend per call
MAX_CHAT_HISTORY_CHARS = 12000    # rough char budget for that history block; oldest turns drop first
MAX_RECENT_FILES_TRACKED = 8      # filenames (not content) carried forward across turns this session

# --- TIMEOUTS ---
# Per-request timeout enforced by each provider client (seconds).
# A slow or hung call raises inside the existing try/except in the tool
# loops instead of blocking the worker thread indefinitely.
GROQ_REQUEST_TIMEOUT_SECONDS = 60
# Cerebras is typically very fast (WSE hardware), but keep the same ceiling
# so agentic tool-loop turns with many sequential calls are still bounded.
CEREBRAS_REQUEST_TIMEOUT_SECONDS = 60

# Hard ceiling for one full agent.run() turn — covers the Gemini scan AND
# the whole Groq tool loop. This is a connection-level safety net: it
# catches a hang ANYWHERE in the turn (not just a slow Groq call), and is
# what stops the frontend's "isRunning" from getting stuck forever.
#
# NOTE: this only stops the frontend from WAITING past this point — the
# underlying worker thread running agent.run() cannot actually be killed
# once started (Python threads aren't forcibly cancellable). See api.py's
# cancel_event / turn_log_callback / turn_approval_callback for how a
# timed-out turn is kept from doing anything further (more tool calls,
# more approval requests, more state mutation) after this fires.
AGENT_TURN_TIMEOUT_SECONDS = 180