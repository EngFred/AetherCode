import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- API KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "PLACEHOLDER_GROQ_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PLACEHOLDER_GEMINI_KEY")

# --- MODEL DEFINITIONS ---
GROQ_EXECUTOR_MODEL = "openai/gpt-oss-120b"
GEMINI_ANALYZER_MODEL = "gemini-3.5-flash"

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
    ".json", ".md", ".env", ".yaml", ".yml", ".sql", ".sh"
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
# Keeps follow-up messages actually continuous, while keeping every call
# token-frugal enough not to eat into free-tier daily limits.
MAX_CHAT_HISTORY_TURNS = 12       # how many past (user, assistant) exchanges to resend per call
MAX_CHAT_HISTORY_CHARS = 16000    # rough char budget for that history block; oldest turns drop first
MAX_RECENT_FILES_TRACKED = 8      # filenames (not content) carried forward across turns this session

# --- TIMEOUTS ---
# Per-request timeout enforced by the Groq client itself (seconds). A slow
# or hung Groq call now raises inside the existing try/except in
# _run_groq_tool_loop / the general-chat path, instead of blocking the
# worker thread indefinitely.
GROQ_REQUEST_TIMEOUT_SECONDS = 60

# Hard ceiling for one full agent.run() turn — covers the Gemini scan AND
# the whole Groq tool loop. This is a connection-level safety net: it
# catches a hang ANYWHERE in the turn (not just a slow Groq call), and is
# what stops the frontend's "isRunning" from getting stuck forever. See
# api.py for the caveat about it not being able to kill the underlying
# thread.
AGENT_TURN_TIMEOUT_SECONDS = 180.