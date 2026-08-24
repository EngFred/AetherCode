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

# --- CONVERSATION MEMORY ---
# Keeps follow-up messages actually continuous, while keeping every call
# token-frugal enough not to eat into free-tier daily limits.
MAX_CHAT_HISTORY_TURNS = 12       # how many past (user, assistant) exchanges to resend per call
MAX_CHAT_HISTORY_CHARS = 16000    # rough char budget for that history block; oldest turns drop first
MAX_RECENT_FILES_TRACKED = 8      # filenames (not content) carried forward across turns this session