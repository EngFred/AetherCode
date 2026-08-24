import os
import json
import threading
from typing import Optional
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from core.agent import AetherAgent
import config

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

SETTINGS_FILE = "settings.json"

class AetherApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AetherCode - Local AI Development Agent")
        self.geometry("920x720")

        self.working_dir: Optional[str] = None
        self.agent: Optional[AetherAgent] = None

        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Top Bar: Directory Selection & Undo
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")

        self.btn_select_dir = ctk.CTkButton(
            self.top_frame, text="Select Project Folder", command=self.select_directory
        )
        self.btn_select_dir.pack(side="left", padx=10, pady=10)

        self.lbl_dir = ctk.CTkLabel(
            self.top_frame, text="No folder selected (General Chat Mode)", text_color="gray"
        )
        self.lbl_dir.pack(side="left", padx=10, pady=10)

        self.btn_undo = ctk.CTkButton(
            self.top_frame, 
            text="Undo Last File Change", 
            fg_color="#D9534F", 
            hover_color="#C9302C", 
            command=self.undo_change
        )
        self.btn_undo.pack(side="right", padx=10, pady=10)

        # Hidden/Disabled Keys Bar (Keys loaded automatically from .env)
        self.key_frame = ctk.CTkFrame(self)
        self.key_frame.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        self.key_frame.grid_columnconfigure(1, weight=1)
        self.key_frame.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(self.key_frame, text="Groq Key:").grid(row=0, column=0, padx=5, pady=5)
        self.entry_groq_key = ctk.CTkEntry(self.key_frame, placeholder_text="Loaded from .env", show="*")
        self.entry_groq_key.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        if config.GROQ_API_KEY != "PLACEHOLDER_GROQ_KEY":
            self.entry_groq_key.insert(0, config.GROQ_API_KEY)

        ctk.CTkLabel(self.key_frame, text="Gemini Key:").grid(row=0, column=2, padx=5, pady=5)
        self.entry_gemini_key = ctk.CTkEntry(self.key_frame, placeholder_text="Loaded from .env", show="*")
        self.entry_gemini_key.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        if config.GEMINI_API_KEY != "PLACEHOLDER_GEMINI_KEY":
            self.entry_gemini_key.insert(0, config.GEMINI_API_KEY)

        # Output Terminal Logs
        self.txt_logs = ctk.CTkTextbox(self, state="disabled", wrap="word", font=("Courier", 13))
        self.txt_logs.grid(row=2, column=0, padx=15, pady=10, sticky="nsew")

        # Bottom Input Bar
        self.bottom_frame = ctk.CTkFrame(self)
        self.bottom_frame.grid(row=3, column=0, padx=15, pady=(5, 15), sticky="ew")
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self.entry_prompt = ctk.CTkEntry(
            self.bottom_frame, placeholder_text="Ask a question or request code changes..."
        )
        self.entry_prompt.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        self.btn_run = ctk.CTkButton(self.bottom_frame, text="Run Agent", command=self.start_agent_thread)
        self.btn_run.grid(row=0, column=1, padx=(5, 10), pady=10)

    def _load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r") as f:
                    data = json.load(f)
                    last_dir = data.get("last_working_dir")
                    if last_dir and os.path.exists(last_dir):
                        self.working_dir = last_dir
                        self.lbl_dir.configure(text=last_dir, text_color="white")
                        self.log_message(f"📁 Auto-loaded last project: {last_dir}")
            except Exception as e:
                pass

    def _save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump({"last_working_dir": self.working_dir}, f)
        except Exception:
            pass

    def select_directory(self):
        folder = filedialog.askdirectory()
        if folder:
            self.working_dir = folder
            self.lbl_dir.configure(text=folder, text_color="white")
            self._save_settings()
            self.log_message(f"📁 Working directory set to: {folder}")

    def log_message(self, text: str):
        self.txt_logs.configure(state="normal")
        self.txt_logs.insert("end", text + "\n")
        self.txt_logs.see("end")
        self.txt_logs.configure(state="disabled")

    def prompt_command_approval(self, command: str) -> bool:
        return messagebox.askyesno(
            "Terminal Command Approval Requested",
            f"The AI Agent requests to run terminal command:\n\n{command}\n\nDo you approve execution?",
            icon="warning"
        )

    def undo_change(self):
        if not self.agent:
            self.log_message("❌ No active agent session to undo.")
            return
        res = self.agent.undo()
        self.log_message(f"⏪ Undo Result: {res}")

    def start_agent_thread(self):
        user_prompt = self.entry_prompt.get().strip()
        if not user_prompt:
            return

        groq_key = self.entry_groq_key.get().strip()
        gemini_key = self.entry_gemini_key.get().strip()

        # Fallback to current directory if no folder selected
        target_dir = self.working_dir or os.getcwd()

        self.btn_run.configure(state="disabled")
        self.entry_prompt.delete(0, "end")
        self.log_message(f"\n🚀 User Task: {user_prompt}")

        threading.Thread(
            target=self._run_agent_task,
            args=(user_prompt, target_dir, gemini_key, groq_key),
            daemon=True
        ).start()

    def _run_agent_task(self, prompt: str, target_dir: str, gemini_key: str, groq_key: str):
        try:
            self.agent = AetherAgent(
                root_dir=target_dir,
                gemini_key=gemini_key,
                groq_key=groq_key
            )
            self.agent.run(
                user_prompt=prompt,
                log_callback=self.log_message,
                command_approval_callback=self.prompt_command_approval
            )
        except Exception as e:
            self.log_message(f"❌ Application Error: {str(e)}")
        finally:
            self.btn_run.configure(state="normal")