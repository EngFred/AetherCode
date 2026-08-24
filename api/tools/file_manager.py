import os
import shutil
import subprocess
from pathlib import Path
import config

class SafeFileManager:
    """
    Manages disk operations with security checks, terminal commands,
    and automatic pre-edit file backups for undo operations.
    """

    def __init__(self, root_dir: str):
        self.root_path = Path(root_dir).resolve()
        if not self.root_path.exists() or not self.root_path.is_dir():
            raise ValueError(f"Invalid root directory: {root_dir}")
        self.backup_dir = self.root_path / ".aether_backups"
        self.history = []  # Stack of (backup_path, relative_original_path)

    def _resolve_safe_path(self, relative_path: str) -> Path:
        target_path = (self.root_path / relative_path).resolve()
        # relative_to() raises ValueError if target_path isn't actually inside
        # root_path — a plain startswith() check would wrongly allow a sibling
        # directory like "project_evil" when root is "project".
        try:
            target_path.relative_to(self.root_path)
        except ValueError:
            raise PermissionError(f"Access denied: Path '{relative_path}' attempts to leave project root.")
        return target_path

    def _create_backup(self, relative_path: str):
        """Creates a hidden backup before any file edit or deletion."""
        try:
            target_path = self._resolve_safe_path(relative_path)
            if target_path.exists() and target_path.is_file():
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                backup_filename = f"{target_path.name}.bak_{len(self.history)}"
                backup_path = self.backup_dir / backup_filename
                shutil.copy2(target_path, backup_path)
                self.history.append((backup_path, relative_path))
        except Exception as e:
            print(f"Warning: Failed to create backup for {relative_path}: {e}")

    def undo_last_change(self) -> str:
        """Restores the most recently modified or deleted file."""
        if not self.history:
            return "No recent changes available to undo."

        backup_path, rel_path = self.history.pop()
        try:
            target_path = self._resolve_safe_path(rel_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, target_path)
            if backup_path.exists():
                os.remove(backup_path)
            return f"Successfully restored '{rel_path}' to its previous state."
        except Exception as e:
            return f"Error restoring backup for '{rel_path}': {str(e)}"

    def read_file(self, relative_path: str) -> str:
        try:
            target_path = self._resolve_safe_path(relative_path)
            if not target_path.exists():
                return f"Error: File '{relative_path}' does not exist."
            if not target_path.is_file():
                return f"Error: Path '{relative_path}' is a directory, not a file."

            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except PermissionError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error reading file '{relative_path}': {str(e)}"

    def write_file(self, relative_path: str, content: str) -> str:
        try:
            ext = Path(relative_path).suffix
            if ext and config.ALLOWED_EXTENSIONS and ext not in config.ALLOWED_EXTENSIONS:
                return f"Error: Writing files with extension '{ext}' is not permitted."

            self._create_backup(relative_path)
            target_path = self._resolve_safe_path(relative_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully saved file: {relative_path}"
        except PermissionError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error writing to file '{relative_path}': {str(e)}"

    def delete_file(self, relative_path: str) -> str:
        try:
            self._create_backup(relative_path)
            target_path = self._resolve_safe_path(relative_path)
            if not target_path.exists():
                return f"Error: File '{relative_path}' does not exist."

            os.remove(target_path)
            return f"Successfully deleted file: {relative_path}"
        except PermissionError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error deleting file '{relative_path}': {str(e)}"

    def run_command(self, command: str) -> str:
        """Executes terminal commands inside the project directory."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.root_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            output = result.stdout or result.stderr
            return output if output.strip() else "Command executed with no output."
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error executing command: {str(e)}"