import os
from pathlib import Path
from config import IGNORED_DIRECTORIES

def generate_directory_tree(root_dir: str, max_depth: int = 4) -> str:
    """
    Scans the given directory and returns a clean, indented text representation
    of the folder tree, skipping ignored directories.
    """
    root_path = Path(root_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        return f"Error: Invalid directory path '{root_dir}'"

    tree_lines = [f"Project Root: {root_path.name}/"]

    def _build(current_path: Path, prefix: str = "", current_depth: int = 0):
        if current_depth > max_depth:
            tree_lines.append(f"{prefix}... (depth limit reached)")
            return

        try:
            entries = sorted(list(current_path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return

        # Filter out hidden or ignored directories
        valid_entries = [
            e for e in entries 
            if e.name not in IGNORED_DIRECTORIES and not e.name.startswith(".")
        ]

        count = len(valid_entries)
        for index, entry in enumerate(valid_entries):
            is_last = (index == count - 1)
            connector = "└── " if is_last else "├── "

            if entry.is_dir():
                tree_lines.append(f"{prefix}{connector}{entry.name}/")
                extension_prefix = "    " if is_last else "│   "
                _build(entry, prefix + extension_prefix, current_depth + 1)
            else:
                tree_lines.append(f"{prefix}{connector}{entry.name}")

    _build(root_path)
    return "\n".join(tree_lines)


def get_filtered_file_list(root_dir: str) -> list[str]:
    """
    Returns a list of relative file paths for all valid files in the root directory.
    """
    root_path = Path(root_dir).resolve()
    file_list = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Mutate dirnames in-place to avoid traversing ignored folders
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRECTORIES and not d.startswith(".")]

        for filename in filenames:
            if filename.startswith("."):
                continue
            full_path = Path(dirpath) / filename
            rel_path = full_path.relative_to(root_path)
            file_list.append(str(rel_path))

    return sorted(file_list)