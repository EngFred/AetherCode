import re
from pathlib import Path
from typing import List, Tuple

import config
from tools.file_manager import SafeFileManager
from tools.tree_builder import get_filtered_file_list

# Matches path-like tokens: absolute/home paths, or multi-segment relative
# paths, ending in a plausible file extension.
_PATH_PATTERN = re.compile(r'(?:[\/~][\w.\-\/]+|[\w\-]+(?:\/[\w\-]+)+)\.[a-zA-Z0-9]{1,6}')
# Fallback: a bare filename with an extension, no path segments.
_BARE_FILENAME_PATTERN = re.compile(r'\b[\w\-]+\.[a-zA-Z]{1,6}\b')


def find_referenced_files(prompt: str, file_manager: SafeFileManager) -> List[Tuple[str, str]]:
    """
    Heuristically detects file(s) the user explicitly named in their prompt
    and returns (relative_path, content) pairs for the ones that actually
    exist inside the project root. Used to skip the Gemini discovery phase
    when the target file is already known.
    """
    root = file_manager.root_path
    found: dict[str, str] = {}

    for raw in set(_PATH_PATTERN.findall(prompt)):
        rel = raw
        if raw.startswith("/") or raw.startswith("~"):
            try:
                abs_path = Path(raw).expanduser().resolve()
                rel = str(abs_path.relative_to(root))
            except ValueError:
                continue  # path is outside the project, ignore silently

        content = file_manager.read_file(rel)
        if content and not content.startswith("Error"):
            found[rel] = content[:config.MAX_REFERENCED_FILE_CHARS]

        if len(found) >= config.MAX_REFERENCED_FILES:
            break

    # Fallback for bare filenames mentioned without a path (e.g. "fix config.py")
    # — only resolved if the name is unambiguous across the whole project.
    if len(found) < config.MAX_REFERENCED_FILES:
        all_files = get_filtered_file_list(str(root))
        for name in set(_BARE_FILENAME_PATTERN.findall(prompt)):
            matches = [f for f in all_files if f == name or f.endswith("/" + name)]
            if len(matches) == 1 and matches[0] not in found:
                content = file_manager.read_file(matches[0])
                if content and not content.startswith("Error"):
                    found[matches[0]] = content[:config.MAX_REFERENCED_FILE_CHARS]
            if len(found) >= config.MAX_REFERENCED_FILES:
                break

    return list(found.items())