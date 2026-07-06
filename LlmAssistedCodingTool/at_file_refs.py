# =============================================================================
# @file.py mentions: load project files into LLM context (under selected root)
# =============================================================================
# Use in chat: "Build 55.py like @10.py and @src/helpers.py". Paths are relative
# to the folder selected in the file browser. No ".." — stays inside the project.
# =============================================================================

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

_AT_PY = re.compile(r"@([A-Za-z0-9_./\\-]+\.py)\b")

# Per-file cap (characters); very large files are truncated with a notice.
MAX_CHARS_PER_FILE = 120_000
# Total injected block size (approximate) to avoid blowing context.
MAX_BLOCK_CHARS = 100_000


def iter_at_py_relpaths(text: str) -> List[str]:
    """Unique order-preserving list of `path/to/file.py` from `@path` mentions."""
    seen = set()
    out: List[str] = []
    for m in _AT_PY.finditer(text or ""):
        raw = m.group(1).replace("\\", "/").strip()
        if not raw:
            continue
        parts = [p for p in raw.split("/") if p]
        if not parts or any(p == ".." for p in parts):
            continue
        rel = "/".join(parts)
        if rel not in seen:
            seen.add(rel)
            out.append(rel)
    return out


def strip_at_mentions_for_query(text: str) -> str:
    """Remove @file.py tokens for snippet search / optional web query (cleaner embeddings)."""
    s = _AT_PY.sub(" ", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _read_py_resolved(path: Path) -> Tuple[Optional[str], str]:
    """Read file text or return (None, error note)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return None, str(e)
    if len(raw) > MAX_CHARS_PER_FILE:
        raw = raw[:MAX_CHARS_PER_FILE] + "\n\n# … [truncated: file exceeds per-file cap]\n"
    return raw, ""


def load_at_files_context(user_text: str, root_folder: str) -> Tuple[str, List[str]]:
    """
    Build a Markdown block of referenced `.py` files for the system prompt.

    Returns (markdown_block, warning_messages). Block is empty if there are no @mentions.
    """
    rels = iter_at_py_relpaths(user_text)
    if not rels:
        return "", []

    root = Path(root_folder).resolve()
    if not root.is_dir():
        return "", ["Project folder is not set or invalid — cannot load @ files."]

    warnings: List[str] = []
    sections: List[str] = [
        "## Referenced project files (@mentions)\n",
        "The user cited these paths **relative to the selected project folder**. "
        "Treat them as **examples** for patterns, structure, and style when fulfilling the request.\n",
    ]
    total = sum(len(s) for s in sections)
    loaded_any = False

    for rel in rels:
        if total >= MAX_BLOCK_CHARS:
            warnings.append(f"Stopped after context limit — not loaded: `@{rel}` (and any further @ mentions).")
            break

        try:
            full = (root / Path(rel)).resolve()
        except OSError as e:
            warnings.append(f"`@{rel}`: invalid path ({e}).")
            continue

        try:
            full.relative_to(root)
        except ValueError:
            warnings.append(f"`@{rel}`: path escapes project folder (not allowed).")
            continue

        if not full.is_file():
            warnings.append(f"`@{rel}`: file not found under project root.")
            continue

        if full.suffix.lower() != ".py":
            warnings.append(f"`@{rel}`: only `.py` files are supported.")
            continue

        content, err = _read_py_resolved(full)
        if content is None:
            warnings.append(f"`@{rel}`: could not read ({err}).")
            continue

        header = f"\n### `{rel}`\n\n```python\n"
        footer = "\n```\n"
        block = header + content + footer
        if total + len(block) > MAX_BLOCK_CHARS:
            warnings.append(f"`@{rel}`: skipped — would exceed total @-context budget.")
            break

        sections.append(block)
        total += len(block)
        loaded_any = True

    if not loaded_any:
        return "", warnings

    return "".join(sections).strip() + "\n", warnings


def format_at_referenced_files_for_system(user_text: str, root_folder: str) -> str:
    """Convenience: Markdown block only (no warnings)."""
    block, _ = load_at_files_context(user_text, root_folder)
    return block
