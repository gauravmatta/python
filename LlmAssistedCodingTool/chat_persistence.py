# =============================================================================
# Per-file chat archive: JSON in project/data/chats/<stem>.json + rolling summary
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

_log = logging.getLogger("chat_persistence")

SCHEMA_VERSION = 1
# Relative to root_folder (selected project directory).
CHATS_DIR = Path("data") / "chats"

# When rolling_summary + sum(message content) exceeds this, compress older turns.
# Count is text only (not JSON file size). Aligned with app MAX_CONTEXT_CHARS (~50k) so
# the archive does not grow far beyond what fits in one context window.
SUMMARIZE_TOTAL_CHAR_THRESHOLD = 50_000
# Verbatim messages to keep after each summarize pass (may be reduced if total messages
# are fewer — see maybe_compress_history).
KEEP_LAST_MESSAGES = 14
# Hard cap for stored rolling summary (characters). Keeps JSON + system prompt lean.
MAX_ROLLING_SUMMARY_CHARS = 12_000
# Max chars of older-turn text sent into one summarize call (input only).
SUMMARIZE_INPUT_BLOCK_CHARS = 32_000
# Ollama max new tokens for summarize (high enough for thinking models to still emit text).
SUMMARY_NUM_PREDICT = 2500
# Short vision caption for archive (no image bytes in JSON).
ARCHIVE_IMAGE_CAPTION_TIMEOUT = 90.0
SUMMARIZE_TIMEOUT = 180.0

SUMMARY_SYSTEM = """You merge older coding-chat turns into a compact memory note for the assistant.

Output Markdown with exactly these sections (headings required):
## Goals
## Decisions / code
## Errors
## Open
## Files

Rules:
- **Critical points only** — short bullets (one line each). No paragraphs, no narration.
- **No code blocks** — name symbols/files and intent in a few words (e.g. "add retry in `fetch()`").
- **No quoting** long assistant explanations; distill to facts the next turn must remember.
- If a section has nothing useful, write a single bullet: `None`.
- Entire output must stay **under ~900 words**; prefer fewer."""


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def stem_from_py_path(file_path: Optional[str]) -> Optional[str]:
    if not file_path:
        return None
    low = str(file_path).lower()
    if not low.endswith(".py"):
        return None
    return Path(file_path).stem


def archive_json_path(root_folder: str, stem: str) -> Path:
    return Path(root_folder).resolve() / CHATS_DIR / f"{stem}.json"


def messages_for_disk(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip images; keep text + optional sources for assistant."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        row: Dict[str, Any] = {"role": role, "content": (m.get("content") or "").strip()}
        if role == "assistant" and m.get("sources"):
            row["sources"] = m["sources"]
        out.append(row)
    return out


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(text, encoding="utf-8", errors="replace")
    tmp.replace(path)


def load_archive(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        bak = path.parent / (
            f"{path.stem}.json.bak.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        )
        try:
            path.rename(bak)
            _log.warning("Corrupt chat archive moved to %s", bak)
        except OSError:
            _log.exception("Could not rename corrupt archive")
        return None
    except OSError as e:
        _log.warning("Could not read chat archive: %s", e)
        return None


def save_chat_from_session(
    session_state: Any,
    root_folder: str,
    current_file: Optional[str],
) -> None:
    stem = stem_from_py_path(current_file)
    if not stem or not (root_folder or "").strip():
        return
    path = archive_json_path(root_folder, stem)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project_root": str(Path(root_folder).resolve()),
        "bound_file": os.path.basename(current_file or ""),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rolling_summary": session_state.get("rolling_summary") or "",
        "messages": messages_for_disk(session_state.get("messages") or []),
    }
    try:
        atomic_write_json(path, payload)
        session_state["metric_chat_disk_saves"] = (
            int(session_state.get("metric_chat_disk_saves", 0)) + 1
        )
    except OSError as e:
        _log.warning("Chat save failed: %s", e)


def load_chat_into_session(
    session_state: Any,
    root_folder: str,
    current_file: Optional[str],
) -> None:
    stem = stem_from_py_path(current_file)
    if not stem or not (root_folder or "").strip():
        session_state["messages"] = []
        session_state["rolling_summary"] = ""
        session_state["_chat_archive_stem"] = None
        return
    path = archive_json_path(root_folder, stem)
    data = load_archive(path)
    session_state["_chat_archive_stem"] = stem
    if not data:
        session_state["rolling_summary"] = ""
        session_state["messages"] = []
        return
    session_state["rolling_summary"] = str(data.get("rolling_summary") or "")
    raw_msgs = data.get("messages") or []
    clean: List[Dict[str, Any]] = []
    for m in raw_msgs:
        if not isinstance(m, dict):
            continue
        r = m.get("role")
        if r not in ("user", "assistant"):
            continue
        clean.append(
            {
                "role": r,
                "content": str(m.get("content") or ""),
                **({"sources": m["sources"]} if m.get("sources") else {}),
            }
        )
    session_state["messages"] = clean


def append_summary_to_system(system_msg: str, rolling_summary: str) -> str:
    rs = (rolling_summary or "").strip()
    if not rs:
        return system_msg
    return (
        system_msg
        + "\n\n## Prior conversation summary (this .py file)\n"
        + rs
    )


def transcript_char_count(rolling_summary: str, messages: List[Dict[str, Any]]) -> int:
    n = len(rolling_summary or "")
    for m in messages:
        n += len(m.get("content") or "")
    return n


def archive_bulk_char_estimate(rolling_summary: str, messages: List[Dict[str, Any]]) -> int:
    """
    Approximate size of the chat portion of `data/chats/*.json` on disk (indent + keys + sources).
    Editors show file bytes; threshold uses this so ~90k files still trigger compress.
    """
    try:
        core = json.dumps(
            {
                "rolling_summary": rolling_summary or "",
                "messages": messages_for_disk(messages),
            },
            ensure_ascii=False,
            indent=2,
        )
        # Real file also has schema_version, project_root, bound_file, updated_at (~200–2k chars).
        return len(core) + 800
    except (TypeError, ValueError):
        return transcript_char_count(rolling_summary, messages) + 10_000


def compress_budget_chars(rolling_summary: str, messages: List[Dict[str, Any]]) -> int:
    """Largest of plain-text transcript vs. on-disk-style bulk (user sees the latter)."""
    return max(
        transcript_char_count(rolling_summary, messages),
        archive_bulk_char_estimate(rolling_summary, messages),
    )


def _trim_oldest_pairs_fallback(session_state: Any, target_chars: int) -> None:
    """Drop oldest user/assistant pairs until under target (or min 4 messages left)."""
    msgs_fb: List[Dict[str, Any]] = list(session_state.get("messages") or [])
    rs = session_state.get("rolling_summary") or ""
    while (
        len(msgs_fb) > 4
        and compress_budget_chars(rs, msgs_fb) > target_chars
    ):
        msgs_fb = msgs_fb[2:]
    session_state["messages"] = msgs_fb
    session_state["metric_chat_trim_fallback"] = (
        int(session_state.get("metric_chat_trim_fallback", 0)) + 1
    )


def ollama_chat_sync(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    timeout: float = 120.0,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content", "") or ""


def maybe_compress_history(
    session_state: Any,
    model: Optional[str],
    base_url: str,
) -> bool:
    """
    If transcript is large, summarize oldest messages into rolling_summary.
    Returns True if state was mutated.
    """
    if not model:
        return False
    rs = session_state.get("rolling_summary") or ""
    msgs: List[Dict[str, Any]] = list(session_state.get("messages") or [])
    total = compress_budget_chars(rs, msgs)
    if total < SUMMARIZE_TOTAL_CHAR_THRESHOLD:
        return False
    # Need at least two messages so we fold at least one into the summary.
    if len(msgs) < 2:
        return False
    # If there are only a few turns but each is huge, still summarize (old code required
    # len(msgs) > KEEP_LAST_MESSAGES, so e.g. 5×30k chars never compressed).
    keep = min(KEEP_LAST_MESSAGES, len(msgs) - 1)

    old = msgs[:-keep]
    recent = msgs[-keep:]
    lines = []
    for m in old:
        role = m.get("role", "")
        content = (m.get("content") or "")[:8000]
        lines.append(f"### {role}\n{content}\n")
    block = "\n".join(lines)[:SUMMARIZE_INPUT_BLOCK_CHARS]
    prior = (rs or "")[:12000]

    user_body = (
        "### Existing summary (may be empty) — compress further if verbose\n"
        + prior
        + "\n\n### Older messages to fold in\n"
        + block
        + "\n\nReturn **one** updated summary in the required section format. "
        "Replace fluffy prior text with tight bullets. Drop repetition and side tangents. "
        "Do not paste code; keep only what changes the next coding steps."
    )
    api_msgs = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": user_body[:28000]},
    ]
    try:
        out = strip_think_tags(
            ollama_chat_sync(
                base_url,
                model,
                api_msgs,
                timeout=SUMMARIZE_TIMEOUT,
                options={"num_predict": SUMMARY_NUM_PREDICT},
            )
        )
        if not out.strip():
            _log.warning(
                "Chat summarization returned empty after strip (model/thinking or num_predict); "
                "trimming oldest turns instead."
            )
            _trim_oldest_pairs_fallback(
                session_state,
                int(SUMMARIZE_TOTAL_CHAR_THRESHOLD * 0.75),
            )
            return True
        merged = ((rs + "\n\n---\n\n") if rs.strip() else "") + out.strip()
        if len(merged) > MAX_ROLLING_SUMMARY_CHARS:
            merged = merged[:MAX_ROLLING_SUMMARY_CHARS] + "\n\n…(summary truncated)"
        session_state["rolling_summary"] = merged
        session_state["messages"] = recent
        session_state["metric_chat_summaries"] = (
            int(session_state.get("metric_chat_summaries", 0)) + 1
        )
        return True
    except Exception as e:
        _log.warning("Chat summarization failed: %s", e)
        _trim_oldest_pairs_fallback(
            session_state,
            int(SUMMARIZE_TOTAL_CHAR_THRESHOLD * 0.75),
        )
        return True


def brief_image_archive_note(
    model: Optional[str],
    base_url: str,
    images: List[Dict[str, str]],
    user_text: str,
) -> str:
    """Text-only note appended to user content for JSON archive (no image bytes)."""
    if not images:
        return ""
    n = len(images)
    b64_list: List[str] = []
    for im in images:
        if isinstance(im, dict) and im.get("b64"):
            b64_list.append(str(im["b64"]))
    if not b64_list:
        return f"\n\n_[Archive: {n} image(s) attached; pixels not stored.]_"

    if not model:
        return f"\n\n_[Archive: {n} image(s) attached; pixels not stored.]_"

    sys_msg = (
        "In one or two short sentences, describe what is visible in the image(s) "
        "that matters for coding or debugging (UI, error text, code). "
        "No preamble."
    )
    user_msg: Dict[str, Any] = {
        "role": "user",
        "content": (user_text or "").strip() or "(User sent only images.)",
        "images": b64_list[:8],
    }
    try:
        raw = ollama_chat_sync(
            base_url,
            model,
            [{"role": "system", "content": sys_msg}, user_msg],
            timeout=ARCHIVE_IMAGE_CAPTION_TIMEOUT,
        )
        note = strip_think_tags(raw).strip()
        if not note:
            raise ValueError("empty caption")
        return f"\n\n_[Image archive note: {note}]_"
    except Exception as e:
        _log.debug("Archive image caption skipped: %s", e)
        return f"\n\n_[Archive: {n} image(s); brief caption unavailable.]_"


def chat_status_line(
    root_folder: str,
    current_file: Optional[str],
) -> str:
    stem = stem_from_py_path(current_file)
    if not stem:
        return "💬 **Chat log:** select a `.py` file to enable saved history (`data/chats/<name>.json`)."
    p = archive_json_path(root_folder, stem)
    exists = p.is_file()
    return (
        f"💬 **Chat log:** `{CHATS_DIR.as_posix()}/{stem}.json` "
        f"({'on disk' if exists else 'will be created on save'})"
    )


def clear_archive_file(root_folder: str, current_file: Optional[str]) -> None:
    stem = stem_from_py_path(current_file)
    if not stem or not root_folder:
        return
    path = archive_json_path(root_folder, stem)
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        _log.warning("Could not delete chat archive %s", path)
