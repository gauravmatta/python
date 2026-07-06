# =============================================================================
# Persist UI settings + project folder + current file path across Streamlit restarts
#
# apply_loaded_settings() must run **once per browser session** (47.py sets
# `_ui_settings_hydrated`). Re-applying on every Streamlit rerun would reload JSON
# and undo widget changes before save_settings runs at end of script.
# =============================================================================

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SETTINGS_PATH = Path(__file__).resolve().parent / "data" / "app_ui_settings.json"
SCHEMA_VERSION = 1

# Keys saved to JSON (must be JSON-serializable). Not: messages, metrics, drafts, shell state.
PERSIST_KEYS = (
    "root_folder",
    "current_file",
    "sel_model",
    "prompt_mode",
    "custom_prompt",
    "code_theme",
    "research_mode",
    "research_deep",
    "research_llm_planner",
    "planning_mode",
    "planner_snack_research",
    "snippet_retrieval_enabled",
    "snippet_top_k",
    "ollama_temperature",
    "ollama_top_p",
    "font_size",
    "_show_all",
    "_recursive",
)


def _ensure_data_dir() -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, text: str) -> None:
    _ensure_data_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", errors="replace")
    tmp.replace(path)


def apply_loaded_settings(session_state: Any, *, app_dir: str) -> None:
    """
    Merge JSON from disk into session_state (after defaults). Validates paths.
    `app_dir` is the directory containing 47.py (default root_folder fallback).
    """
    if not SETTINGS_PATH.is_file():
        return
    try:
        raw = SETTINGS_PATH.read_text(encoding="utf-8", errors="replace")
        data: Dict[str, Any] = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return

    if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
        return

    # root_folder
    rf = data.get("root_folder")
    if isinstance(rf, str) and rf.strip():
        try:
            r = Path(rf).expanduser().resolve()
            if r.is_dir():
                session_state["root_folder"] = str(r)
        except OSError:
            pass

    root = str(session_state.get("root_folder") or app_dir)

    for k in PERSIST_KEYS:
        if k in ("root_folder", "current_file"):
            continue
        if k not in data:
            continue
        val = data[k]
        try:
            if k in (
                "research_deep",
                "research_llm_planner",
                "planner_snack_research",
                "snippet_retrieval_enabled",
                "_show_all",
                "_recursive",
            ):
                session_state[k] = bool(val)
            elif k == "snippet_top_k":
                session_state[k] = max(1, min(20, int(val)))
            elif k in ("ollama_temperature", "ollama_top_p"):
                session_state[k] = float(val)
            elif k == "font_size":
                session_state[k] = max(10, min(28, int(val)))
            elif k == "custom_prompt":
                session_state[k] = str(val) if val is not None else ""
            elif k in ("research_mode", "planning_mode"):
                if val in ("Off", "Auto", "Always"):
                    session_state[k] = val
            elif k == "code_theme":
                themes = ("monokai", "native", "friendly", "vs", "dracula")
                if val in themes:
                    session_state[k] = val
            elif k == "prompt_mode":
                modes = ("Default", "Teacher", "Senior Dev")
                if val in modes:
                    session_state[k] = val
            else:
                session_state[k] = val
        except (TypeError, ValueError):
            continue

    # current_file + content (after root; may force _recursive for nested paths)
    cf = data.get("current_file")
    if isinstance(cf, str) and cf.strip():
        try:
            fp = Path(cf).expanduser().resolve()
            root_p = Path(root).resolve()
            fp.relative_to(root_p)
            if fp.is_file():
                session_state["current_file"] = str(fp)
                session_state["current_content"] = fp.read_text(
                    encoding="utf-8", errors="replace"
                )
                if fp.parent != root_p:
                    session_state["_recursive"] = True
        except ValueError:
            pass
        except OSError:
            pass


def save_settings(session_state: Any) -> None:
    """Write whitelisted keys + metadata to disk."""
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for k in PERSIST_KEYS:
        if k not in session_state:
            continue
        try:
            payload[k] = session_state[k]
        except Exception:
            continue

    try:
        _atomic_write(SETTINGS_PATH, json.dumps(payload, ensure_ascii=False, indent=2))
    except OSError:
        pass


def fix_model_selection(session_state: Any, model_names: list) -> None:
    """If saved model is missing from Ollama, fall back to first tag."""
    if not model_names:
        return
    cur = session_state.get("sel_model")
    if cur not in model_names:
        session_state["sel_model"] = model_names[0]
