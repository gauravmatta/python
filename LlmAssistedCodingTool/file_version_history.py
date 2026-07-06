# =============================================================================
# Per-file version ring (last N snapshots before each save) — persisted under data/
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger("file_version_history")

VERSIONS_PER_FILE = 10
PREVIEW_CHARS = 120
DATA_DIR = Path(__file__).resolve().parent / "data" / "file_versions"


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _legacy_hash_json_path(resolved_abs: str) -> Path:
    """Old on-disk layout: only a SHA-256 prefix as the filename."""
    h = hashlib.sha256(resolved_abs.encode("utf-8", errors="replace")).hexdigest()[:20]
    return DATA_DIR / f"{h}.json"


def _meaningful_json_name(resolved_abs: str) -> str:
    """
    Human-readable JSON filename from the source file, e.g. `55_a3f2b1c9_versions.json`.
    The 8-char id disambiguates same basename in different folders.
    """
    p = Path(resolved_abs)
    stem = (p.stem or "file").strip()
    safe = re.sub(r"[^\w\-]", "_", stem).strip("_") or "file"
    if len(safe) > 48:
        safe = safe[:48]
    h = hashlib.sha256(resolved_abs.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"{safe}_{h}_versions.json"


def _storage_path(resolved_abs: str) -> Path:
    """
    JSON path for this file. Prefers the meaningful name; migrates legacy hash-only
    files once when found.
    """
    _ensure_dir()
    new_p = DATA_DIR / _meaningful_json_name(resolved_abs)
    leg_p = _legacy_hash_json_path(resolved_abs)
    if new_p.is_file():
        return new_p
    if leg_p.is_file():
        try:
            new_p.write_bytes(leg_p.read_bytes())
            leg_p.unlink(missing_ok=True)
            _log.info("Migrated version history to %s", new_p.name)
        except OSError as e:
            _log.warning("Could not migrate %s — using legacy: %s", leg_p, e)
            return leg_p
    return new_p


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
        errors="replace",
    )
    tmp.replace(path)


def record_presave_snapshot(abs_path: str, old_content: str) -> None:
    """
    Call immediately before overwriting a file. `old_content` must be the
    current file bytes as read from disk (the state the user is leaving).
    """
    ap = str(Path(abs_path).resolve())
    sp = _storage_path(ap)
    data: Dict[str, Any] = {"path": ap, "versions": []}
    if sp.is_file():
        try:
            data = json.loads(sp.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError) as e:
            _log.warning("Corrupt version file %s: %s", sp, e)
    vers: List[Dict[str, Any]] = [v for v in data.get("versions") or [] if isinstance(v, dict)]
    vers.append(
        {
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "content": old_content,
        }
    )
    while len(vers) > VERSIONS_PER_FILE:
        vers.pop(0)
    data["path"] = ap
    data["versions"] = vers
    try:
        _atomic_write_json(sp, data)
    except OSError as e:
        _log.warning("Could not write version history: %s", e)


def maybe_record_before_write(filepath: str, new_content: str) -> None:
    """
    If `filepath` already exists and its current text differs from `new_content`,
    record that current text as a presave snapshot (ring buffer). Skips when the
    file does not exist yet (first write).
    """
    p = Path(filepath)
    try:
        ap = str(p.resolve())
    except OSError:
        return
    if not p.is_file():
        return
    try:
        old = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    if old == new_content:
        return
    record_presave_snapshot(ap, old)


def _load_versions(abs_path: str) -> List[Dict[str, Any]]:
    ap = str(Path(abs_path).resolve())
    sp = _storage_path(ap)
    if not sp.is_file():
        return []
    try:
        data = json.loads(sp.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return []
    vers = data.get("versions") or []
    return [v for v in vers if isinstance(v, dict) and "content" in v]


def list_versions_meta(abs_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Return entries **newest first** for UI. Each item:
    storage_index (0=oldest in file), at, chars, preview, line1
    """
    if not abs_path:
        return []
    raw = _load_versions(abs_path)
    out: List[Dict[str, Any]] = []
    for i, v in enumerate(raw):
        c = str(v.get("content") or "")
        line1 = (c.splitlines() or [""])[0].strip()[:80]
        out.append(
            {
                "storage_index": i,
                "at": str(v.get("at") or ""),
                "chars": len(c),
                "preview": (c[:PREVIEW_CHARS] + "…") if len(c) > PREVIEW_CHARS else c,
                "line1": line1 or "(empty)",
            }
        )
    # newest = last in storage — show first in UI
    return list(reversed(out))


def get_version_content(abs_path: str, storage_index: int) -> Optional[str]:
    raw = _load_versions(abs_path)
    if storage_index < 0 or storage_index >= len(raw):
        return None
    return str(raw[storage_index].get("content") or "")
