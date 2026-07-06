# =============================================================================
# Vision / multimodal chat helpers for Ollama (/api/chat)
# =============================================================================
# User messages may include base64-encoded images (e.g. qwen3.5 — see
# https://ollama.com/library/qwen3.5). Production-oriented: re-encode with
# Pillow (strip EXIF, downscale), total payload caps, schema version.
# =============================================================================

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

MAX_IMAGES_PER_MESSAGE = 4
MAX_BYTES_PER_IMAGE = 4 * 1024 * 1024  # input file read cap (before re-encode)
MAX_EDGE_PX = 2048
# Sum of base64 string lengths across all images in one message (rough wire size)
MAX_TOTAL_BASE64_CHARS = 6_000_000

VISION_MESSAGE_SCHEMA_VERSION = 1

# Heuristic: Ollama model names that commonly accept `images` in /api/chat
_VISION_NAME_HINTS = (
    "vl", "vision", "llava", "moondream", "bakllava", "qwen3.5", "qwen2-vl",
    "qwen-vl", "gemma3", "mistral-small3", "minicpm-v", "internvl",
    "llama3.2-vision", "granite-vision", "pixtral",
)


def is_likely_vision_model(model_name: str) -> bool:
    if not model_name:
        return False
    low = model_name.lower()
    return any(h in low for h in _VISION_NAME_HINTS)


def _pil_available() -> bool:
    try:
        import PIL.Image  # noqa: F401
        return True
    except ImportError:
        return False


def _sniff_mime(data: bytes) -> Optional[str]:
    if len(data) < 12:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _reencode_with_pil(raw: bytes, filename: str = "") -> Optional[Tuple[bytes, str, str]]:
    """EXIF-stripped, downscaled bytes; (data, mime, short_name). None if Pillow missing/invalid."""
    try:
        from io import BytesIO

        from PIL import Image, ImageOps
    except ImportError:
        return None

    try:
        bio = BytesIO(raw)
        im = Image.open(bio)
        im = ImageOps.exif_transpose(im)
        im.load()
        try:
            im.seek(0)
        except EOFError:
            pass
    except Exception as e:
        log.debug("PIL could not open image: %s", e)
        return None

    w, h = im.size
    m = max(w, h)
    if m > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / m
        im = im.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.Resampling.LANCZOS,
        )

    out = BytesIO()
    base_name = (filename or "image").split("/")[-1].rsplit(".", 1)[0][:80] or "image"

    if im.mode in ("RGBA", "LA", "P"):
        im.save(out, format="PNG", optimize=True)
        mime = "image/png"
        name = f"{base_name}.png"
    else:
        rgb = im.convert("RGB")
        rgb.save(out, format="JPEG", quality=88, optimize=True)
        mime = "image/jpeg"
        name = f"{base_name}.jpg"

    data = out.getvalue()
    if len(data) > MAX_BYTES_PER_IMAGE:
        out2 = BytesIO()
        rgb = im.convert("RGB") if im.mode != "RGB" else im
        rgb.save(out2, format="JPEG", quality=72, optimize=True)
        data = out2.getvalue()
        mime = "image/jpeg"
        name = f"{base_name}.jpg"

    if len(data) > MAX_BYTES_PER_IMAGE:
        log.warning("Image still exceeds max bytes after re-encode; skipping.")
        return None

    return data, mime, name


def process_uploaded_image(raw: bytes, filename: str = "") -> Optional[Dict[str, str]]:
    """
    Validate and normalize image bytes for Ollama.
    Prefer Pillow re-encode (metadata stripped, downscaled); else raw sniff + base64.
    """
    if not raw or len(raw) > MAX_BYTES_PER_IMAGE:
        return None

    prepared = _reencode_with_pil(raw, filename)
    if prepared:
        data, mime, name = prepared
    else:
        mime = _sniff_mime(raw)
        if not mime:
            return None
        data = raw
        name = (filename or "image").split("/")[-1][:120]

    return {
        "b64": base64.standard_b64encode(data).decode("ascii"),
        "name": name,
        "mime": mime,
    }


def total_base64_chars(images: Optional[List[Dict[str, str]]]) -> int:
    n = 0
    for im in images or []:
        if isinstance(im, dict):
            n += len(im.get("b64") or "")
    return n


def normalize_pending_count(
    images: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Cap count and total base64 payload (drop oldest until under limit)."""
    if not images:
        return []
    out = images[:MAX_IMAGES_PER_MESSAGE]
    while out and total_base64_chars(out) > MAX_TOTAL_BASE64_CHARS:
        log.warning("Vision queue: dropping oldest image to satisfy payload cap")
        out = out[1:]
    while len(out) > MAX_IMAGES_PER_MESSAGE:
        out = out[1:]
    return out


def pillow_status() -> str:
    return "ok" if _pil_available() else "missing"


def message_payload_chars(msg: Dict[str, Any]) -> int:
    """Rough size for context trimming (text + embedded images)."""
    n = len(msg.get("content") or "") + 24
    for im in msg.get("images") or []:
        if isinstance(im, dict):
            n += len(im.get("b64") or "")
        elif isinstance(im, str):
            n += len(im)
    return n


def trim_messages_for_context(
    messages: List[Dict[str, Any]],
    system_msg: str,
    max_total_chars: int,
) -> List[Dict[str, Any]]:
    """Keep the newest messages whose total payload fits under max_total_chars."""
    budget = max_total_chars - len(system_msg)
    if budget < 4000:
        budget = 4000
    result: List[Dict[str, Any]] = []
    total = 0
    for msg in reversed(messages):
        w = message_payload_chars(msg)
        if total + w > budget and result:
            break
        entry: Dict[str, Any] = {
            "role": msg["role"],
            "content": msg.get("content") or "",
        }
        if msg.get("images"):
            entry["images"] = msg["images"]
        if msg.get("vision_schema"):
            entry["vision_schema"] = msg["vision_schema"]
        result.insert(0, entry)
        total += w
    return result


def to_ollama_messages(
    trimmed: List[Dict[str, Any]],
    system_content: str,
) -> List[Dict[str, Any]]:
    """Build JSON payload for Ollama /api/chat."""
    out: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
    for m in trimmed:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "assistant":
            out.append({"role": "assistant", "content": content})
            continue
        om: Dict[str, Any] = {"role": "user", "content": content}
        imgs = m.get("images") or []
        if imgs:
            b64_list = []
            for im in imgs:
                if isinstance(im, dict) and im.get("b64"):
                    b64_list.append(im["b64"])
                elif isinstance(im, str):
                    b64_list.append(im)
            if b64_list:
                om["images"] = b64_list
        out.append(om)
    return out


def ollama_messages_include_images(ollama_messages: List[Dict[str, Any]]) -> bool:
    for m in ollama_messages:
        if m.get("role") == "user" and m.get("images"):
            return True
    return False
