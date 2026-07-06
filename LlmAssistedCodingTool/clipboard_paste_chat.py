# =============================================================================
# Clipboard image → chat image queue (Streamlit)
# =============================================================================
# Uses streamlit-paste-button (optional). Browsers: Chrome / Edge / Safari;
# clipboard API needs secure context (https or localhost).
#
# When `clipboard_paste_frontend/` is present (bundled wide/tall iframe + full-width
# button), that UI is used; otherwise falls back to streamlit_paste_button.
# =============================================================================

from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    from streamlit_paste_button import paste_image_button
except ImportError:
    paste_image_button = None  # type: ignore

_LOCAL_FRONTEND = Path(__file__).resolve().parent / "clipboard_paste_frontend"
if _LOCAL_FRONTEND.is_dir() and (_LOCAL_FRONTEND / "index.html").is_file():
    _paste_component = components.declare_component(
        "clipboard_paste_wide",
        path=str(_LOCAL_FRONTEND),
    )
else:
    _paste_component = None

# True if bundled UI is available or the pip package is installed.
HAS_STREAMLIT_PASTE = (_paste_component is not None) or (paste_image_button is not None)


@dataclass
class PasteResult:
    image_data: Optional[Image.Image] = None


def _data_url_to_image(data_url: str) -> Image.Image:
    _, _data_url = data_url.split(";base64,", 1)
    return Image.open(io.BytesIO(base64.b64decode(_data_url)))


def _paste_image_button_local(
    label: str,
    *,
    text_color: Optional[str] = "#ffffff",
    background_color: Optional[str] = "#3498db",
    hover_background_color: Optional[str] = "#2980b9",
    key: Optional[str] = "paste_button",
    errors: Optional[str] = "ignore",
) -> PasteResult:
    if _paste_component is None:
        return PasteResult()
    component_value = _paste_component(
        label=label,
        text_color=text_color,
        background_color=background_color,
        hover_background_color=hover_background_color,
        key=key,
    )
    if component_value is None:
        return PasteResult()
    if isinstance(component_value, str) and component_value.startswith("error"):
        if errors == "raise":
            if component_value.startswith("error: no image"):
                st.error("**Error**: No image found in clipboard", icon="🚨")
            else:
                st.error(
                    re.sub(r"error: (.+)(: .+)", r"**\1**\2", component_value),
                    icon="🚨",
                )
        return PasteResult()
    return PasteResult(image_data=_data_url_to_image(component_value))


def try_clipboard_paste(
    session_state: Any,
    ovc: Any,
    *,
    allow_append: bool = True,
) -> bool:
    """
    Render the paste button and, if the user just pasted an image, append it
    to session_state.chat_pending_images (via ovc.normalize_pending_count).

    Remounts the widget key after a successful paste so the same image is not
    re-queued on every rerun.

    If allow_append is False (e.g. queue full), returns False without rendering.
    Returns True if an image was added (caller should st.rerun()).
    """
    if not allow_append:
        return False
    if not HAS_STREAMLIT_PASTE:
        return False

    pk = int(session_state.get("clipboard_paste_key", 0))
    if _paste_component is not None:
        pr = _paste_image_button_local(
            "📋 Paste from clipboard",
            key=f"cb_paste_{pk}",
            errors="ignore",
        )
    elif paste_image_button is not None:
        pr = paste_image_button(
            "📋 Paste from clipboard",
            key=f"cb_paste_{pk}",
            errors="ignore",
        )
    else:
        return False
    if pr.image_data is None:
        return False

    buf = io.BytesIO()
    try:
        pr.image_data.save(buf, format="PNG")
    except Exception:
        session_state["clipboard_paste_key"] = pk + 1
        return False

    raw = buf.getvalue()
    pic = ovc.process_uploaded_image(raw, "clipboard.png")
    if not pic:
        session_state["clipboard_paste_key"] = pk + 1
        return False

    pending = list(session_state.get("chat_pending_images") or [])
    pending.append(pic)
    session_state.chat_pending_images = ovc.normalize_pending_count(pending)
    session_state["clipboard_paste_key"] = pk + 1
    session_state["vision_upload_key"] = int(session_state.get("vision_upload_key", 0)) + 1
    return True
