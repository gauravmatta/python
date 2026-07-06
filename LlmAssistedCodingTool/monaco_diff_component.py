# =============================================================================
# Monaco diff viewer (Streamlit custom component)
# =============================================================================
# Side-by-side diff in the main editor area when Apply Changes is pending.
# Frontend: monaco_diff_frontend/index.html (CDN Monaco + streamlit-component-lib).
# =============================================================================

from __future__ import annotations

import os
from typing import Any

import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monaco_diff_frontend")

_monaco_diff = components.declare_component(
    "monaco_diff_editor",
    path=_FRONTEND,
)


def monaco_diff_view(
    *,
    original_text: str,
    modified_text: str,
    height: int,
    theme: str,
    language: str,
    font_size: int,
    key: str,
) -> Any:
    """Render Monaco diff editor. Returns dict with action accept|cancel or None."""
    return _monaco_diff(
        original_text=original_text,
        modified_text=modified_text,
        height=int(height),
        theme=str(theme),
        language=str(language),
        font_size=int(font_size),
        key=key,
        default=None,
    )


def is_available() -> bool:
    return os.path.isdir(_FRONTEND) and os.path.isfile(
        os.path.join(_FRONTEND, "index.html")
    )
