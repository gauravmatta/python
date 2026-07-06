# =============================================================================
# Selection-aware code editor + floating "Explain" (custom Streamlit component)
# =============================================================================
# Uses Ace inside declare_component (explain_selection_frontend/index.html).
# Explain prompts include the **full file** (up to EXPLAIN_MAX_USER_CHARS for the user message).
# Depends: streamlit, requests. No extra pip beyond the main app.
# =============================================================================

from __future__ import annotations

import inspect
import json
import os
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explain_selection_frontend")

_explain_editor = components.declare_component(
    "explain_selection_editor",
    path=_FRONTEND,
)

# Max characters for the Explain / Deep-dive **user** message to Ollama (not tokens).
EXPLAIN_MAX_USER_CHARS = 100_000


def _clamp_font_size(n: int) -> int:
    """Match editor font bounds in 47.py (number_input 10–28)."""
    try:
        x = int(n)
    except (TypeError, ValueError):
        x = 14
    return max(10, min(28, x))


def build_explain_popup_css(font_size: int) -> str:
    """Typography tied to sidebar / file bar code font size (st.session_state.font_size)."""
    fs = _clamp_font_size(font_size)
    h4 = round(fs * 1.08)
    h3 = round(fs * 1.15)
    h2 = round(fs * 1.22)
    h1 = round(fs * 1.3)
    cap = max(10, fs - 1)
    return f"""
<style>
[data-testid="stDialog"] [data-testid="stMarkdownContainer"],
[data-testid="stDialog"] [data-testid="stVerticalBlock"] {{
  max-width: 100% !important;
  overflow-x: auto !important;
  box-sizing: border-box !important;
  font-size: {fs}px !important;
  line-height: 1.55 !important;
}}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h1 {{
  font-size: {h1}px !important;
}}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h2 {{
  font-size: {h2}px !important;
}}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h3 {{
  font-size: {h3}px !important;
}}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h4,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h5,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] h6 {{
  font-size: {h4}px !important;
}}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p code,
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] li code {{
  font-size: {fs}px !important;
}}
[data-testid="stDialog"] [data-testid="stCaption"],
[data-testid="stDialog"] [data-testid="stCaption"] p,
[data-testid="stDialog"] [data-testid="stCaption"] span {{
  font-size: {cap}px !important;
}}
[data-testid="stDialog"] pre,
[data-testid="stDialog"] pre code {{
  white-space: pre-wrap !important;
  word-wrap: break-word !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  max-width: 100% !important;
  overflow-x: auto !important;
  box-sizing: border-box !important;
  font-size: {fs}px !important;
  line-height: 1.5 !important;
}}
[data-testid="stDialog"] [data-testid="stCodeBlock"],
[data-testid="stDialog"] [class*="stCodeBlock"] {{
  max-width: 100% !important;
  overflow-x: auto !important;
  box-sizing: border-box !important;
  font-size: {fs}px !important;
}}
[data-testid="stDialog"] [data-testid="stCodeBlock"] pre,
[data-testid="stDialog"] [data-testid="stCodeBlock"] code {{
  font-size: {fs}px !important;
}}
/* Bordered fallback when st.dialog is unavailable (container key=explain_selection_fb) */
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"],
[class*="st-key-explain_selection_fb"] [data-testid="stVerticalBlock"] {{
  max-width: 100% !important;
  overflow-x: auto !important;
  font-size: {fs}px !important;
  line-height: 1.55 !important;
}}
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] h1 {{
  font-size: {h1}px !important;
}}
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] h2 {{
  font-size: {h2}px !important;
}}
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] h3 {{
  font-size: {h3}px !important;
}}
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] h4,
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] h5,
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] h6 {{
  font-size: {h4}px !important;
}}
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] p code,
[class*="st-key-explain_selection_fb"] [data-testid="stMarkdownContainer"] li code {{
  font-size: {fs}px !important;
}}
[class*="st-key-explain_selection_fb"] pre,
[class*="st-key-explain_selection_fb"] pre code {{
  white-space: pre-wrap !important;
  word-wrap: break-word !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
  max-width: 100% !important;
  overflow-x: auto !important;
  font-size: {fs}px !important;
  line-height: 1.5 !important;
}}
[class*="st-key-explain_selection_fb"] [data-testid="stCodeBlock"],
[class*="st-key-explain_selection_fb"] [class*="stCodeBlock"] {{
  font-size: {fs}px !important;
}}
</style>
"""


def _explain_dialog_decorator(dialog_fn: Any, title: str):
    """Use wide dialog when Streamlit supports the width= argument."""
    try:
        sig = inspect.signature(dialog_fn)
        if "width" in sig.parameters:
            return dialog_fn(title, width="large")
    except (TypeError, ValueError):
        pass
    return dialog_fn(title)


EXPLAIN_SYSTEM = """You are a patient Python tutor embedded in a coding IDE.
The user highlighted a snippet and wants to understand it.

The user message may include the **entire file** for context (imports, definitions elsewhere, call sites). Still **center your answer on the highlighted lines**; use the rest only to relate definitions, usage, and program flow.

Rules:
- Explain what the **selection** does and how it fits the rest of the file when relevant (e.g. a function defined earlier and used later).
- Use clear headings and short paragraphs; prefer bullet steps for multi-statement snippets.
- Define jargon when it appears (e.g. decorator, comprehension, context manager).
- If the selection is syntactically incomplete, say so and suggest expanding the highlight.
- Do not rewrite the whole file unless the user snippet requires a one-line fix example.
- Output GitHub-flavored Markdown only (no outer markdown fences)."""

EXPLAIN_DEEP_SYSTEM = """You are a patient Python tutor. The learner already read a **short** explanation of the same code in the IDE.

Your job now is a **second pass** that goes further for learning:
- Add **1–2 analogies** where they genuinely help (avoid forced metaphors).
- Include **small concrete examples**: e.g. sample values, before/after, or tiny code snippets that illustrate one idea.
- Connect to **beginner-friendly** mental models; call out common mistakes or misconceptions if relevant.
- Do **not** repeat the short summary verbatim; build on it or reorganize for depth.
- Stay accurate; note if something depends on Streamlit, OS, or Python version.
- Use GitHub-flavored Markdown only (no outer markdown fences)."""


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def explain_editor(
    *,
    value: str,
    height: int,
    theme: str,
    language: str,
    font_size: int,
    key: str,
) -> Any:
    """Renders Ace editor with floating Explain; returns last event dict or None."""
    return _explain_editor(
        value=value,
        height=int(height),
        theme=str(theme),
        language=str(language),
        font_size=int(font_size),
        key=key,
        default=None,
    )


def build_explain_user_message(
    *,
    filepath: str,
    basename: str,
    start_line: int,
    end_line: int,
    selected_text: str,
    language: str = "python",
    max_user_chars: int = EXPLAIN_MAX_USER_CHARS,
) -> str:
    """Build user message with **full file** (truncated only if it exceeds the char budget)."""
    body = selected_text.strip()
    fence = "python" if (language or "").lower() == "python" else "text"
    learner = (
        "Explain the **highlighted** lines for a learner learning Python. "
        "Use the full file above/below only as supporting context."
        if fence == "python"
        else "Explain the **highlighted** lines for a learner. "
        "Use the full file only as supporting context (file may not be Python)."
    )

    header = (
        f"### Selection location\n**File:** `{basename}` (path: `{filepath}`)\n"
        f"**Highlighted lines:** {start_line}–{end_line} (1-based, inclusive)\n\n"
    )
    file_intro = (
        f"### Full file (`{basename}`)\n"
        "Entire file content for context (definitions, imports, usages elsewhere in this file).\n\n"
    )
    file_fence_open = f"```{fence}\n"
    file_fence_close = "\n```\n\n"

    highlight_block = (
        f"### Highlighted code (focus your explanation here)\n"
        f"```{fence}\n{body}\n```\n\n"
        f"{learner}"
    )

    full_text = ""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            full_text = f.read()
    except OSError:
        full_text = ""

    overhead = (
        len(header)
        + len(file_intro)
        + len(file_fence_open)
        + len(file_fence_close)
        + len(highlight_block)
        + 400
    )
    budget = max(0, int(max_user_chars) - overhead)
    trunc_note = ""

    if not full_text.strip():
        file_body = "_(Could not read file or file is empty.)_"
    elif len(full_text) <= budget:
        file_body = full_text
    else:
        file_body = full_text[:budget]
        trunc_note = (
            f"\n\n> _File truncated to fit {max_user_chars:,} character budget: "
            f"showing first {len(file_body):,} of {len(full_text):,} characters._\n"
        )

    return (
        header
        + file_intro
        + file_fence_open
        + file_body
        + file_fence_close
        + trunc_note
        + highlight_block
    )


def fetch_explanation_sync(
    base_url: str,
    model: str,
    user_content: str,
    *,
    system: str = EXPLAIN_SYSTEM,
    timeout: float = 120.0,
) -> str:
    if not model:
        return "**Error:** No model selected in the sidebar."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content[:EXPLAIN_MAX_USER_CHARS]},
    ]
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = (resp.json().get("message") or {}).get("content", "") or ""
    return strip_think_tags(raw) or "(Empty response from model.)"


def iter_explain_ollama_stream(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    timeout: float = 300.0,
) -> Iterator[str]:
    """Yields assistant content chunks from Ollama /api/chat (stream=true)."""
    if not model:
        yield "**Error:** No model selected in the sidebar."
        return
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = (data.get("message") or {}).get("content", "") or ""
            if content:
                yield content
            if data.get("done"):
                break
    except requests.ConnectionError:
        yield "**Error:** Cannot connect to Ollama (`ollama serve`)."
    except Exception as e:
        yield f"**Error:** {e}"


def _dedupe_explain_click(evt: Dict[str, Any]) -> bool:
    """Return True if this explain event was already handled (duplicate)."""
    eid = evt.get("event_id")
    if eid is None:
        return False
    key = "_explain_click_event_id"
    if st.session_state.get(key) == eid:
        return True
    st.session_state[key] = eid
    return False


def process_explain_click(
    evt: Dict[str, Any],
    *,
    base_url: str,
    model: Optional[str],
    filepath: str,
    basename: str,
    language: str = "python",
    persist_buffer: Optional[Callable[[str], Tuple[bool, Optional[str]]]] = None,
) -> None:
    """
    Runs after parent syncs file from evt['full_value'] if needed.
    persist_buffer: if provided, called with full buffer string; should save and update session; returns (ok, err).
    """
    if _dedupe_explain_click(evt):
        return
    sel = (evt.get("selected_text") or "").strip()
    if not sel:
        st.toast("Nothing selected to explain.", icon="⚠️")
        return
    if len(sel) > 12000:
        st.toast("Selection too long (max ~12k chars).", icon="⚠️")
        return

    fv = evt.get("full_value")
    if persist_buffer is not None and isinstance(fv, str):
        ok, err = persist_buffer(fv)
        if not ok:
            st.error(f"Could not save before explain: {err}")
            return

    try:
        sl = int(evt.get("start_line", 1))
        el = int(evt.get("end_line", sl))
    except (TypeError, ValueError):
        sl, el = 1, 1

    user_msg = build_explain_user_message(
        filepath=filepath,
        basename=basename,
        start_line=sl,
        end_line=el,
        selected_text=sel,
        language=language,
    )

    api_messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": user_msg[:EXPLAIN_MAX_USER_CHARS]},
    ]

    st.session_state["_explain_popup_title"] = f"{basename} · lines {sl}–{el}"
    st.session_state["_explain_popup_md"] = ""
    st.session_state["_explain_popup_open"] = True
    st.session_state["_explain_stream_pending"] = True
    st.session_state["_explain_api_messages"] = api_messages
    st.session_state.pop("_explain_deep_stream_pending", None)
    st.session_state.pop("_explain_deep_api_messages", None)
    st.session_state["_explain_deep_done"] = False
    st.session_state["_explain_ctx"] = {
        "filepath": filepath,
        "basename": basename,
        "start_line": sl,
        "end_line": el,
        "selected_text": sel,
        "language": language,
        "base_url": base_url,
        "model": model or "",
    }


def build_deep_explain_user_message(
    *,
    filepath: str,
    basename: str,
    start_line: int,
    end_line: int,
    selected_text: str,
    language: str,
    brief_explanation_md: str,
) -> str:
    recap = (brief_explanation_md or "").strip()
    if len(recap) > 4500:
        recap = recap[:4500] + "\n\n…(truncated)"
    extra = (
        "\n\n### What they already read (expand on this; do not copy it)\n"
        + recap
        + "\n\n### Instruction\n"
        "Write the **deeper** lesson: analogies, examples, and intuition. "
        "Use clear Markdown headings."
    )
    reserve = len(extra) + 800
    base = build_explain_user_message(
        filepath=filepath,
        basename=basename,
        start_line=start_line,
        end_line=end_line,
        selected_text=selected_text,
        language=language,
        max_user_chars=max(50_000, EXPLAIN_MAX_USER_CHARS - reserve),
    )
    return base + extra


def _render_explain_main_body() -> None:
    """Initial Explain stream, Deep dive stream, or static markdown."""
    ctx = st.session_state.get("_explain_ctx") or {}
    base_url = str(ctx.get("base_url", "http://localhost:11434"))
    model = str(ctx.get("model", ""))

    deep_pending = bool(st.session_state.get("_explain_deep_stream_pending"))
    deep_msgs = st.session_state.get("_explain_deep_api_messages")

    if deep_pending and deep_msgs:
        brief = st.session_state.get("_explain_popup_md") or ""
        st.markdown(brief)
        st.markdown("---\n\n### Deeper dive — examples & analogies\n\n")
        st.caption("⏳ Streaming deeper explanation…")
        acc: list[str] = []

        def deep_chunks() -> Iterator[str]:
            for c in iter_explain_ollama_stream(
                base_url, model, deep_msgs, timeout=300.0
            ):
                acc.append(c)
                yield c

        stream_fn = getattr(st, "write_stream", None)
        if stream_fn:
            out = stream_fn(deep_chunks())
            raw = ((out or "").strip() or "".join(acc)).strip()
        else:
            raw = fetch_explanation_sync(
                base_url,
                model,
                deep_msgs[1]["content"],
                system=deep_msgs[0]["content"],
                timeout=180.0,
            )
            st.markdown(raw)

        sep = "\n\n---\n\n### Deeper dive — examples & analogies\n\n"
        st.session_state["_explain_popup_md"] = brief + sep + (
            strip_think_tags(raw) or "(Empty response from model.)"
        )
        st.session_state["_explain_deep_stream_pending"] = False
        st.session_state.pop("_explain_deep_api_messages", None)
        st.session_state["_explain_deep_done"] = True
        return

    init_pending = bool(st.session_state.get("_explain_stream_pending"))
    init_msgs = st.session_state.get("_explain_api_messages")

    if init_pending and init_msgs:
        st.caption("⏳ Streaming from Ollama…")
        acc2: list[str] = []

        def init_chunks() -> Iterator[str]:
            for c in iter_explain_ollama_stream(
                base_url, model, init_msgs, timeout=300.0
            ):
                acc2.append(c)
                yield c

        stream_fn = getattr(st, "write_stream", None)
        if stream_fn:
            out = stream_fn(init_chunks())
            raw = ((out or "").strip() or "".join(acc2)).strip()
        else:
            raw = fetch_explanation_sync(
                base_url,
                model,
                init_msgs[1]["content"],
                system=init_msgs[0]["content"],
                timeout=180.0,
            )
            st.markdown(raw)

        st.session_state["_explain_popup_md"] = (
            strip_think_tags(raw) or "(Empty response from model.)"
        )
        st.session_state["_explain_stream_pending"] = False
        st.session_state.pop("_explain_api_messages", None)
        return

    st.markdown(st.session_state.get("_explain_popup_md") or "")


def render_explain_popup() -> None:
    """Call once per run (e.g. after the editor). Shows dialog or bordered fallback."""
    if not st.session_state.get("_explain_popup_open"):
        return

    title = st.session_state.get("_explain_popup_title") or "Explain selection"
    body = st.session_state.get("_explain_popup_md") or ""

    # Match file bar "Font" (code editor size); same session key as 47.py.
    _fs = _clamp_font_size(st.session_state.get("font_size", 14))

    # Apply before opening the dialog so rules exist when the modal mounts.
    st.markdown(build_explain_popup_css(_fs), unsafe_allow_html=True)

    def _run_deep_dive() -> None:
        # Read from session only (dialog fragment reruns do not re-execute outer scope).
        ctx = st.session_state.get("_explain_ctx") or {}
        if not isinstance(ctx, dict) or not ctx.get("selected_text"):
            st.toast("Missing selection context — close and use Explain again.", icon="⚠️")
            return
        brief = st.session_state.get("_explain_popup_md") or ""
        if "### Deeper dive — examples & analogies" in brief:
            st.session_state["_explain_deep_done"] = True
            st.toast("Deep dive already added.", icon="ℹ️")
            return
        brief_core = brief.split("### Deeper dive — examples & analogies")[0].strip()
        um = build_deep_explain_user_message(
            filepath=str(ctx.get("filepath", "")),
            basename=str(ctx.get("basename", "")),
            start_line=int(ctx.get("start_line", 1)),
            end_line=int(ctx.get("end_line", 1)),
            selected_text=str(ctx.get("selected_text", "")),
            language=str(ctx.get("language", "python")),
            brief_explanation_md=brief_core,
        )
        st.session_state["_explain_deep_stream_pending"] = True
        st.session_state["_explain_deep_api_messages"] = [
            {"role": "system", "content": EXPLAIN_DEEP_SYSTEM},
            {"role": "user", "content": um[:EXPLAIN_MAX_USER_CHARS]},
        ]
        st.rerun()

    dialog_fn = getattr(st, "dialog", None)
    if callable(dialog_fn):
        _dec = _explain_dialog_decorator(dialog_fn, "Explain selection")

        @_dec
        def _dlg():
            st.caption(st.session_state.get("_explain_popup_title") or title)
            _render_explain_main_body()
            _dd = bool(st.session_state.get("_explain_deep_done"))
            c1, c2 = st.columns(2)
            with c1:
                if _dd:
                    st.caption("Deep dive already added for this explanation.")
                else:
                    if st.button(
                        "Deep dive",
                        key="_explain_deep_dive",
                        use_container_width=True,
                    ):
                        _run_deep_dive()
            with c2:
                if st.button("Close", key="_explain_close_dlg", use_container_width=True):
                    st.session_state["_explain_popup_open"] = False
                    st.session_state["_explain_stream_pending"] = False
                    st.session_state["_explain_deep_stream_pending"] = False
                    st.session_state.pop("_explain_api_messages", None)
                    st.session_state.pop("_explain_deep_api_messages", None)
                    st.rerun()

        _dlg()
        return

    def _fallback_body():
        st.markdown(f"#### {st.session_state.get('_explain_popup_title') or title}")
        _render_explain_main_body()
        _dd = bool(st.session_state.get("_explain_deep_done"))
        c1, c2 = st.columns(2)
        with c1:
            if _dd:
                st.caption("Deep dive already added.")
            else:
                if st.button(
                    "Deep dive",
                    key="_explain_deep_dive_fb",
                ):
                    _run_deep_dive()
        with c2:
            if st.button("Close", key="_explain_close_fb"):
                st.session_state["_explain_popup_open"] = False
                st.session_state["_explain_stream_pending"] = False
                st.session_state["_explain_deep_stream_pending"] = False
                st.session_state.pop("_explain_api_messages", None)
                st.session_state.pop("_explain_deep_api_messages", None)
                st.rerun()

    try:
        with st.container(border=True, key="explain_selection_fb"):
            _fallback_body()
    except TypeError:
        with st.container(border=True):
            _fallback_body()
