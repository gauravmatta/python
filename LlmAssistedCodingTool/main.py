# =============================================================================
# Local LLM-Assisted Coding Tool
# =============================================================================
#
# A Streamlit-powered coding assistant that lets you browse, view, and edit
# Python files with the help of local Ollama LLMs.
#
# Features:
#   - File browser with Python file filtering (recursive optional)
#   - Syntax-highlighted code viewer with line numbers (Pygments)
#   - Line range selection for focused LLM context
#   - Sidebar chat with streaming Ollama responses
#   - Apply Changes workflow (extract code from LLM, show diff, write to file)
#   - Integrated code runner with error capture
#   - Shell command line (pip install, streamlit run, etc.) — background run + Stop button
#   - "Fix This Error" auto-sends runtime errors to LLM
#   - Customizable system prompt (Teacher / Senior Dev / Default)
#   - Optional web research (DDG + heuristics + LLM queries, GitHub raw fetch, larger budgets; web_research_coding.py)
#   - Optional code planning: rich JSON roadmap, import summary + stderr, recent chat digest,
#     definition_of_done, step lint, JSON repair, web snack, "Stronger plan" pass; queued screenshots
#     are sent to the planner and again on Proceed when using a vision-capable model (planning_helper.py)
#   - Learner editor (explain_selection.py): Ace + floating Explain near selection, modal result,
#     optional follow-up via sidebar chat (custom component; explain_selection_frontend/index.html)
#   - Apply Changes review: Monaco side-by-side diff in the main area when pending (monaco_diff_component.py +
#     monaco_diff_frontend/index.html); Apply/Cancel on the diff toolbar; fallback Apply/Discard if Monaco missing
#   - Vision: Pillow re-encode, EXIF strip, downscale, payload caps, queue edit, model hint,
#     stream→sync fallback, metrics + logging (ollama_vision_chat.py)
#   - Per-.py chat persistence: project/data/chats/<stem>.json, rolling_summary + messages,
#     atomic save, corrupt → .json.bak.* (chat_persistence.py)
#   - Cross-project snippet library: spaCy embeddings, NumPy cosine dedup (>0.95) + LLM merge,
#     top-k snippets injected into chat context; data/snippet_library.json (snippet_library.py)
#   - @file.py mentions: paste @10.py or @pkg/module.py in chat to inject full file text from the
#     selected project folder into planner + chat context (at_file_refs.py)
#   - Last 10 on-disk snapshots per file before each save (editor / Apply); sidebar restore (file_version_history.py)
#   - UI settings + last folder/file persisted in data/app_ui_settings.json (app_settings_persistence.py)
#   - Optional clipboard: pip install streamlit-paste-button (clipboard_paste_chat.py)
#
# Install (pinned versions from pip show on this PC):
#   pip install streamlit==1.55.0 requests==2.32.5 pygments==2.17.2 streamlit-ace==0.1.1 ddgs==9.12.0 beautifulsoup4==4.12.2 numpy==1.26.4
#   pip install spacy==3.7.4 && python -m spacy download en_core_web_md   # model en-core-web-md 3.7.1; snippet library + retrieval
#   pip install streamlit-paste-button==0.1.2 pillow==10.4.0   # clipboard + image re-encode / EXIF strip
#
# Run:
#   streamlit main.py
#
# Requirements:
#   - Ollama must be running (ollama serve)
#   - At least one model installed (e.g. ollama pull qwen3:8b)
# =============================================================================

import os
import re
import sys
import json
import base64
import html
import time
import logging
import hashlib
import subprocess
import requests
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from typing import Any, Dict, Optional

import web_research_coding as wrc
import planning_helper as ph
import ollama_vision_chat as ovc
import clipboard_paste_chat as cpc
import chat_persistence as cp
import snippet_library as sl
import at_file_refs as atf
import file_version_history as fvh
import app_settings_persistence as asp

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
_app_log = logging.getLogger("code_assistant_47")

try:
    from streamlit_ace import st_ace
    HAS_ACE = True
except ImportError:
    HAS_ACE = False

try:
    import explain_selection as esel

    _EXPLAIN_FRONTEND = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "explain_selection_frontend",
    )
    HAS_EXPLAIN_EDITOR = os.path.isdir(_EXPLAIN_FRONTEND)
except Exception:
    esel = None
    HAS_EXPLAIN_EDITOR = False

try:
    import monaco_diff_component as mdc

    HAS_MONACO_DIFF = mdc.is_available()
except Exception:
    mdc = None
    HAS_MONACO_DIFF = False

try:
    from background_shell_runner import BackgroundShellRunner
except ImportError:
    BackgroundShellRunner = None

ACE_THEME_MAP = {
    "monokai": "monokai",
    "native": "terminal",
    "friendly": "chrome",
    "vs": "textmate",
    "dracula": "dracula",
}


# =============================================================================
# CONFIG
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
MAX_CONTEXT_CHARS = 50000
SHELL_TIMEOUT = 120
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".tox", ".eggs", ".egg-info", "dist", "build",
}


def planner_invoke(model, user_prompt: str, planner_images=None):
    """Planner with import summary, full path, stderr, optional brief web snack, JSON repair."""
    if not model:
        return None, "No model", []
    at_block, at_warnings = atf.load_at_files_context(
        user_prompt or "",
        str(st.session_state.get("root_folder") or ""),
    )
    snack = ""
    if st.session_state.get("planner_snack_research"):
        snack = ph.build_planner_research_snack(
            user_prompt,
            st.session_state.current_content or "",
            st.session_state.run_stderr or "",
        )
    snippet_ctx = ""
    if st.session_state.get("snippet_retrieval_enabled", True):
        snippet_ctx = sl.relevant_snippets_markdown(
            atf.strip_at_mentions_for_query(user_prompt or ""),
            top_k=int(st.session_state.get("snippet_top_k", 5)),
            enabled=True,
        )
    plan, raw = ph.call_planner_sync(
        OLLAMA_BASE_URL,
        model,
        user_prompt,
        filename=os.path.basename(st.session_state.current_file or "unknown"),
        file_path=st.session_state.current_file or "",
        code_excerpt=st.session_state.current_content or "",
        error_excerpt=st.session_state.run_stderr or "",
        research_snack=snack,
        snippet_library_context=snippet_ctx,
        referenced_files_context=at_block,
        messages_for_digest=st.session_state.messages,
        planner_images=planner_images,
    )
    return plan, raw, at_warnings


# =============================================================================
# OLLAMA HELPERS
# =============================================================================

def strip_think_tags(text):
    """Remove <think>...</think> reasoning blocks (e.g. from Qwen models)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_ollama_models():
    """Fetch the list of models installed in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def ollama_generation_options_from_session() -> Dict[str, Any]:
    """Build Ollama `options` from sidebar (temperature, top_p)."""
    return {
        "temperature": float(st.session_state.get("ollama_temperature", 0.4)),
        "top_p": float(st.session_state.get("ollama_top_p", 0.4)),
    }


def ollama_chat_sync_complete(
    model,
    messages,
    timeout: float,
    options: Optional[Dict[str, Any]] = None,
):
    """Single non-streaming /api/chat response (full assistant text)."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content", "") or ""


def stream_ollama_response(
    model,
    messages,
    options: Optional[Dict[str, Any]] = None,
):
    """Streaming Ollama call; vision requests use longer timeout and sync fallback."""
    has_img = ovc.ollama_messages_include_images(messages)
    timeout = 300.0 if has_img else 120.0
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if options:
        payload["options"] = options
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=timeout,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    break
    except requests.ConnectionError:
        yield "**Error:** Cannot connect to Ollama. Make sure it's running (`ollama serve`)."
    except Exception as e:
        _app_log.warning("Ollama stream failed: %s", type(e).__name__)
        if has_img:
            try:
                _app_log.info(
                    "Retrying vision request without streaming (model=%s)", model
                )
                text = ollama_chat_sync_complete(
                    model, messages, timeout=timeout, options=options
                )
                if text:
                    yield text
                    return
            except Exception as e2:
                _app_log.exception("Ollama vision sync retry failed")
                yield (
                    "**Error:** Multimodal request failed (stream and non-stream). "
                    f"Try another vision model or smaller images. ({type(e2).__name__})"
                )
                return
        yield f"**Error:** {e}"


def trim_message_history(messages, system_msg, max_total_chars=MAX_CONTEXT_CHARS):
    """Keep the most recent messages that fit within the context budget (text + vision payloads)."""
    return ovc.trim_messages_for_context(messages, system_msg, max_total_chars)


# =============================================================================
# FILE SYSTEM HELPERS
# =============================================================================

def list_files(root, extensions=None, show_all=False, recursive=False):
    """List files in a directory, optionally filtered by extensions."""
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    files = []
    try:
        if recursive:
            for dirpath, dirnames, filenames in os.walk(root_path):
                dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
                for fname in sorted(filenames):
                    fpath = Path(dirpath) / fname
                    if show_all or (extensions and fpath.suffix.lower() in extensions):
                        files.append(fpath)
        else:
            for item in sorted(root_path.iterdir()):
                if item.is_file():
                    if show_all or (extensions and item.suffix.lower() in extensions):
                        files.append(item)
    except PermissionError:
        pass
    return files


def load_file(filepath):
    """Read file content. Returns (content, error_message)."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), None
    except Exception as e:
        return "", str(e)


def save_file(filepath, content, record_version=True):
    """Write content to file. Optionally ring-buffer presave snapshots under data/file_versions/."""
    try:
        if record_version:
            fvh.maybe_record_before_write(filepath, content)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True, None
    except Exception as e:
        return False, str(e)


NEW_PY_TEMPLATE = '''# New Python file\n'''

_WIN_ILLEGAL = '<>:"/\\|?*'


def sanitize_py_filename(name):
    """Return a safe *.py basename or None if invalid."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if not name:
        return None
    name = os.path.basename(name)
    for c in _WIN_ILLEGAL:
        name = name.replace(c, "")
    name = name.strip().strip(".")
    if not name:
        return None
    if not name.lower().endswith(".py"):
        name += ".py"
    if len(name) > 200:
        return None
    return name


def create_new_python_file(root_dir, desired_name=None):
    """Create a new .py file in root_dir. If desired_name is empty/None, uses untitled_N.py."""
    base = Path(root_dir)
    if not base.is_dir():
        return None, "Not a valid folder"
    if desired_name and desired_name.strip():
        safe = sanitize_py_filename(desired_name)
        if not safe:
            return None, "Invalid file name (use letters, numbers, _ - only)"
        path = base / safe
        if path.exists():
            return None, f"'{safe}' already exists"
    else:
        path = base / "untitled.py"
        n = 1
        while path.exists():
            path = base / f"untitled_{n}.py"
            n += 1
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(NEW_PY_TEMPLATE)
        return str(path.resolve()), None
    except Exception as e:
        return None, str(e)


def extract_code_blocks(text):
    """Extract fenced code blocks from markdown-formatted LLM response."""
    blocks = []
    pattern = r"```(?:python)?\s*\n(.*?)```"
    for match in re.finditer(pattern, text, re.DOTALL):
        code = match.group(1).rstrip()
        if code:
            blocks.append(code)
    return blocks


# =============================================================================
# CODE EXECUTION
# =============================================================================

def run_python_file(filepath):
    """Execute a Python file; block until it exits. No timeout (games, training, etc.)."""
    fp = os.path.abspath(str(filepath))
    cwd = os.path.dirname(fp) or "."
    exe = sys.executable
    try:
        result = subprocess.run(
            [exe, fp],
            capture_output=True,
            text=True,
            timeout=None,
            cwd=cwd,
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), -1


def run_shell_command(cmd, cwd, timeout=SHELL_TIMEOUT):
    """Run a shell command in cwd (Windows: cmd.exe /c). Returns (stdout, stderr, returncode)."""
    if not cmd or not str(cmd).strip():
        return "", "Empty command.", -1
    cmd = str(cmd).strip()
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["cmd.exe", "/c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or None,
            )
        else:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or None,
            )
        return result.stdout or "", result.stderr or "", result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout} seconds.", -1
    except Exception as e:
        return "", str(e), -1


def get_shell_runner():
    if BackgroundShellRunner is None:
        return None
    if "_shell_runner" not in st.session_state:
        st.session_state._shell_runner = BackgroundShellRunner()
    return st.session_state._shell_runner


def sync_shell_runner_to_session(runner):
    """Mirror background shell output into run_* for the shared output panel."""
    if runner is None:
        return
    so, se, rc, run, err, cmd, rid = runner.snapshot()
    pe = runner.pop_start_error()
    if pe:
        st.error(pe)
    if not cmd and not run:
        return
    prefix = f"$ {cmd}\n\n" if cmd else ""
    if run:
        st.session_state.run_stdout = prefix + so
        st.session_state.run_stderr = se
        st.session_state.run_returncode = None
    elif rc is not None:
        sig = (rid, rc)
        if st.session_state.get("_shell_done_sig") != sig:
            st.session_state["_shell_done_sig"] = sig
            st.session_state.run_stdout = prefix + so
            st.session_state.run_stderr = se
            st.session_state.run_returncode = rc


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROMPT_MODES = {
    "Default": (
        "You are an expert Python developer and coding assistant. "
        "Give clear, practical advice with well-structured code."
    ),
    "Teacher": (
        "You are a patient programming teacher. Explain concepts step by step, "
        "use analogies, and make sure the student understands the 'why' behind "
        "every choice. Include examples where helpful."
    ),
    "Senior Dev": (
        "You are a senior software engineer doing code review. Be concise and "
        "direct. Focus on production readiness, edge cases, performance, and "
        "best practices. Skip pleasantries."
    ),
}


def finalize_user_msg_for_chat(user_msg: dict, model) -> None:
    """Append text-only image archive note; keep images on the dict for the model."""
    imgs = user_msg.get("images")
    if not imgs:
        return
    note = cp.brief_image_archive_note(
        model,
        OLLAMA_BASE_URL,
        imgs,
        user_msg.get("content") or "",
    )
    base = (user_msg.get("content") or "").rstrip()
    user_msg["content"] = base + note


def build_system_prompt(mode, custom_prompt, filename, content):
    """Build the system prompt with code context injected."""
    base = custom_prompt.strip() if custom_prompt.strip() else PROMPT_MODES.get(mode, PROMPT_MODES["Default"])
    parts = [base]

    if content:
        total_lines = content.count("\n") + 1
        parts.append(f"\nThe user is working on the file: `{filename}` ({total_lines} lines)")
        parts.append(f"\nFile content:\n```python\n{content}\n```")

    parts.append(
        "\nWhen suggesting code changes, return the COMPLETE modified file "
        "inside a single ```python code block so the user can apply it directly."
    )
    return "\n".join(parts)


def run_writer_pipeline(
    prompt_to_process,
    plan_suffix,
    chat_container,
    selected_model,
):
    """Append assistant reply: research (if enabled) + stream. User message must already be in messages."""
    filename = os.path.basename(st.session_state.current_file or "unknown")
    system_msg = build_system_prompt(
        st.session_state.get("prompt_mode", "Default"),
        st.session_state.get("custom_prompt", ""),
        filename,
        st.session_state.current_content,
    )

    research_sources = []
    rmode = st.session_state.get("research_mode", "Auto")
    if rmode != "Off" and selected_model:
        err_txt = st.session_state.run_stderr or ""
        run_auto = rmode == "Always" or (
            rmode == "Auto"
            and wrc.should_run_research_auto(prompt_to_process, err_txt)
        )
        if run_auto:
            chat_tail = ""
            for m in st.session_state.messages[:-1][-6:]:
                bit = (m.get("content") or "")[:500]
                if m.get("images"):
                    bit += f" [{len(m['images'])} image(s)]"
                chat_tail += f"{m['role']}: {bit}\n"
            try:
                with st.status("Web research…", expanded=True) as rs:
                    rs.update(label="Planning queries & searching…")
                    rblock, research_sources = wrc.build_research_context(
                        user_goal=prompt_to_process,
                        code_snippet=st.session_state.current_content or "",
                        error_text=err_txt,
                        chat_tail=chat_tail,
                        model=selected_model,
                        base_url=OLLAMA_BASE_URL,
                        deep_fetch=st.session_state.get("research_deep", True),
                        use_llm_planner=st.session_state.get(
                            "research_llm_planner", True
                        ),
                    )
                    rs.update(label="Research ready", state="complete")
                if rblock.strip():
                    system_msg = system_msg + "\n\n" + rblock
            except Exception as ex:
                st.warning(f"Web research skipped: {ex}")

    if plan_suffix and plan_suffix.strip():
        system_msg = system_msg + "\n\n" + plan_suffix.strip()

    system_msg = cp.append_summary_to_system(
        system_msg, st.session_state.get("rolling_summary") or ""
    )

    at_block, at_warnings = atf.load_at_files_context(
        prompt_to_process or "",
        str(st.session_state.get("root_folder") or ""),
    )
    for _w in at_warnings:
        st.warning(_w)
    if at_block:
        system_msg = system_msg + "\n\n" + at_block.strip()

    system_msg = sl.append_relevant_snippets_to_system(
        system_msg,
        atf.strip_at_mentions_for_query(prompt_to_process or ""),
        top_k=int(st.session_state.get("snippet_top_k", 5)),
        enabled=bool(st.session_state.get("snippet_retrieval_enabled", True)),
    )

    trimmed = trim_message_history(st.session_state.messages, system_msg)
    ollama_messages = ovc.to_ollama_messages(trimmed, system_msg)

    last_user = st.session_state.messages[-1] if st.session_state.messages else {}
    if last_user.get("role") == "user" and last_user.get("images"):
        st.session_state.metric_chat_vision_turns = (
            int(st.session_state.get("metric_chat_vision_turns", 0)) + 1
        )
        _app_log.info(
            "vision_chat model=%s images=%d approx_b64_chars=%d",
            selected_model,
            len(last_user["images"]),
            ovc.total_base64_chars(last_user["images"]),
        )
    elif last_user.get("role") == "user":
        st.session_state.metric_chat_text_turns = (
            int(st.session_state.get("metric_chat_text_turns", 0)) + 1
        )

    with chat_container:
        with st.chat_message("user"):
            st.markdown(last_user.get("content") or prompt_to_process)
            for im in last_user.get("images") or []:
                if isinstance(im, dict) and im.get("b64"):
                    try:
                        st.image(
                            base64.standard_b64decode(im["b64"]),
                            caption=im.get("name") or "",
                            width=320,
                        )
                    except Exception:
                        st.caption("(Could not display image thumbnail.)")
        with st.chat_message("assistant"):
            t0 = time.time()
            response = st.write_stream(
                stream_ollama_response(
                    selected_model,
                    ollama_messages,
                    options=ollama_generation_options_from_session(),
                )
            )
            st.caption(f"⏱️ {time.time() - t0:.1f}s")

    clean = strip_think_tags(response) if response else ""
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": clean,
            "sources": research_sources if research_sources else None,
        }
    )

    msgs = st.session_state.messages
    if len(msgs) >= 2 and msgs[-2].get("role") == "user":
        u_prev = msgs[-2]
        if u_prev.get("images"):
            u_prev.pop("images", None)
            u_prev.pop("vision_schema", None)

    cp.maybe_compress_history(st.session_state, selected_model, OLLAMA_BASE_URL)
    cp.save_chat_from_session(
        st.session_state,
        st.session_state.root_folder,
        st.session_state.get("current_file"),
    )

    blocks = extract_code_blocks(clean)
    if blocks:
        st.session_state.apply_code = max(blocks, key=len)
    else:
        st.session_state.apply_code = None

    st.rerun()


# =============================================================================
# STREAMLIT PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="LLM Code Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injected via st.html so rules are not stripped (Streamlit may sanitize <style> in st.markdown).
# Main column: default Streamlit top padding is ~5rem; ~3rem is tighter but still clears the
# fixed header (values near 1rem clip the first heading — do not go that low).
CUSTOM_CSS = """
<style>
    .stApp .main .block-container,
    .stApp [data-testid="stMain"] .block-container {
        padding-top: 3rem !important;
    }
    /* Tighter space between the page title (h2) and the next block (e.g. File selection) */
    .stApp [data-testid="stMain"] .block-container h2 {
        margin-bottom: 0.35rem !important;
        line-height: 1.25 !important;
    }

    /* Wider sidebar */
    [data-testid="stSidebar"] { min-width: 380px; }
    [data-testid="stSidebar"] > div:first-child { padding-top: 0.8rem; }

    /* Diff colors */
    .diff-add { color: #3fb950; background: rgba(63,185,80,0.08); }
    .diff-del { color: #f85149; background: rgba(248,81,73,0.08); }
    .diff-hdr { color: #58a6ff; font-weight: bold; }

    /* File info */
    .file-info { font-family: monospace; font-size: 0.82em; color: #999;
                 padding: 2px 8px; }
</style>
"""
st.html(CUSTOM_CSS)


# =============================================================================
# SESSION STATE
# =============================================================================

_defaults = {
    "messages": [],
    "root_folder": os.path.dirname(os.path.abspath(__file__)),
    "current_file": None,
    "current_content": "",
    "run_stdout": "",
    "run_stderr": "",
    "run_returncode": None,
    "pending_action": None,
    "apply_code": None,
    "font_size": 14,
    "chat_pending_images": [],
    "vision_upload_key": 0,
    "clipboard_paste_key": 0,
    "metric_chat_vision_turns": 0,
    "metric_chat_text_turns": 0,
    "rolling_summary": "",
    "metric_chat_disk_saves": 0,
    "metric_chat_summaries": 0,
    "metric_chat_trim_fallback": 0,
    "snippet_draft": None,
    "snippet_draft_id": 0,
    "snippet_retrieval_enabled": True,
    "snippet_top_k": 5,
    "ollama_temperature": 0.4,
    "ollama_top_p": 0.4,
    "_show_all": False,
    "_recursive": False,
}
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Load disk settings only once per Streamlit session. Re-applying every rerun would
# overwrite widget updates with stale JSON before save_settings runs at end of script.
if not st.session_state.get("_ui_settings_hydrated"):
    asp.apply_loaded_settings(st.session_state, app_dir=_APP_DIR)
    st.session_state["_ui_settings_hydrated"] = True

# Reload sidebar chat from `data/chats/<stem>.json` once per browser session (e.g. after F5).
# File / folder changes still call `load_chat_into_session` in the file browser below.
if not st.session_state.get("_chat_archive_session_loaded"):
    cp.load_chat_into_session(
        st.session_state,
        str(st.session_state.get("root_folder") or _APP_DIR),
        st.session_state.get("current_file"),
    )
    st.session_state["_chat_archive_session_loaded"] = True

# =============================================================================
# SIDEBAR — Part 1: Header + model & adjustments (single expander)
# =============================================================================

with st.sidebar:
    st.markdown("## 🤖 Code Assistant")

    with st.expander("🎛️ Model & adjustments", expanded=False):
        models = get_ollama_models()
        has_models = bool(models)
        if has_models:
            asp.fix_model_selection(st.session_state, models)

        if has_models:
            selected_model = st.selectbox("Model", models, key="sel_model")
        else:
            st.warning("⚠️ Cannot connect to Ollama — chat disabled.")
            selected_model = None

        with st.expander("🎛️ Sampling (Ollama)", expanded=False):
            st.caption(
                "Applied as `options` on each **sidebar chat** request; overrides Modelfile defaults for this app."
            )
            st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                format="%.2f",
                key="ollama_temperature",
                help="Lower → more deterministic (often better for code). Typical: 0.2–0.5.",
            )
            st.number_input(
                "top_p",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                key="ollama_top_p",
                help="Nucleus sampling. Lower → tighter token choice. Match your model (e.g. 0.95) or try 0.4.",
            )

        with st.expander("⚙️ Settings", expanded=False):
            prompt_mode = st.selectbox("Persona", list(PROMPT_MODES.keys()), key="prompt_mode")
            custom_prompt = st.text_area(
                "Custom system prompt", height=80, key="custom_prompt",
                placeholder="Override the persona above...",
            )
            code_theme = st.selectbox(
                "Code theme",
                ["monokai", "native", "friendly", "vs", "dracula"],
                key="code_theme",
            )

        with st.expander("🌐 Web research", expanded=False):
            st.radio(
                "When to search",
                ["Off", "Auto", "Always"],
                index=1,
                horizontal=True,
                key="research_mode",
                help=(
                    "Auto: stderr, bug/how-to questions, or build/create/project-style asks "
                    "(see web_research_coding.should_run_research_auto)"
                ),
            )
            st.toggle(
                "Deep page read",
                value=True,
                key="research_deep",
                help="Fetch full text for top results (slower, richer context)",
            )
            st.toggle(
                "LLM query planner",
                value=True,
                key="research_llm_planner",
                help="Use the selected model to craft targeted search queries",
            )

        with st.expander("📋 Code planning", expanded=False):
            st.radio(
                "When to plan first",
                ["Off", "Auto", "Always"],
                index=1,
                horizontal=True,
                key="planning_mode",
                help=(
                    "Before codegen, the same chat model builds a JSON roadmap + todos. "
                    "Auto: longer asks and build/implement-style prompts. "
                    "Queued screenshots are included when you use a vision model. "
                    "You confirm or edit the plan, then Proceed (images go to the writer too)."
                ),
            )
            st.toggle(
                "Brief web snack for planner",
                value=False,
                key="planner_snack_research",
                help="2 quick DuckDuckGo searches (snippets only) fed into the planner for API/library hints",
            )

        with st.expander("📚 Snippet library (cross-project)", expanded=False):
            st.caption(
                "Extract **reusable code patterns** from the **currently selected .py** file — "
                "not the whole project. The model proposes snippets; you **review and confirm** before save."
            )
            _snip_ok = (
                bool(st.session_state.get("current_file"))
                and str(st.session_state.current_file).lower().endswith(".py")
                and (st.session_state.get("current_content") or "").strip()
            )
            if st.button(
                "🔍 Analyze current file for snippets",
                use_container_width=True,
                disabled=not (_snip_ok and has_models),
                key="snippet_analyze_btn",
                help="Uses the selected model to propose task-labeled snippets from this file only.",
            ):
                with st.spinner("Analyzing current file…"):
                    _fn = os.path.basename(st.session_state.current_file or "file.py")
                    _entries, _err = sl.propose_snippets(
                        OLLAMA_BASE_URL,
                        selected_model,
                        _fn,
                        st.session_state.get("current_content") or "",
                    )
                if _err:
                    st.error(_err)
                else:
                    st.session_state.snippet_draft = _entries
                    st.session_state.snippet_draft_id = (
                        int(st.session_state.get("snippet_draft_id", 0)) + 1
                    )
                    st.rerun()
            st.toggle(
                "Inject top similar snippets into chat (spaCy retrieval)",
                key="snippet_retrieval_enabled",
                help="Prepends up to K saved snippets by similarity — used in **chat** and **code planning** (roadmap).",
            )
            st.number_input(
                "Top-K snippets",
                min_value=1,
                max_value=20,
                key="snippet_top_k",
                help="How many library snippets to attach before each assistant reply.",
            )
            try:
                _n_lib = sl.library_entry_count()
            except Exception:
                _n_lib = "?"
            st.caption(
                f"Library: `{sl.library_path().name}`  ·  **{_n_lib}** entries (also stored as `entry_count` in JSON)  ·  "
                + (
                    "Select a non-empty `.py` file to enable."
                    if not _snip_ok
                    else "Review panel opens in the main area below."
                )
            )

        if st.session_state.current_file:
            _cf = st.session_state.current_file
            _fh = hashlib.md5(_cf.encode("utf-8", errors="replace")).hexdigest()[:12]
            with st.expander("📜 File versions (last 10)", expanded=False):
                st.caption(
                    "Before each **Save** or **Apply**, the previous file on disk is stored here (ring). "
                    "**Restore** writes that snapshot back; the current file is snapshotted first. "
                    "History files are named from the `.py` file (under `data/file_versions/`)."
                )
                _meta = fvh.list_versions_meta(_cf)
                if not _meta:
                    st.info("No snapshots yet. Change and save this file, or use **Apply**.")
                else:
                    for _m in _meta:
                        _si = int(_m["storage_index"])
                        st.markdown(
                            f"**{_m['at']}** · {_m['chars']:,} chars  \n"
                            f"`{_m['line1']}`"
                        )
                        if st.button(
                            "↩ Restore this version",
                            key=f"fvh_r_{_fh}_{_si}",
                            use_container_width=True,
                        ):
                            _txt = fvh.get_version_content(_cf, _si)
                            if _txt is None:
                                st.error("Could not load that snapshot.")
                            else:
                                _ok, _err = save_file(_cf, _txt)
                                if _ok:
                                    st.session_state.current_content = _txt
                                    st.session_state["_ace_ver"] = (
                                        st.session_state.get("_ace_ver", 0) + 1
                                    )
                                    st.toast("Restored from snapshot.", icon="↩")
                                    st.rerun()
                                else:
                                    st.error(_err or "Save failed")


# =============================================================================
# MAIN AREA — File Browser
# =============================================================================

st.markdown("## 💻 Local LLM-Assisted Coding Tool")

_snippet_draft = st.session_state.get("snippet_draft")
if _snippet_draft:
    _vid = int(st.session_state.get("snippet_draft_id", 0))
    _snip_nlp = sl.try_get_nlp()
    _snip_lib_data = sl.load_library() if _snip_nlp else None
    if _snip_nlp and _snip_lib_data is not None:
        sl.ensure_library_embeddings(_snip_lib_data, _snip_nlp)
    with st.container(border=True):
        st.markdown("##### 📚 Snippet proposals — review before save")
        st.caption(
            "Edit **task** or **code** if needed. Uncheck **Include** to skip a row. "
            "Then save to the shared library (usable from any project folder)."
        )
        for _i, _e in enumerate(_snippet_draft):
            st.markdown(f"**{_i + 1}.**")
            _r1, _r2 = st.columns([0.14, 0.86])
            with _r1:
                st.checkbox(
                    "Include",
                    value=True,
                    key=f"sd_inc_{_vid}_{_i}",
                    help="Save this snippet to the library",
                )
            with _r2:
                st.text_input(
                    "Task / when to use this",
                    value=_e.get("task_prompt", ""),
                    key=f"sd_task_{_vid}_{_i}",
                )
            st.text_area(
                "Code",
                value=_e.get("code", ""),
                height=min(220, max(100, _e.get("code", "").count("\n") * 18 + 50)),
                key=f"sd_code_{_vid}_{_i}",
            )
            _tp_cur = (
                st.session_state.get(f"sd_task_{_vid}_{_i}") or _e.get("task_prompt") or ""
            ).strip()
            _cd_cur = (
                st.session_state.get(f"sd_code_{_vid}_{_i}") or _e.get("code") or ""
            ).strip()
            st.markdown(
                sl.library_similarity_caption(
                    _tp_cur,
                    _cd_cur,
                    _data=_snip_lib_data,
                    _nlp=_snip_nlp,
                )
            )
            if _i < len(_snippet_draft) - 1:
                st.divider()
        _sc1, _sc2, _sc3 = st.columns([1, 1, 2])
        with _sc1:
            if st.button("💾 Save checked to library", type="primary", key="snippet_save_btn"):
                _to_save = []
                for _i in range(len(_snippet_draft)):
                    if not st.session_state.get(f"sd_inc_{_vid}_{_i}", True):
                        continue
                    _tp = (st.session_state.get(f"sd_task_{_vid}_{_i}") or "").strip()
                    _cd = (st.session_state.get(f"sd_code_{_vid}_{_i}") or "").strip()
                    if _tp and _cd:
                        _to_save.append({"task_prompt": _tp, "code": _cd})
                with st.spinner(
                    "Saving to library… (merge uses your Ollama model and can take tens of seconds per merged row)"
                ):
                    _stats, _serr = sl.upsert_confirmed_snippets(
                        _to_save,
                        source_file=st.session_state.get("current_file") or "",
                        source_root=st.session_state.get("root_folder") or "",
                        ollama_base_url=OLLAMA_BASE_URL,
                        ollama_model=selected_model,
                    )
                if _serr:
                    st.error(f"Save failed: {_serr}")
                else:
                    st.session_state.snippet_draft = None
                    _a = int(_stats.get("added", 0))
                    _m = int(_stats.get("merged", 0))
                    _msg = []
                    if _a:
                        _msg.append(f"added {_a}")
                    if _m:
                        _msg.append(f"merged {_m}")
                    st.toast(
                        "Library: " + ", ".join(_msg) if _msg else "No changes.",
                        icon="✅",
                    )
                    st.rerun()
        with _sc2:
            if st.button("Discard draft", key="snippet_discard_btn"):
                st.session_state.snippet_draft = None
                st.rerun()
        with _sc3:
            st.caption(f"Path: `{sl.library_path()}`")

root_path = Path(st.session_state.root_folder)
py_ext = None if st.session_state.get("_show_all", False) else {".py"}
files = list_files(
    st.session_state.root_folder,
    extensions=py_ext,
    show_all=st.session_state.get("_show_all", False),
    recursive=st.session_state.get("_recursive", False),
)

file_strs = [str(f) for f in files]
display_names = []
for f in files:
    try:
        display_names.append(str(f.relative_to(root_path)))
    except ValueError:
        display_names.append(str(f))

with st.expander("📁 File selection", expanded=False):
    fb_c1, fb_c2, fb_c3, fb_c4, fb_c5 = st.columns([2.6, 3.8, 1, 1, 1])
    with fb_c1:
        root = st.text_input(
            "folder", value=st.session_state.root_folder,
            label_visibility="collapsed", placeholder="Project folder path...",
        )
        if root != st.session_state.root_folder:
            old_root = st.session_state.root_folder
            cp.save_chat_from_session(
                st.session_state,
                old_root,
                st.session_state.get("current_file"),
            )
            st.session_state.root_folder = root
            cp.load_chat_into_session(
                st.session_state,
                st.session_state.root_folder,
                st.session_state.get("current_file"),
            )
            st.rerun()
    with fb_c2:
        pick_col, new_col = st.columns([3.4, 2.2])
        with pick_col:
            if file_strs:
                default_idx = 0
                if st.session_state.current_file in file_strs:
                    default_idx = file_strs.index(st.session_state.current_file)
                selected_path = st.selectbox(
                    "file", options=file_strs,
                    format_func=lambda p: display_names[file_strs.index(p)],
                    index=default_idx, label_visibility="collapsed",
                )
                content, err = load_file(selected_path)
                if err:
                    st.error(f"Cannot read file: {err}")
                else:
                    prev_fp = st.session_state.get("current_file")
                    if prev_fp != selected_path:
                        cp.save_chat_from_session(
                            st.session_state,
                            st.session_state.root_folder,
                            prev_fp,
                        )
                        cp.load_chat_into_session(
                            st.session_state,
                            st.session_state.root_folder,
                            selected_path,
                        )
                    st.session_state.current_file = selected_path
                    st.session_state.current_content = content
            elif root_path.is_dir():
                ext_label = "all" if st.session_state.get("_show_all") else ".py"
                st.info(f"No {ext_label} files found.")
            else:
                st.error("Folder not found.")
        with new_col:
            if root_path.is_dir():
                with st.form("new_py_form", clear_on_submit=True):
                    nf = st.text_input(
                        "New file",
                        placeholder="my_script.py — or leave empty for untitled.py",
                        help="Letters, numbers, dots, dashes, underscores. .py added if missing.",
                    )
                    submitted = st.form_submit_button(
                        "Create file",
                        type="primary",
                        use_container_width=True,
                    )
                if submitted:
                    new_path, cerr = create_new_python_file(
                        st.session_state.root_folder, nf
                    )
                    if cerr:
                        st.error(cerr)
                    else:
                        old_fp = st.session_state.get("current_file")
                        cp.save_chat_from_session(
                            st.session_state,
                            st.session_state.root_folder,
                            old_fp,
                        )
                        cp.load_chat_into_session(
                            st.session_state,
                            st.session_state.root_folder,
                            new_path,
                        )
                        st.session_state.current_file = new_path
                        st.session_state.current_content = NEW_PY_TEMPLATE
                        bk = f"_backed_up_{new_path}"
                        if bk in st.session_state:
                            del st.session_state[bk]
                        st.session_state["_ace_ver"] = (
                            st.session_state.get("_ace_ver", 0) + 1
                        )
                        st.toast(f"Created {os.path.basename(new_path)}", icon="✅")
                        st.rerun()
    with fb_c3:
        st.session_state["_show_all"] = st.checkbox(
            "All files",
            value=bool(st.session_state.get("_show_all", False)),
        )
    with fb_c4:
        st.session_state["_recursive"] = st.checkbox(
            "Recursive",
            value=bool(st.session_state.get("_recursive", False)),
        )
    with fb_c5:
        new_fs = st.number_input(
            "🔤 Font", min_value=10, max_value=28, step=2,
            value=st.session_state.font_size, key="font_input",
            help="Adjust editor & output font size",
        )
        if new_fs != st.session_state.font_size:
            st.session_state.font_size = new_fs
            st.session_state["_ace_ver"] = st.session_state.get("_ace_ver", 0) + 1
            st.rerun()


# =============================================================================
# MAIN AREA — Editor / outline / run (below file selection)
# =============================================================================

if st.session_state.current_file:
    cur_content = st.session_state.current_content or ""
    cur_file = st.session_state.current_file
    cur_name = os.path.basename(cur_file)
    total_lines = cur_content.count("\n") + 1

    st.markdown(
        f'<div class="file-info">📄 {cur_name} &mdash; {total_lines} lines &mdash; '
        f'{len(cur_content):,} chars &mdash; {cur_file}</div>',
        unsafe_allow_html=True,
    )

    # --- Editor (learner Ace + floating Explain, or streamlit-ace, or read-only fallback) ---
    ace_theme = ACE_THEME_MAP.get(
        st.session_state.get("code_theme", "monokai"), "monokai"
    )
    editor_height = 600  # fixed; was scaled by line count (min 300 / max 600)
    editor_lang = "python" if cur_name.endswith(".py") else "text"

    apply_pending = st.session_state.get("apply_code")
    show_monaco_diff = bool(apply_pending) and HAS_MONACO_DIFF and mdc is not None

    if apply_pending and not show_monaco_diff:
        st.info(
            "Pending changes from chat — Monaco diff is unavailable. "
            "Apply the assistant’s proposal as-is, or discard it."
        )
        fb1, fb2 = st.columns(2)
        with fb1:
            if st.button(
                "✅ Apply proposal",
                key="apply_pending_fallback_apply",
                type="primary",
                use_container_width=True,
            ):
                ok, err = save_file(cur_file, apply_pending)
                if ok:
                    st.session_state.current_content = apply_pending
                    st.session_state.apply_code = None
                    st.session_state["_ace_ver"] = (
                        st.session_state.get("_ace_ver", 0) + 1
                    )
                    st.toast("Changes applied!", icon="✅")
                    st.rerun()
                else:
                    st.error(f"Save failed: {err}")
        with fb2:
            if st.button(
                "❌ Discard",
                key="apply_pending_fallback_discard",
                use_container_width=True,
            ):
                st.session_state.apply_code = None
                st.rerun()

    if show_monaco_diff:
        diff_evt = mdc.monaco_diff_view(
            original_text=cur_content,
            modified_text=apply_pending,
            height=editor_height,
            theme=st.session_state.get("code_theme", "monokai"),
            language=editor_lang,
            font_size=st.session_state.font_size,
            key=f"mdiff_{cur_file}_{st.session_state.get('_ace_ver', 0)}",
        )
        if isinstance(diff_evt, dict) and diff_evt.get("action"):
            eid = diff_evt.get("event_id")
            if eid is not None and eid == st.session_state.get("_mdiff_seen_eid"):
                pass
            else:
                if eid is not None:
                    st.session_state["_mdiff_seen_eid"] = eid
                if diff_evt.get("action") == "accept":
                    merged = diff_evt.get("value", "")
                    ok, err = save_file(cur_file, merged)
                    if ok:
                        st.session_state.current_content = merged
                        st.session_state.apply_code = None
                        st.session_state["_ace_ver"] = (
                            st.session_state.get("_ace_ver", 0) + 1
                        )
                        st.toast("Changes applied!", icon="✅")
                        st.rerun()
                    else:
                        st.error(f"Apply failed: {err}")
                elif diff_evt.get("action") == "cancel":
                    st.session_state.apply_code = None
                    st.rerun()
        if HAS_EXPLAIN_EDITOR and esel is not None:
            esel.render_explain_popup()
    elif HAS_EXPLAIN_EDITOR and esel is not None:
        evt = esel.explain_editor(
            value=cur_content,
            height=editor_height,
            theme=ace_theme,
            language=editor_lang,
            font_size=st.session_state.font_size,
            key=f"eace_{cur_file}_{st.session_state.get('_ace_ver', 0)}",
        )
        if isinstance(evt, dict) and evt.get("action") == "save":
            edited = evt.get("value", "")
            if edited == cur_content:
                st.toast("No changes to save.", icon="ℹ️")
            else:
                ok, err = save_file(cur_file, edited)
                if ok:
                    st.session_state.current_content = edited
                    st.toast("Saved", icon="💾")
                else:
                    st.error(f"Save failed: {err}")
        elif isinstance(evt, dict) and evt.get("action") == "explain":

            def _persist_buf(b: str):
                ok, err = save_file(cur_file, b)
                if ok:
                    st.session_state.current_content = b
                return ok, err

            esel.process_explain_click(
                evt,
                base_url=OLLAMA_BASE_URL,
                model=selected_model,
                filepath=cur_file,
                basename=cur_name,
                language=editor_lang,
                persist_buffer=_persist_buf,
            )
        esel.render_explain_popup()
    elif HAS_ACE:
        edited = st_ace(
            value=cur_content,
            language=editor_lang,
            theme=ace_theme,
            height=editor_height,
            font_size=st.session_state.font_size,
            show_gutter=True,
            show_print_margin=False,
            wrap=True,
            auto_update=False,
            key=f"ace_{cur_file}_{st.session_state.get('_ace_ver', 0)}",
        )
        if st.button("💾 Save to file", key="manual_save_st_ace", use_container_width=True):
            if edited == cur_content:
                st.toast("No changes to save.", icon="ℹ️")
            else:
                ok, err = save_file(cur_file, edited)
                if ok:
                    st.session_state.current_content = edited
                    st.toast("Saved", icon="💾")
                else:
                    st.error(f"Save failed: {err}")
    else:
        st.code(cur_content, language="python", line_numbers=True)
        st.warning("Install `streamlit-ace` for the full editor: `pip install streamlit-ace`")

    if st.session_state.get("apply_code") and HAS_MONACO_DIFF and mdc is not None:
        st.caption(
            "**Review diff (Monaco):** Left = saved file, right = proposed (editable). "
            "**Apply merged changes** saves and closes the preview. **Cancel** discards the proposal. "
            "Inline **Explain** is unavailable until you apply or cancel."
        )
    # --- Run current file + shell (one compact row) ---
    proj_dir = st.session_state.root_folder
    shell_runner = get_shell_runner()
    run_pair_col, shell_txt_col, shell_go_col, shell_stop_col = st.columns([2, 4, 1, 1])
    with run_pair_col:
        run_btn_c, fix_btn_c = st.columns(2)
        with run_btn_c:
            if st.button("▶️ Run", use_container_width=True, type="primary", key="run_py_btn"):
                with st.spinner("Running… (no timeout — stop the process from Task Manager if needed)"):
                    stdout, stderr, rc = run_python_file(cur_file)
                st.session_state.run_stdout = stdout
                st.session_state.run_stderr = stderr
                st.session_state.run_returncode = rc
        # After Run updates session_state, compute in the same run (not above Run — that was one rerun stale).
        _fix_err_active = bool(
            (st.session_state.get("run_stderr") or "").strip()
            and st.session_state.get("run_returncode") != 0
            and has_models
        )
        with fix_btn_c:
            if st.button(
                "🔧 Fix Error",
                use_container_width=True,
                key="fix_err_btn",
                disabled=not _fix_err_active,
                help=(
                    "Send stderr to the assistant when the last **Run** failed"
                    if _fix_err_active
                    else "Runs after **Run** reports an error and a model is selected"
                ),
            ):
                err_prompt = (
                    "The code produced an error when executed. Fix the bug and "
                    "return the complete corrected file.\n\n"
                    f"Error output:\n```\n{st.session_state.run_stderr}\n```"
                )
                st.session_state.pending_action = err_prompt
                st.rerun()
    with shell_txt_col:
        shell_cmd = st.text_input(
            "Shell command",
            label_visibility="collapsed",
            placeholder="pip install …  or  streamlit run script.py  —  project folder",
            key="shell_cmd_line",
        )
    with shell_go_col:
        if st.button("Run cmd", use_container_width=True, type="primary", key="run_shell_btn"):
            cmd = (shell_cmd or "").strip()
            if not cmd:
                st.warning("Enter a command first.")
            elif shell_runner is None:
                with st.spinner("Running command..."):
                    so, se, rc = run_shell_command(shell_cmd, proj_dir)
                prefix = f"$ {cmd}\n\n"
                st.session_state.run_stdout = prefix + (so or "")
                st.session_state.run_stderr = se or ""
                st.session_state.run_returncode = rc
            else:
                if shell_runner.is_running():
                    shell_runner.stop()
                    st.toast("Stopped previous command", icon="⏹️")
                st.session_state.pop("_shell_done_sig", None)
                shell_runner.start(cmd, proj_dir)
                st.rerun()
    with shell_stop_col:
        if shell_runner:
            if st.button(
                "⏹️ Stop",
                use_container_width=True,
                key="stop_shell_btn",
                disabled=not shell_runner.is_running(),
                help="Kill the shell command and its child processes (e.g. nested Streamlit)",
            ):
                shell_runner.stop()
                st.rerun()

    sync_shell_runner_to_session(shell_runner)
    aux_r1, aux_r3 = st.columns([1, 5])
    with aux_r1:
        if shell_runner and shell_runner.is_running():
            if st.button("🔄 Refresh output", key="shell_refresh_log"):
                st.rerun()
    with aux_r3:
        if shell_runner and shell_runner.is_running():
            st.caption("⏳ Shell running — use **Stop** or **Refresh**.")
        elif st.session_state.run_returncode is not None:
            if st.session_state.run_returncode == 0:
                st.caption(f"✅ Exit code {st.session_state.run_returncode}")
            else:
                st.caption(f"❌ Exit code {st.session_state.run_returncode}")

    if st.session_state.run_stdout or st.session_state.run_stderr:
        parts = []
        if st.session_state.run_stdout:
            parts.append(html.escape(st.session_state.run_stdout))
        if st.session_state.run_stderr:
            escaped = html.escape(st.session_state.run_stderr)
            parts.append(f'<span class="err">{escaped}</span>')
        combined = "\n".join(parts)
        term_h = 400  # fixed output panel height (px)
        term_font = st.session_state.font_size
        terminal_html = f"""<!DOCTYPE html>
<html><head><style>
    html, body {{ margin:0; padding:0; }}
    .term {{
        background: #1e1e2e; color: #cdd6f4;
        font-family: 'Cascadia Code','Fira Code','Consolas','Courier New',monospace;
        font-size: {term_font}px; line-height: 1.6;
        padding: 12px 16px; white-space: pre-wrap; word-wrap: break-word;
    }}
    .err {{ color: #f38ba8; }}
</style></head>
<body><div class="term">{combined}</div></body></html>"""
        components.html(terminal_html, height=term_h, scrolling=True)

else:
    st.markdown("---")
    st.markdown(
        "### 👋 Welcome\n"
        "Enter a folder path above and select a file to get started.  \n"
        "Then use the **sidebar** to chat with your local LLM about the code."
    )


# =============================================================================
# SIDEBAR — Part 2: Chat
# =============================================================================

with st.sidebar:
    st.divider()
    st.markdown("##### 💬 Chat")
    st.caption(
        cp.chat_status_line(
            st.session_state.root_folder,
            st.session_state.get("current_file"),
        )
    )

    chat_container = st.container(height=320)

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "user" and msg.get("images"):
                    cols = st.columns(min(4, max(1, len(msg["images"]))))
                    for i, im in enumerate(msg["images"]):
                        if not isinstance(im, dict) or not im.get("b64"):
                            continue
                        with cols[i % len(cols)]:
                            try:
                                st.image(
                                    base64.standard_b64decode(im["b64"]),
                                    width=120,
                                )
                            except Exception:
                                st.caption("…")
                if msg.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for s in msg["sources"]:
                            st.markdown(
                                f"- [{s.get('title', 'link')}]({s.get('href', '')})"
                            )

    # --- Plan review (roadmap before codegen) ---
    draft = st.session_state.get("planning_draft")
    if draft and isinstance(draft, dict) and draft.get("plan"):
        st.warning("📋 **Plan ready** — answer optional questions, edit if needed, then proceed.")
        pimgs = draft.get("planner_images") or []
        if pimgs:
            st.caption(
                f"📎 **{len(pimgs)}** image(s) were used for planning and will be sent again when you **Proceed**."
            )
            thumbs = st.columns(min(4, max(1, len(pimgs))))
            for i, im in enumerate(pimgs):
                with thumbs[i % len(thumbs)]:
                    try:
                        st.image(
                            base64.standard_b64decode(im["b64"]),
                            width=100,
                        )
                    except Exception:
                        st.caption("…")
        with st.expander("Roadmap & checklist", expanded=True):
            st.markdown(draft.get("display_md") or ph.format_plan_markdown(draft["plan"]))
        plan = draft["plan"]
        oqs = plan.get("open_questions") or []
        for q in oqs:
            qid = q.get("id", "q")
            opts = q.get("options") or []
            if opts:
                st.radio(
                    q.get("question", qid),
                    opts,
                    index=ph.radio_index_for_question(q),
                    key=f"plan_q_{qid}",
                )
            else:
                st.text_input(q.get("question", qid), key=f"plan_t_{qid}")
        st.text_area(
            "Edit plan (optional — overrides structured plan for the writer if non-empty)",
            height=100,
            key="plan_manual_override",
        )
        pr1, pr2 = st.columns(2)
        with pr1:
            if st.button("✅ Proceed", type="primary", use_container_width=True):
                answers = {}
                for q in oqs:
                    qid = q.get("id", "")
                    opts = q.get("options") or []
                    if opts:
                        answers[qid] = st.session_state.get(f"plan_q_{qid}", opts[0])
                    else:
                        answers[qid] = st.session_state.get(f"plan_t_{qid}", "") or ""
                manual = (st.session_state.get("plan_manual_override") or "").strip()
                if manual:
                    plan_suffix = "### User-edited plan\n" + manual
                else:
                    plan_suffix = ph.format_plan_for_system_prompt(plan, answers)
                up = draft["user_prompt"]
                plan_imgs = ovc.normalize_pending_count(
                    list(draft.get("planner_images") or [])
                )
                user_msg: dict = {"role": "user", "content": up}
                if plan_imgs:
                    user_msg["images"] = plan_imgs
                    user_msg["vision_schema"] = ovc.VISION_MESSAGE_SCHEMA_VERSION
                finalize_user_msg_for_chat(user_msg, selected_model)
                st.session_state.messages.append(user_msg)
                st.session_state.pop("planning_draft", None)
                run_writer_pipeline(up, plan_suffix, chat_container, selected_model)
        with pr2:
            if st.button("Cancel", use_container_width=True):
                d = st.session_state.pop("planning_draft", None)
                if isinstance(d, dict) and d.get("planner_images"):
                    st.session_state.chat_pending_images = ovc.normalize_pending_count(
                        list(d["planner_images"])
                    )
                st.rerun()
        pr3, pr4 = st.columns(2)
        with pr3:
            if st.button("🔄 Regenerate", use_container_width=True):
                with st.spinner("Re-planning…"):
                    pnew, raw, _ = planner_invoke(
                        selected_model,
                        draft["user_prompt"],
                        planner_images=draft.get("planner_images"),
                    )
                if pnew:
                    pnew = ph.validate_plan_shape(pnew)
                    draft["plan"] = pnew
                    draft["display_md"] = ph.format_plan_markdown(pnew)
                    draft["raw"] = (raw or "")[:2000]
                    st.success("Plan updated.")
                else:
                    st.error("Regenerate failed; try Proceed or Cancel.")
                st.rerun()
        with pr4:
            if st.button("💪 Stronger plan", use_container_width=True):
                with st.spinner("Improving plan…"):
                    pnew, raw = ph.stronger_plan_sync(
                        OLLAMA_BASE_URL,
                        selected_model,
                        draft["plan"],
                        context_images=draft.get("planner_images"),
                    )
                if pnew:
                    pnew = ph.validate_plan_shape(pnew)
                    draft["plan"] = pnew
                    draft["display_md"] = ph.format_plan_markdown(pnew)
                    draft["raw"] = (raw or "")[:2000]
                    st.success("Stronger plan applied.")
                else:
                    st.error("Stronger pass failed; try Regenerate.")
                st.rerun()

    _clip_queue_full = (
        len(st.session_state.get("chat_pending_images") or [])
        >= ovc.MAX_IMAGES_PER_MESSAGE
    )
    _img_col, _paste_col = st.columns(2, vertical_alignment="center")
    _clip_added = False
    with _img_col:
        with st.expander("🖼️ Images for chat (vision models)", expanded=False):
            if ovc.pillow_status() != "ok":
                st.warning(
                    "Install **Pillow** (`pip install pillow`) for EXIF strip, max-edge downscaling (~2048px), "
                    "and JPEG/PNG normalization."
                )

            pend = list(st.session_state.get("chat_pending_images") or [])
            queue_full = len(pend) >= ovc.MAX_IMAGES_PER_MESSAGE
            if (
                pend
                and has_models
                and selected_model
                and not ovc.is_likely_vision_model(selected_model)
            ):
                st.warning(
                    "Selected model name may be **text-only**. Attached images might be ignored or cause errors. "
                    "Prefer a **vision** tag (e.g. **qwen3.5**, **llava**, **qwen2-vl**)."
                )
            if queue_full:
                st.warning(
                    f"Queue full (**{ovc.MAX_IMAGES_PER_MESSAGE}** images max; total base64 payload also capped). "
                    "Remove one to add more."
                )

            vuk = st.session_state.get("vision_upload_key", 0)
            batch = st.file_uploader(
                "Screenshot / image files",
                type=["png", "jpg", "jpeg", "webp", "gif"],
                accept_multiple_files=True,
                key=f"vision_fu_{vuk}",
                label_visibility="collapsed",
            )
            iq1, iq2 = st.columns(2)
            with iq1:
                if st.button(
                    "➕ Add to queue",
                    use_container_width=True,
                    key="vision_add_btn",
                    disabled=queue_full,
                ):
                    if queue_full:
                        st.warning("Queue is full.")
                    elif batch:
                        pending = list(st.session_state.get("chat_pending_images") or [])
                        files = batch if isinstance(batch, list) else [batch]
                        for f in files:
                            raw = f.getvalue()
                            pic = ovc.process_uploaded_image(raw, getattr(f, "name", "") or "")
                            if pic:
                                pending.append(pic)
                        st.session_state.chat_pending_images = ovc.normalize_pending_count(
                            pending
                        )
                        st.session_state.vision_upload_key = vuk + 1
                        st.rerun()
                    else:
                        st.warning("Choose one or more images first.")
            with iq2:
                if st.button("🗑️ Clear queue", use_container_width=True, key="vision_clear_q"):
                    st.session_state.chat_pending_images = []
                    st.session_state.vision_upload_key = vuk + 1
                    st.rerun()

            pend = st.session_state.get("chat_pending_images") or []
            if pend:
                st.caption(
                    f"{len(pend)} in queue · ~{ovc.total_base64_chars(pend):,} b64 chars "
                    f"(max {ovc.MAX_TOTAL_BASE64_CHARS:,})"
                )
                for i, im in enumerate(pend):
                    row = st.columns([4, 1, 1, 1])
                    with row[0]:
                        try:
                            st.image(
                                base64.standard_b64decode(im["b64"]),
                                width=120,
                                caption=im.get("name") or "",
                            )
                        except Exception:
                            st.text("…")
                    with row[1]:
                        if st.button("✕", key=f"vq_rm_{vuk}_{i}", help="Remove from queue"):
                            pend.pop(i)
                            st.session_state.chat_pending_images = pend
                            st.rerun()
                    with row[2]:
                        if i > 0 and st.button("↑", key=f"vq_up_{vuk}_{i}"):
                            pend[i - 1], pend[i] = pend[i], pend[i - 1]
                            st.session_state.chat_pending_images = pend
                            st.rerun()
                    with row[3]:
                        if i < len(pend) - 1 and st.button("↓", key=f"vq_dn_{vuk}_{i}"):
                            pend[i + 1], pend[i] = pend[i], pend[i + 1]
                            st.session_state.chat_pending_images = pend
                            st.rerun()

    with _paste_col:
        if cpc.HAS_STREAMLIT_PASTE and not _clip_queue_full:
            _clip_added = cpc.try_clipboard_paste(
                st.session_state, ovc, allow_append=True
            )

    if _clip_added:
        st.toast("Clipboard image added to queue", icon="📋")
        st.rerun()

    # --- Chat input form ---
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "msg", height=68,
            placeholder="Message or @file.py — optional if you only send images.",
            label_visibility="collapsed",
        )
        send_btn = st.form_submit_button("Send ➤", use_container_width=True)

    # --- Clear chat ---
    if st.button("🗑️ Clear", use_container_width=True):
        _sr = st.session_state.get("_shell_runner")
        if _sr is not None:
            _sr.stop()
        st.session_state.pop("_shell_done_sig", None)
        st.session_state.messages = []
        st.session_state.rolling_summary = ""
        cp.clear_archive_file(
            st.session_state.root_folder,
            st.session_state.get("current_file"),
        )
        st.session_state.apply_code = None
        st.session_state.run_stdout = ""
        st.session_state.run_stderr = ""
        st.session_state.run_returncode = None
        st.session_state.pop("planning_draft", None)
        st.session_state.pop("_explain_popup_open", None)
        st.session_state.pop("_explain_stream_pending", None)
        st.session_state.pop("_explain_deep_stream_pending", None)
        st.session_state.pop("_explain_api_messages", None)
        st.session_state.pop("_explain_deep_api_messages", None)
        st.session_state.chat_pending_images = []
        st.session_state.vision_upload_key = (
            st.session_state.get("vision_upload_key", 0) + 1
        )
        st.session_state.clipboard_paste_key = (
            st.session_state.get("clipboard_paste_key", 0) + 1
        )
        st.rerun()

    # --- Determine what to process ---
    prompt_to_process = None
    if st.session_state.get("planning_draft"):
        if st.session_state.pending_action:
            st.session_state.pending_action = None
        elif send_btn and (user_input.strip() or st.session_state.get("chat_pending_images")):
            st.warning("Finish or **Cancel** the active plan review before sending a new message.")
    else:
        if st.session_state.pending_action:
            prompt_to_process = st.session_state.pending_action
            st.session_state.pending_action = None
        elif send_btn and (user_input.strip() or st.session_state.get("chat_pending_images")):
            prompt_to_process = user_input.strip()
            if not prompt_to_process and st.session_state.get("chat_pending_images"):
                prompt_to_process = (
                    "The user attached image(s) with no text. Describe what you see "
                    "and offer any help relevant to coding or debugging."
                )

    # --- Process the message ---
    # Only attach queued images on a real form Send (not pending_action e.g. Fix Error).
    attach = (
        ovc.normalize_pending_count(
            list(st.session_state.get("chat_pending_images") or [])
        )
        if send_btn
        else []
    )
    if prompt_to_process is not None and has_models and selected_model:
        pmode = st.session_state.get("planning_mode", "Auto")
        use_plan = ph.planning_mode_should_run(pmode, prompt_to_process)

        if use_plan:
            with st.spinner("Building roadmap (same model as chat)…"):
                plan_dict, raw, _ = planner_invoke(
                    selected_model,
                    prompt_to_process,
                    planner_images=attach if attach else None,
                )
            if plan_dict:
                plan_dict = ph.validate_plan_shape(plan_dict)
                if "plan_manual_override" in st.session_state:
                    del st.session_state["plan_manual_override"]
                st.session_state.planning_draft = {
                    "user_prompt": prompt_to_process,
                    "plan": plan_dict,
                    "raw": (raw or "")[:2000],
                    "display_md": ph.format_plan_markdown(plan_dict),
                    "planner_images": list(attach) if attach else [],
                }
                if attach:
                    st.session_state.chat_pending_images = []
                    st.session_state.vision_upload_key = (
                        st.session_state.get("vision_upload_key", 0) + 1
                    )
                st.rerun()
            else:
                st.warning(
                    "Planner did not return valid JSON — continuing without roadmap. "
                    f"Raw (truncated): {(raw or '')[:300]}"
                )
                use_plan = False

        if not use_plan:
            user_msg: dict = {"role": "user", "content": prompt_to_process}
            if attach:
                user_msg["images"] = attach
                user_msg["vision_schema"] = ovc.VISION_MESSAGE_SCHEMA_VERSION
                st.session_state.chat_pending_images = []
                st.session_state.vision_upload_key = (
                    st.session_state.get("vision_upload_key", 0) + 1
                )
            finalize_user_msg_for_chat(user_msg, selected_model)
            st.session_state.messages.append(user_msg)
            run_writer_pipeline(
                prompt_to_process,
                "",
                chat_container,
                selected_model,
            )

    elif prompt_to_process and not has_models:
        st.warning("Start Ollama to chat (`ollama serve`).")


asp.save_settings(st.session_state)
