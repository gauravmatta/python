# =============================================================================
# Web research for coding assistants (Phase 1 + thin Phase 2)
# =============================================================================
#
# DuckDuckGo search, optional page fetch, multi-query planning via Ollama,
# heuristics for errors/imports and greenfield projects, GitHub raw URL fetch,
# deduped context for LLM injection (no third-party search APIs).
#
# Optional deps (same as 47_websearch_chat.py):
#   pip install ddgs requests beautifulsoup4
# =============================================================================

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


# --- Phase 1: config ---------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_PAGE_CHARS = 6000
DEEP_FETCH_TOP_K = 6
MAX_RESULTS_PER_QUERY = 5
MAX_QUERIES = 10
MAX_TOTAL_RESEARCH_CHARS = 22000
SEARCH_TIMEOUT = 15
FETCH_TIMEOUT = 15
PLANNER_TIMEOUT = 30

# Domains to prefer when merging (soft boost: listed first after sort)
PREFERRED_DOMAINS = (
    "docs.python.org",
    "readthedocs.io",
    "pypi.org",
    "github.com",
    "raw.githubusercontent.com",
    "stackoverflow.com",
    "stackexchange.com",
    "wiki.python.org",
    "numpy.org",
    "pandas.pydata.org",
)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _domain_rank(url: str) -> int:
    d = _domain(url)
    for i, pref in enumerate(PREFERRED_DOMAINS):
        if pref in d:
            return i
    return len(PREFERRED_DOMAINS) + 1


# --- GitHub: HTML blob page -> raw code URL (no API; plain HTTP GET) ------------

_GITHUB_BLOB = re.compile(
    r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$",
    re.I,
)


def github_blob_to_raw_url(url: str) -> Optional[str]:
    """
    If URL is a github.com/blob/... file page, return raw.githubusercontent.com
    URL so we fetch source text instead of HTML shell.
    """
    if not url:
        return None
    u = url.split("?", 1)[0].strip().rstrip("/")
    m = _GITHUB_BLOB.match(u)
    if not m:
        return None
    user, repo, branch, path = m.groups()
    return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"


# --- Search & fetch (ported from 47_websearch_chat.py) -----------------------

def search_web(query: str, max_results: int = 5) -> List[dict]:
    """DuckDuckGo text search. Returns list of dicts with title, href, body."""
    if not HAS_DDGS or not query or not str(query).strip():
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(str(query).strip(), max_results=max_results))
    except Exception:
        return []


def fetch_page_text(url: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Fetch URL and extract text; GitHub blob pages use raw.githubusercontent.com."""
    if not url:
        return ""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CodeResearchBot/1.1)"}

    raw_u = github_blob_to_raw_url(url)
    if raw_u:
        try:
            rr = requests.get(raw_u, headers=headers, timeout=FETCH_TIMEOUT, allow_redirects=True)
            rr.raise_for_status()
            ct = (rr.headers.get("Content-Type") or "").lower()
            body = rr.text.strip()
            if body and ("text" in ct or "json" in ct or "javascript" in ct or len(body) < max_chars * 3):
                return body[:max_chars]
            if body and not body.lstrip().startswith("<"):
                return body[:max_chars]
        except Exception:
            pass

    if not HAS_BS4:
        return ""
    try:
        resp = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            tag.decompose()

        code_chunks: List[str] = []
        for tag in soup.find_all(["pre", "code"]):
            t = tag.get_text(separator="\n", strip=True)
            if len(t) > 8:
                code_chunks.append(t)
        code_block = "\n\n---\n\n".join(code_chunks) if code_chunks else ""

        text = soup.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 12]
        prose = "\n".join(lines)

        if code_block and prose:
            combined = (
                "[Code / pre blocks]\n"
                + code_block[: max_chars // 2 + 1000]
                + "\n\n[Page text]\n"
                + prose
            )
        elif code_block:
            combined = code_block
        else:
            combined = prose

        return combined[:max_chars] if len(combined) > max_chars else combined
    except Exception:
        return ""


def format_results_block(
    results: List[dict],
    deep_fetch: bool,
    deep_fetch_top_k: int,
    label: str = "Web research",
) -> Tuple[str, List[dict]]:
    """Build one markdown-ish context block + sources list."""
    if not results:
        return "", []

    sources: List[dict] = []
    lines = [f"--- {label} ---"]

    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("href", r.get("url", ""))
        snippet = r.get("body", r.get("excerpt", ""))
        sources.append({"title": title, "href": url})

        lines.append(f"\n[{i}] {title}")
        lines.append(f"    URL: {url}")

        if deep_fetch and i <= deep_fetch_top_k and url:
            page_text = fetch_page_text(url)
            if page_text:
                lines.append(f"    Content excerpt:\n{page_text}")
            else:
                lines.append(f"    Snippet: {snippet}")
        else:
            lines.append(f"    Snippet: {snippet}")

    lines.append("\n---")
    return "\n".join(lines), sources


# --- Phase 2: heuristics -----------------------------------------------------

_MODULE_NOT_FOUND = re.compile(
    r"ModuleNotFoundError:\s*No module named\s+['\"]?(\w+)['\"]?", re.I
)
_IMPORT_ERROR = re.compile(
    r"ImportError:.*?['\"](\w+)['\"]", re.I
)

# Heuristic: user is asking for a new program / app / feature (not only a one-line fix)
_PROJECT_GOAL_HINTS = (
    "create ",
    "build ",
    "new project",
    "scaffold",
    "from scratch",
    "implement ",
    "write a ",
    "develop ",
    "make a ",
    "game",
    "application",
    "dashboard",
    "cli ",
    "web app",
    "fastapi",
    "flask",
    "django",
    "streamlit",
    "similar ",
    "tutorial",
    "example project",
    "full project",
    "project that",
)


def _project_style_goal(g: str) -> bool:
    low = (g or "").lower()
    return any(h in low for h in _PROJECT_GOAL_HINTS)


def heuristic_search_queries(
    user_goal: str,
    code_snippet: str,
    error_text: str,
) -> List[str]:
    """Rule-based queries (docs, PyPI, SO, GitHub); project-style goals get extra GitHub-oriented lines."""
    priority: List[str] = []
    text = f"{error_text}\n{code_snippet}\n{user_goal}"

    for m in _MODULE_NOT_FOUND.finditer(text):
        mod = m.group(1)
        priority.append(f"{mod} site:pypi.org")
        priority.append(f"{mod} site:stackoverflow.com")
        priority.append(f"python {mod} site:readthedocs.io OR site:github.com")

    for m in _IMPORT_ERROR.finditer(text):
        mod = m.group(1)
        if mod and not any(mod in q for q in priority):
            priority.append(f"python ImportError {mod} site:stackoverflow.com")

    if error_text and len(error_text.strip()) > 20:
        first_line = error_text.strip().splitlines()[0][:120]
        priority.append(f"{first_line} site:stackoverflow.com")

    extra: List[str] = []
    g = (user_goal or "").strip()
    gs = g[:140].strip() if g else ""
    if gs and _project_style_goal(g):
        extra.append(f"{gs} python site:github.com")
        extra.append(f"{gs} python example site:github.com")
        extra.append(f"{gs} site:readthedocs.io")

    tail: List[str] = []
    if g:
        tail.append(f"{g} site:stackoverflow.com OR site:github.com")

    combined = priority + extra + tail

    seen = set()
    uniq: List[str] = []
    for q in combined:
        q = q.strip()
        if q and q not in seen and len(q) < 220:
            seen.add(q)
            uniq.append(q)
    return uniq[:MAX_QUERIES]


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ollama_chat_sync(
    model: str,
    messages: List[dict],
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: int = PLANNER_TIMEOUT,
) -> str:
    """Non-streaming Ollama /api/chat."""
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
        return _strip_think(content)
    except Exception:
        return ""


def plan_search_queries_llm(
    model: str,
    user_goal: str,
    code_snippet: str,
    error_text: str,
    chat_tail: str,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> List[str]:
    """Ask the model for search queries as a JSON array of strings."""
    code_snippet = (code_snippet or "")[:6000]
    error_text = (error_text or "")[:4000]
    chat_tail = (chat_tail or "")[:2000]

    sys = (
        "You plan DuckDuckGo web searches to help fix bugs OR implement new Python projects. "
        "Include queries that can surface example repos and tutorials: use site:github.com, "
        "site:stackoverflow.com, site:readthedocs.io, site:pypi.org where helpful. "
        "For greenfield apps (games, CLIs, APIs), add at least one query aimed at GitHub examples. "
        "Output ONLY a JSON array of 3 to 10 short search query strings, no markdown. "
        "Example: [\"pandas read_csv dtype site:stackoverflow.com\", \"pygame snake example site:github.com\"]"
    )
    user = (
        f"User goal / question:\n{user_goal}\n\n"
        f"Recent chat (truncated):\n{chat_tail}\n\n"
        f"Error / traceback (if any):\n{error_text or '(none)'}\n\n"
        f"Code (truncated):\n{code_snippet or '(none)'}\n\n"
        "JSON array only:"
    )
    raw = ollama_chat_sync(
        model,
        [{"role": "system", "content": sys}, {"role": "user", "content": user}],
        base_url=base_url,
        timeout=PLANNER_TIMEOUT,
    )
    if not raw:
        return []

    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            out = [str(x).strip() for x in data if str(x).strip() and len(str(x)) < 220]
            return out[:MAX_QUERIES]
    except json.JSONDecodeError:
        pass

    # Fallback: lines that look like queries
    lines = [ln.strip().strip("\",'") for ln in raw.splitlines() if len(ln.strip()) > 5]
    return [ln for ln in lines if not ln.startswith("[")][:MAX_QUERIES]


def merge_queries(
    llm_queries: List[str],
    heuristic_queries: List[str],
) -> List[str]:
    """Dedupe preserving order: LLM first, then heuristics."""
    seen = set()
    out: List[str] = []
    for q in llm_queries + heuristic_queries:
        q = q.strip()
        if not q or q in seen or len(q) > 220:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= MAX_QUERIES:
            break
    return out


def _dedupe_results_by_url(all_results: List[dict]) -> List[dict]:
    seen = set()
    merged: List[dict] = []
    for r in all_results:
        url = r.get("href", r.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(r)
    merged.sort(key=lambda r: _domain_rank(r.get("href", r.get("url", ""))))
    return merged


def build_research_context(
    user_goal: str,
    code_snippet: str = "",
    error_text: str = "",
    chat_tail: str = "",
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    deep_fetch: bool = True,
    use_llm_planner: bool = True,
    max_results_per_query: int = MAX_RESULTS_PER_QUERY,
    deep_fetch_top_k: int = DEEP_FETCH_TOP_K,
    max_total_chars: int = MAX_TOTAL_RESEARCH_CHARS,
) -> Tuple[str, List[dict]]:
    """
    Run multi-query web research and return (context_string, sources).

    Phase 1: search + optional deep fetch + char budget.
    Phase 2: LLM query planner + heuristics + URL dedupe + domain preference sort.
    """
    if not HAS_DDGS:
        return (
            "\n[Web research unavailable: install ddgs -> pip install ddgs]\n",
            [],
        )

    h = heuristic_search_queries(user_goal, code_snippet, error_text)
    llm_q: List[str] = []
    if use_llm_planner and model:
        llm_q = plan_search_queries_llm(
            model, user_goal, code_snippet, error_text, chat_tail, base_url=base_url
        )
    queries = merge_queries(llm_q, h)
    if not queries:
        queries = [f"{user_goal[:200]} python programming"]

    all_hits: List[dict] = []
    for q in queries:
        hits = search_web(q, max_results=max_results_per_query)
        all_hits.extend(hits)

    ranked = _dedupe_results_by_url(all_hits)
    # Cap unique URLs (snippets for all; deep fetch only for top_k)
    max_urls = max(deep_fetch_top_k * 2, 12)
    ranked = ranked[:max_urls]

    block, sources = format_results_block(
        ranked,
        deep_fetch=deep_fetch,
        deep_fetch_top_k=deep_fetch_top_k,
        label="Web research (docs, GitHub, Stack Overflow, etc.)",
    )

    if len(block) > max_total_chars:
        block = block[: max_total_chars - 80] + "\n\n[... research truncated ...]\n"

    instruction = (
        "\nUse the research above when it helps. Prefer official docs and "
        "reputable examples. If sources conflict or are weak, say so. "
        "Do not copy long passages verbatim; summarize patterns and cite ideas.\n"
    )
    return instruction + block, sources


def should_run_research_auto(
    user_message: str,
    error_text: str,
) -> bool:
    """Cheap heuristic for 'Auto' mode without an extra LLM call."""
    if error_text and len(error_text.strip()) > 15:
        return True
    msg = (user_message or "").lower()
    keys = (
        "error",
        "traceback",
        "exception",
        "fix",
        "broken",
        "not working",
        "install",
        "import",
        "module",
        "stackoverflow",
        "how do i",
        "how to",
        "documentation",
        "docs",
        "example",
        # Greenfield / project-style (trigger research without stderr)
        "create ",
        "build ",
        "project",
        "scaffold",
        "implement",
        "develop",
        "write a",
        "new app",
        "application",
        "game",
        "tutorial",
        "best practice",
        "pattern",
        "architecture",
        "similar",
        "reference",
        "github",
    )
    return any(k in msg for k in keys)
