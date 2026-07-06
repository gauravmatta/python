# =============================================================================
# Cross-project snippet library: spaCy embeddings, NumPy similarity, LLM merge
# =============================================================================

from __future__ import annotations

import difflib
from collections import defaultdict
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from chat_persistence import strip_think_tags

_log = logging.getLogger("snippet_library")

SCHEMA_VERSION = 2
MAX_SNIPPETS_PER_EXTRACT = 15
MAX_FILE_CHARS = 120_000
EXTRACT_TIMEOUT = 180.0
MERGE_TIMEOUT = 120.0

SPACY_MODEL = "en_core_web_md"
# spaCy doc vectors are noisy for "same idea, different wording" — 0.95 rarely triggers.
MERGE_SIMILARITY_THRESHOLD = 0.88
# If embedding is only moderate but normalized code is almost the same, still merge.
MERGE_HYBRID_EMB_MIN = 0.76
MERGE_HYBRID_CODE_RATIO = 0.90
MERGE_STRONG_CODE_RATIO = 0.96
MERGE_STRONG_CODE_EMB_MIN = 0.68
RETRIEVAL_TOP_K = 5
# Total Markdown budget for injected snippets (system prompt). Increase if you use large snippets.
RETRIEVAL_MAX_CHARS = 14_000
# Previously 8 — short questions ("pygame help") retrieved nothing. Empty query still skips.
RETRIEVAL_MIN_QUERY_CHARS = 1
# Hybrid score: spaCy cosine (semantic) + lexical overlap (identifiers / words vs snippet text).
# Lexical helps code-specific terms (pygame, asyncio) where word vectors are weak.
RETRIEVAL_EMB_WEIGHT = 0.52
RETRIEVAL_LEX_WEIGHT = 0.48
# spaCy cosine is clipped to [0, 1] before mixing (negative cosines treated as 0).
# English stopwords removed from query/doc token sets so "how to use" does not drown signal.
RETRIEVAL_LEX_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "and",
        "or",
        "not",
        "no",
        "as",
        "at",
        "it",
        "if",
        "so",
        "we",
        "you",
        "your",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "this",
        "that",
        "these",
        "those",
        "from",
        "use",
        "using",
        "into",
        "by",
        "my",
        "me",
        "i",
        "get",
        "got",
        "make",
        "just",
        "like",
        "also",
        "then",
        "than",
        "here",
        "there",
        "out",
        "up",
        "down",
        "all",
        "any",
        "some",
        "one",
        "two",
        "way",
        "will",
        "have",
        "has",
        "had",
        "need",
        "help",
        "please",
        "want",
        "add",
        "new",
    }
)

DATA_DIR = Path(__file__).resolve().parent / "data"
LIBRARY_JSON = DATA_DIR / "snippet_library.json"

_nlp = None
_NLP_ERROR: Optional[str] = None

EXTRACT_SYSTEM = f"""You analyze a single Python source file and extract SMALL reusable pieces.

Return **only** a JSON array (no markdown fences, no commentary). Each element must be an object:
{{"task_prompt": "<short description of when this code is useful>", "code": "<python snippet>"}}

Rules:
- `task_prompt`: one line naming **libraries, domain, and intent** so search matches user questions
  (e.g. "pygame snake grid movement and food spawn", "HTTP GET with timeout and retries", "Parse CSV with utf-8-sig").
  Include **import names** (`pygame`, `asyncio`, `fastapi`) when relevant — retrieval uses these keywords.
- `code`: the minimal snippet that stands alone or with obvious imports — a function, class, or a few lines — NOT the whole file.
- Prefer 3–12 snippets if the file has distinct concerns; fewer is fine for small files.
- Skip boilerplate, empty `if __name__` blocks, and trivial one-liners unless they encode a non-obvious pattern.
- Do not include secrets, API keys, or passwords. If you see any, omit that snippet.
- Maximum {MAX_SNIPPETS_PER_EXTRACT} snippets."""

MERGE_SYSTEM = """You merge two similar Python snippet records into ONE reusable entry.
Return **only** a JSON object (no markdown fences):
{"task_prompt": "<one clear line: when to use this>", "code": "<merged Python code>"}

Rules:
- Preserve useful lines from **both** code blocks; remove exact duplicates and dead code.
- The task_prompt should cover both intents if needed (short, comprehensive).
- Output valid Python in `code` (complete minimal snippet).
- Do not invent features not present in either snippet."""


def library_path() -> Path:
    return LIBRARY_JSON


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_nlp():
    """Lazy-load spaCy model (call from worker / Streamlit)."""
    global _nlp, _NLP_ERROR
    if _nlp is not None:
        return _nlp
    if _NLP_ERROR:
        raise RuntimeError(_NLP_ERROR)
    try:
        import spacy

        _nlp = spacy.load(SPACY_MODEL)
        return _nlp
    except Exception as e:
        _NLP_ERROR = str(e)
        _log.warning("spaCy load failed: %s", e)
        raise RuntimeError(_NLP_ERROR) from e


def try_get_nlp():
    """Return nlp or None if spaCy/model unavailable."""
    try:
        return get_nlp()
    except Exception:
        return None


def embedding_concat_text(task_prompt: str, code: str) -> str:
    return f"{task_prompt.strip()}\n\n{code.strip()}"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def vector_from_doc(nlp, text: str) -> np.ndarray:
    return l2_normalize(np.asarray(nlp(text).vector, dtype=np.float64))


def embed_snippet(nlp, task_prompt: str, code: str) -> np.ndarray:
    return vector_from_doc(nlp, embedding_concat_text(task_prompt, code))


def embed_query_text(nlp, query: str) -> np.ndarray:
    return vector_from_doc(nlp, (query or "").strip()[:12000])


def retrieval_token_set(text: str) -> set:
    """Lowercase identifiers and words for lexical overlap (favors code-like tokens)."""
    text = (text or "").lower()
    raw = set(re.findall(r"[a-z_][a-z0-9_]{1,}", text))
    return {t for t in raw if t not in RETRIEVAL_LEX_STOPWORDS and len(t) >= 2}


def lexical_snippet_score(query: str, task_prompt: str, code: str) -> float:
    """
    Return 0..1: share of query tokens present in task_prompt + code (recall-style).
    Pulls up snippets when the user names libraries, symbols, or patterns embeddings miss.
    """
    q = retrieval_token_set(query)
    if not q:
        return 0.0
    doc = retrieval_token_set(f"{task_prompt}\n{code}")
    if not doc:
        return 0.0
    return len(q & doc) / len(q)


def hybrid_retrieval_scores(
    sims: np.ndarray,
    row_to_idx: List[int],
    entries: List[Dict[str, Any]],
    query: str,
) -> np.ndarray:
    """Per-row hybrid scores (same length as sims). Cosine clipped to [0,1], then weighted mix."""
    n = int(sims.shape[0])
    lex = np.zeros(n, dtype=np.float64)
    for r in range(n):
        ei = row_to_idx[r]
        e = entries[ei]
        lex[r] = lexical_snippet_score(
            query,
            str(e.get("task_prompt") or ""),
            str(e.get("code") or ""),
        )
    emb = np.maximum(0.0, sims.astype(np.float64))
    w_e = RETRIEVAL_EMB_WEIGHT
    w_l = RETRIEVAL_LEX_WEIGHT
    return w_e * emb + w_l * lex


def ensure_library_embeddings(data: Dict[str, Any], nlp) -> bool:
    """Fill missing `embedding` lists in-place (migration). Returns True if any change."""
    entries: List[Dict[str, Any]] = list(data.get("entries") or [])
    changed = False
    for e in entries:
        if not isinstance(e, dict):
            continue
        emb = e.get("embedding")
        if (
            not emb
            or not isinstance(emb, list)
            or len(emb) < 50
        ):
            tp = str(e.get("task_prompt") or "")
            cd = str(e.get("code") or "")
            if tp or cd:
                vec = embed_snippet(nlp, tp, cd)
                e["embedding"] = vec.tolist()
                changed = True
    data["entries"] = entries
    data["embedding_model"] = SPACY_MODEL
    return changed


def build_embedding_matrix(
    entries: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[int]]:
    """Rows L2-normalized; returns (N, D) and original indices."""
    rows: List[np.ndarray] = []
    idxs: List[int] = []
    dim: Optional[int] = None
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        emb = e.get("embedding")
        if not emb or not isinstance(emb, list):
            continue
        v = l2_normalize(np.asarray(emb, dtype=np.float64))
        if v.size == 0:
            continue
        if dim is None:
            dim = int(v.size)
        elif int(v.size) != dim:
            _log.debug("Skip entry %s: embedding dim %s != %s", i, v.size, dim)
            continue
        rows.append(v)
        idxs.append(i)
    if not rows:
        return np.zeros((0, 1)), []
    return np.stack(rows, axis=0), idxs


def normalize_code_dedup(code: str) -> str:
    """Strip line endings / blank lines so small formatting edits still match."""
    lines = [ln.rstrip() for ln in (code or "").splitlines()]
    return "\n".join(lines).strip()


def find_merge_entry_index(
    v_new: np.ndarray,
    entries: List[Dict[str, Any]],
    M: np.ndarray,
    row_to_entry_idx: List[int],
    incoming_code: str,
) -> int:
    """
    Pick **one** library row: among rows that pass merge eligibility rules, choose the
    **highest embedding cosine** to this incoming snippet. Returns -1 if none qualify.
    """
    ent, _sim = best_merge_target_and_similarity(
        v_new, entries, M, row_to_entry_idx, incoming_code
    )
    return ent


def best_merge_target_and_similarity(
    v_new: np.ndarray,
    entries: List[Dict[str, Any]],
    M: np.ndarray,
    row_to_entry_idx: List[int],
    incoming_code: str,
) -> Tuple[int, float]:
    """
    Returns (entry_index, cosine_sim) for the single best library row, or (-1, -1.0).
    Best = maximum cosine similarity among rows that satisfy any merge rule.
    """
    if M.shape[0] == 0 or not row_to_entry_idx:
        return -1, -1.0
    sims = (M @ v_new).reshape(-1)
    inc = normalize_code_dedup(incoming_code)
    best_ent = -1
    best_sim = -1.0
    for r, ent_i in enumerate(row_to_entry_idx):
        sim_e = float(sims[r])
        ex = normalize_code_dedup(str(entries[ent_i].get("code") or ""))
        cr = difflib.SequenceMatcher(None, inc, ex).ratio() if inc or ex else 0.0
        eligible = (
            sim_e > MERGE_SIMILARITY_THRESHOLD
            or (
                sim_e >= MERGE_HYBRID_EMB_MIN
                and cr >= MERGE_HYBRID_CODE_RATIO
            )
            or (
                cr >= MERGE_STRONG_CODE_RATIO
                and sim_e >= MERGE_STRONG_CODE_EMB_MIN
            )
        )
        if eligible and sim_e > best_sim:
            best_sim = sim_e
            best_ent = ent_i
    return best_ent, best_sim


def library_similarity_caption(
    task_prompt: str,
    code: str,
    *,
    _data: Optional[Dict[str, Any]] = None,
    _nlp=None,
) -> str:
    """
    Short markdown line: max embedding cosine vs library, max code ratio, merge decision.
    Used in the review UI (does not write the library file).
    Pass _data and _nlp from the caller to avoid reloading JSON for every row.
    """
    tp = (task_prompt or "").strip()
    cd = (code or "").strip()
    if not tp or not cd:
        return "_Add task + code to see similarity vs saved library._"
    nlp = _nlp if _nlp is not None else try_get_nlp()
    if nlp is None:
        return (
            "**Similarity:** spaCy not available — install `spacy` and "
            f"`python -m spacy download {SPACY_MODEL}`."
        )
    data = _data if _data is not None else load_library()
    ensure_library_embeddings(data, nlp)
    entries = [e for e in data.get("entries") or [] if isinstance(e, dict)]
    v_new = embed_snippet(nlp, tp, cd)
    M, row_to_entry_idx = build_embedding_matrix(entries)
    if M.shape[0] == 0:
        return "**Library:** empty — this row will be added as **new**."
    sims = (M @ v_new).reshape(-1)
    max_emb = float(np.max(sims))
    inc = normalize_code_dedup(cd)
    max_cr = 0.0
    for ent_i in row_to_entry_idx:
        ex = normalize_code_dedup(str(entries[ent_i].get("code") or ""))
        cr = difflib.SequenceMatcher(None, inc, ex).ratio() if inc or ex else 0.0
        max_cr = max(max_cr, cr)
    merge_i = find_merge_entry_index(v_new, entries, M, row_to_entry_idx, cd)
    if merge_i >= 0:
        decision = "**MERGE** (matches existing entry)"
    else:
        decision = "**NEW** (below merge rules)"
    rules = (
        f"*Merge if: embedding > {MERGE_SIMILARITY_THRESHOLD}, or "
        f"(embedding ≥ {MERGE_HYBRID_EMB_MIN} and code ≥ {MERGE_HYBRID_CODE_RATIO}), or "
        f"(code ≥ {MERGE_STRONG_CODE_RATIO} and embedding ≥ {MERGE_STRONG_CODE_EMB_MIN}). "
        f"Among eligible rows, the **highest cosine** wins. In one save, only **one** incoming row "
        f"can merge into a given library entry (highest sim); others become new.*"
    )
    return (
        f"**vs library:** max embedding **{max_emb:.3f}** · max code similarity **{max_cr:.3f}** → {decision}\n\n"
        + rules
    )


def parse_snippets_array(raw: str) -> Tuple[List[Dict[str, str]], Optional[str]]:
    text = strip_think_tags(raw or "")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    else:
        text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return [], "Could not find a JSON array in the model response."
    blob = text[start : end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        return [], f"Invalid JSON: {e}"
    if not isinstance(data, list):
        return [], "Top-level JSON must be an array."
    out: List[Dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        tp = str(item.get("task_prompt") or item.get("prompt") or "").strip()
        code = str(item.get("code") or "").strip()
        if not tp or not code:
            continue
        out.append({"task_prompt": tp, "code": code})
        if len(out) >= MAX_SNIPPETS_PER_EXTRACT:
            break
    if not out:
        return [], "No valid snippets after parsing (need task_prompt + code)."
    return out, None


def parse_merge_object(raw: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    text = strip_think_tags(raw or "")
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "No JSON object in merge response."
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        return None, str(e)
    if not isinstance(obj, dict):
        return None, "Merge JSON is not an object."
    tp = str(obj.get("task_prompt") or "").strip()
    code = str(obj.get("code") or "").strip()
    if not tp or not code:
        return None, "Merge result missing task_prompt or code."
    return {"task_prompt": tp, "code": code}, None


def ollama_chat_sync(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    timeout: float = 120.0,
    num_predict: int = 8000,
) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": num_predict},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content", "") or ""


def merge_snippets_with_llm(
    base_url: str,
    model: str,
    existing: Dict[str, Any],
    incoming: Dict[str, str],
) -> Tuple[Dict[str, str], Optional[str]]:
    user = (
        "### Existing (library)\n"
        f"task_prompt:\n{existing.get('task_prompt', '')}\n\n"
        f"code:\n```python\n{existing.get('code', '')}\n```\n\n"
        "### New (incoming)\n"
        f"task_prompt:\n{incoming.get('task_prompt', '')}\n\n"
        f"code:\n```python\n{incoming.get('code', '')}\n```\n"
    )
    try:
        raw = ollama_chat_sync(
            base_url,
            model,
            [
                {"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": user},
            ],
            timeout=MERGE_TIMEOUT,
            num_predict=4096,
        )
    except Exception as e:
        return fallback_merge(existing, incoming), str(e)
    merged, err = parse_merge_object(raw)
    if err or not merged:
        return fallback_merge(existing, incoming), err or "parse failed"
    return merged, None


def fallback_merge(existing: Dict[str, Any], incoming: Dict[str, str]) -> Dict[str, str]:
    return {
        "task_prompt": (
            str(existing.get("task_prompt") or "").strip()
            + " | merged: "
            + str(incoming.get("task_prompt") or "")[:240]
        ).strip()[:2000],
        "code": (
            str(existing.get("code") or "").rstrip()
            + "\n\n# --- merged from newer snippet ---\n\n"
            + str(incoming.get("code") or "")
        ).strip(),
    }


def propose_snippets(
    base_url: str,
    model: str,
    filename: str,
    content: str,
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    if not model:
        return [], "No model selected."
    if not content or not content.strip():
        return [], "File is empty."
    body = content
    truncated = False
    if len(body) > MAX_FILE_CHARS:
        body = body[:MAX_FILE_CHARS]
        truncated = True
    user_msg = (
        f"Filename: `{filename}`\n\n"
        f"{'(Content was truncated for analysis.)' if truncated else ''}\n\n"
        f"```python\n{body}\n```\n\n"
        "Respond with the JSON array only."
    )
    try:
        raw = ollama_chat_sync(
            base_url,
            model,
            [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            timeout=EXTRACT_TIMEOUT,
            num_predict=8000,
        )
    except Exception as e:
        _log.warning("Snippet extract request failed: %s", e)
        return [], str(e)
    return parse_snippets_array(raw)


def load_library() -> Dict[str, Any]:
    ensure_data_dir()
    if not LIBRARY_JSON.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "entry_count": 0,
            "embedding_model": SPACY_MODEL,
            "entries": [],
        }
    try:
        data = json.loads(LIBRARY_JSON.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(data, dict):
            return {
                "schema_version": SCHEMA_VERSION,
                "entry_count": 0,
                "embedding_model": SPACY_MODEL,
                "entries": [],
            }
        if "entries" not in data or not isinstance(data["entries"], list):
            data["entries"] = []
        if data.get("schema_version", 1) < SCHEMA_VERSION:
            data["schema_version"] = SCHEMA_VERSION
        data.setdefault("embedding_model", SPACY_MODEL)
        data["entry_count"] = len(data["entries"])
        return data
    except json.JSONDecodeError:
        _log.warning("Corrupt snippet library; starting fresh.")
        return {
            "schema_version": SCHEMA_VERSION,
            "entry_count": 0,
            "embedding_model": SPACY_MODEL,
            "entries": [],
        }


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", errors="replace")
    tmp.replace(path)


def serialize_library_json(data: Dict[str, Any]) -> str:
    """
    Top-level field order: schema_version, entry_count, embedding_model,
    optional updated_at, entries. entry_count always matches len(entries).
    """
    raw_entries = data.get("entries")
    entries: List[Any] = raw_entries if isinstance(raw_entries, list) else []
    n = len(entries)
    out: Dict[str, Any] = {
        "schema_version": int(data.get("schema_version", SCHEMA_VERSION)),
        "entry_count": n,
        "embedding_model": str(data.get("embedding_model") or SPACY_MODEL),
    }
    u = data.get("updated_at")
    if u:
        out["updated_at"] = u
    out["entries"] = entries
    return json.dumps(out, ensure_ascii=False, indent=2)


def library_entry_count() -> int:
    """Current number of snippet entries (reads JSON; cheap)."""
    data = load_library()
    ent = data.get("entries")
    if isinstance(ent, list):
        return len(ent)
    return int(data.get("entry_count") or 0)


def upsert_confirmed_snippets(
    items: List[Dict[str, str]],
    *,
    source_file: str,
    source_root: str,
    ollama_base_url: str,
    ollama_model: Optional[str],
) -> Tuple[Dict[str, int], Optional[str]]:
    """
    Embed each snippet vs the **initial** library snapshot. Each incoming row picks the
    **single** library entry with highest eligible cosine similarity. If several incoming
    rows target the **same** library entry in one save, only the one with **highest**
    similarity merges there; the others are saved as **new** entries. At most one LLM
    merge per library row per save.
    """
    if not items:
        return {"added": 0, "merged": 0}, None
    nlp = try_get_nlp()
    if nlp is None:
        return (
            {"added": 0, "merged": 0},
            "spaCy model not available. Install: pip install spacy && python -m spacy download "
            + SPACY_MODEL,
        )

    data = load_library()
    ensure_library_embeddings(data, nlp)
    entries: List[Dict[str, Any]] = [e for e in data.get("entries") or [] if isinstance(e, dict)]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        root_short = str(Path(source_root).resolve()) if source_root else ""
    except OSError:
        root_short = source_root or ""
    base_name = os.path.basename(source_file or "") if source_file else ""

    indexed: List[Tuple[int, str, str]] = []
    for k, it in enumerate(items):
        tp = str(it.get("task_prompt") or "").strip()
        cd = str(it.get("code") or "").strip()
        if tp and cd:
            indexed.append((k, tp, cd))

    if not indexed:
        return {"added": 0, "merged": 0}, None

    added = 0
    merged = 0

    M, row_to_entry_idx = build_embedding_matrix(entries)
    if M.shape[0] == 0:
        for _k, tp, cd in indexed:
            v_new = embed_snippet(nlp, tp, cd)
            entries.append(_new_entry_dict(tp, cd, v_new, base_name, root_short, now))
            added += 1
        data["entries"] = entries
        data["schema_version"] = SCHEMA_VERSION
        data["embedding_model"] = SPACY_MODEL
        data["updated_at"] = now
        try:
            atomic_write(LIBRARY_JSON, serialize_library_json(data))
        except OSError as e:
            return {"added": 0, "merged": 0}, str(e)
        return {"added": added, "merged": merged}, None

    # Phase 1: best merge target per incoming row (same library snapshot for all).
    merge_wants: List[Tuple[int, int, float]] = []
    new_only_ks: List[int] = []
    for k, tp, cd in indexed:
        v_new = embed_snippet(nlp, tp, cd)
        ent_i, sim = best_merge_target_and_similarity(
            v_new, entries, M, row_to_entry_idx, cd
        )
        if ent_i < 0:
            new_only_ks.append(k)
        else:
            merge_wants.append((k, ent_i, sim))

    # Phase 2: at most one incoming per library row per save — highest similarity wins.
    by_ent: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for k, ent_i, sim in merge_wants:
        by_ent[ent_i].append((k, sim))

    winner_by_ent: Dict[int, int] = {}
    loser_ks = set()
    for ent_i, lst in by_ent.items():
        k_win = max(lst, key=lambda x: x[1])[0]
        winner_by_ent[ent_i] = k_win
        for k, sim in lst:
            if k != k_win:
                loser_ks.add(k)

    # Phase 3: apply merges (one LLM call per winning library row).
    for ent_i, k in winner_by_ent.items():
        it = items[k]
        tp = str(it.get("task_prompt") or "").strip()
        cd = str(it.get("code") or "").strip()
        existing = entries[ent_i]
        if ollama_model:
            merged_body, _err = merge_snippets_with_llm(
                ollama_base_url,
                ollama_model,
                existing,
                {"task_prompt": tp, "code": cd},
            )
        else:
            merged_body = fallback_merge(existing, {"task_prompt": tp, "code": cd})
        new_vec = embed_snippet(nlp, merged_body["task_prompt"], merged_body["code"])
        merge_count = int(existing.get("merge_count") or 0) + 1
        entries[ent_i] = {
            "id": existing.get("id") or str(uuid.uuid4()),
            "task_prompt": merged_body["task_prompt"],
            "code": merged_body["code"],
            "embedding": new_vec.tolist(),
            "language": "python",
            "source_file": base_name or existing.get("source_file", ""),
            "source_root": root_short or existing.get("source_root", ""),
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "merge_count": merge_count,
        }
        merged += 1

    # Phase 4: append new entries (no merge + conflict losers).
    append_ks = set(new_only_ks) | loser_ks
    for k in sorted(append_ks):
        it = items[k]
        tp = str(it.get("task_prompt") or "").strip()
        cd = str(it.get("code") or "").strip()
        if not tp or not cd:
            continue
        v_new = embed_snippet(nlp, tp, cd)
        entries.append(_new_entry_dict(tp, cd, v_new, base_name, root_short, now))
        added += 1

    data["entries"] = entries
    data["schema_version"] = SCHEMA_VERSION
    data["embedding_model"] = SPACY_MODEL
    data["updated_at"] = now
    try:
        atomic_write(LIBRARY_JSON, serialize_library_json(data))
    except OSError as e:
        return {"added": 0, "merged": 0}, str(e)
    return {"added": added, "merged": merged}, None


def _new_entry_dict(
    task_prompt: str,
    code: str,
    vec: np.ndarray,
    source_file: str,
    source_root: str,
    now: str,
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "task_prompt": task_prompt,
        "code": code,
        "embedding": vec.tolist(),
        "language": "python",
        "source_file": source_file,
        "source_root": source_root,
        "created_at": now,
        "updated_at": now,
        "merge_count": 0,
    }


def format_top_k_snippets_block(
    entries_subset: List[Dict[str, Any]],
) -> str:
    parts = [
        "\n\n## Relevant saved snippets (from your library)\n",
        "These were ranked for the current question (semantic + keyword overlap).\n",
        "When a snippet fits the task: **reuse its patterns, names, and structure**; keep behavior consistent.\n",
        "Only adapt when the file or request clearly requires something different.\n",
    ]
    total = 0
    for i, e in enumerate(entries_subset, 1):
        tp = str(e.get("task_prompt") or "")
        cd = str(e.get("code") or "")
        block = f"\n### {i}. {tp}\n```python\n{cd}\n```\n"
        if total + len(block) > RETRIEVAL_MAX_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts)


def relevant_snippets_markdown(
    user_query: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    enabled: bool = True,
) -> str:
    """
    Top-K saved snippets as one Markdown block: **hybrid** spaCy cosine + lexical overlap
    (identifiers / words in the query vs task_prompt + code). Empty if disabled, no library,
    or empty query. Used by chat system prompt and planner.
    """
    if not enabled:
        return ""
    q = (user_query or "").strip()
    if len(q) < RETRIEVAL_MIN_QUERY_CHARS:
        return ""
    nlp = try_get_nlp()
    if nlp is None:
        return ""
    data = load_library()
    entries = [e for e in data.get("entries") or [] if isinstance(e, dict)]
    if not entries:
        return ""
    if ensure_library_embeddings(data, nlp):
        data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            atomic_write(LIBRARY_JSON, serialize_library_json(data))
        except OSError:
            pass
    entries = [e for e in data.get("entries") or [] if isinstance(e, dict)]
    M, row_to_idx = build_embedding_matrix(entries)
    if M.shape[0] == 0:
        return ""
    qv = embed_query_text(nlp, q)
    sims = M @ qv
    hybrid = hybrid_retrieval_scores(sims, row_to_idx, entries, q)
    k = max(1, int(top_k))
    order = np.argsort(-hybrid)[:k]
    picked: List[Dict[str, Any]] = []
    for j in order:
        if j < 0 or j >= len(row_to_idx):
            continue
        ei = row_to_idx[int(j)]
        picked.append(entries[ei])
    if not picked:
        return ""
    return format_top_k_snippets_block(picked)


def append_relevant_snippets_to_system(
    system_msg: str,
    user_query: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    enabled: bool = True,
) -> str:
    block = relevant_snippets_markdown(
        user_query, top_k=top_k, enabled=enabled
    )
    if not block:
        return system_msg
    return system_msg + block


# Backwards compatibility: old callers
def append_entries(
    items: List[Dict[str, str]],
    *,
    source_file: str,
    source_root: str,
) -> Tuple[int, Optional[str]]:
    stats, err = upsert_confirmed_snippets(
        items,
        source_file=source_file,
        source_root=source_root,
        ollama_base_url="http://localhost:11434",
        ollama_model=None,
    )
    if err:
        return 0, err
    return stats.get("added", 0) + stats.get("merged", 0), None
