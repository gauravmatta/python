# =============================================================================
# Pre-flight planning for coding assistant (roadmap + todos + optional questions)
# =============================================================================
# Used by 47.py. Planner uses the same Ollama model as chat (sync /api/chat).
# Features: rich JSON schema, import summary, chat digest, JSON repair pass,
# optional web snack, stronger-plan second pass, definition_of_done, step lint.
# =============================================================================

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

AUTO_PLAN_KEYWORDS = (
    "project", "build", "implement", "create", "refactor", "rewrite",
    "game", "application", "app", "train", "training", "dataset", "model",
    "complete", "full", "entire", "from scratch", "step by step",
    "architecture", "design", "multiple", "several files", "module",
)


def should_auto_plan(user_text: str) -> bool:
    """Heuristic: larger or keyword-heavy asks get a planning pass."""
    t = (user_text or "").strip()
    if len(t) >= 180:
        return True
    low = t.lower()
    if sum(1 for k in AUTO_PLAN_KEYWORDS if k in low) >= 2:
        return True
    if any(k in low for k in ("snake", "dashboard", "api", "backend", "frontend", "cli ")):
        return True
    return False


def planning_mode_should_run(mode: str, user_text: str) -> bool:
    if mode == "Always":
        return True
    if mode == "Off":
        return False
    return should_auto_plan(user_text)


def summarize_imports(code: str, max_lines: int = 35) -> str:
    """First N import/from lines for planner context."""
    lines = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            lines.append(s[:160])
        if len(lines) >= max_lines:
            break
    return "\n".join(lines) if lines else "(no import lines in excerpt)"


def build_chat_digest(
    messages: List[Dict[str, Any]],
    *,
    max_turns: int = 8,
    max_chars_per_message: int = 450,
    max_total_chars: int = 3500,
) -> str:
    """Recent user/assistant turns for planner context (excludes system)."""
    if not messages:
        return ""
    tail = messages[-max_turns:]
    parts: List[str] = []
    total = 0
    for m in tail:
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        chunk = content[:max_chars_per_message]
        if len(content) > max_chars_per_message:
            chunk += " …"
        line = f"[{role}]: {chunk}"
        if total + len(line) + 2 > max_total_chars:
            break
        parts.append(line)
        total += len(line) + 2
    return "\n\n".join(parts) if parts else ""


# --- JSON schema description (shared by planner, repair, stronger) ----------

PLAN_JSON_SCHEMA_DOC = """
Required keys: "goal" (string), "definition_of_done" (string), "steps" (array of objects with id, task, acceptance).

Full schema (include all keys; use [] or "" when not applicable):
{
  "goal": "one sentence outcome",
  "definition_of_done": "observable criteria: when this is true, the request is satisfied (tests, behavior, files)",
  "assumptions": ["..."],
  "constraints": ["e.g. stdlib only"],
  "non_goals": ["what we will NOT build"],
  "dependencies": ["libraries or stdlib-only"],
  "files_touched": [{"path": "main.py", "role": "create|modify|reference"}],
  "risks": ["short risk bullets"],
  "test_plan": "how to verify the result (one short paragraph)",
  "needs_research": false,
  "research_queries": ["optional extra search phrases if needs_research true"],
  "steps": [{"id": 1, "task": "...", "acceptance": "testable check"}],
  "open_questions": [{
    "id": "q1",
    "question": "...",
    "options": ["A", "B"],
    "recommended_default": "A"
  }]
}
Rules:
- definition_of_done: concrete, testable finish line (not vague like "works well").
- steps: 3–12 items, id from 1 upward; each task and acceptance must be non-empty strings.
- open_questions: 0–3. Include recommended_default matching one option when options non-empty; else "".
- needs_research: true only if official docs/API facts are uncertain.
- Keep JSON under ~8000 characters. Output ONLY JSON, no markdown fences.
"""


PLANNER_SYSTEM = f"""You are a senior software planner. The user asks for coding help.
Prefer editing the OPEN FILE unless they explicitly want new files.
When the user message includes attached images (screenshots, UI, diagrams, photos of errors), ground your plan in what is visible: reference concrete on-screen text, widgets, layout, and stack traces you can read from the image(s). Align steps, risks, and open_questions with that evidence.
Produce a concise, actionable plan.

{PLAN_JSON_SCHEMA_DOC}
"""


REPAIR_SYSTEM = f"""You fix malformed JSON. The user message contains broken or partial JSON that should match this schema:

{PLAN_JSON_SCHEMA_DOC}

Output ONLY a single valid JSON object. No markdown, no commentary."""


STRONGER_SYSTEM = f"""You improve an existing development plan. Read the JSON plan, mentally note weaknesses (vague steps, missing tests, unclear acceptance criteria, missing risks). If images are attached (same context as planning), use them to sharpen steps and acceptance checks against what is visible.
Output ONE replacement JSON plan that is strictly clearer and more actionable. Same schema as below.

{PLAN_JSON_SCHEMA_DOC}

Output ONLY the improved JSON object."""


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _image_b64_list(
    planner_images: Optional[List[Dict[str, Any]]],
) -> List[str]:
    if not planner_images:
        return []
    out: List[str] = []
    for im in planner_images:
        if isinstance(im, dict) and im.get("b64"):
            out.append(str(im["b64"]))
    return out


def _ollama_sync(base_url: str, model: str, messages: List[dict], timeout: int) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return _strip_think(resp.json().get("message", {}).get("content", "").strip())


def parse_planner_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        data = json.loads(s)
        if isinstance(data, dict) and "goal" in data and "steps" in data:
            return data
    except json.JSONDecodeError:
        pass
    return None


def repair_planner_json(
    base_url: str,
    model: str,
    broken_text: str,
    timeout: int = 60,
) -> Optional[Dict[str, Any]]:
    """Second LLM pass to fix invalid JSON."""
    if not broken_text or not broken_text.strip():
        return None
    try:
        raw = _ollama_sync(
            base_url,
            model,
            [
                {"role": "system", "content": REPAIR_SYSTEM},
                {
                    "role": "user",
                    "content": "Fix this into valid JSON only:\n\n"
                    + broken_text.strip()[:14000],
                },
            ],
            timeout=timeout,
        )
    except Exception:
        return None
    return parse_planner_json(raw)


def stronger_plan_sync(
    base_url: str,
    model: str,
    plan: Dict[str, Any],
    timeout: int = 90,
    context_images: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Critique and rewrite plan; returns (new_dict_or_None, raw_text)."""
    try:
        payload = json.dumps(plan, ensure_ascii=False, indent=2)[:12000]
        imgs = _image_b64_list(context_images)
        if imgs:
            user_content = (
                "Current plan JSON:\n"
                + payload
                + "\n\nThe same screenshot(s) or reference image(s) from planning are attached. "
                "Improve the plan so steps and acceptance criteria match what is visible where relevant."
            )
            timeout = max(timeout, 150)
        else:
            user_content = f"Current plan JSON:\n{payload}"
        user_msg: Dict[str, Any] = {"role": "user", "content": user_content}
        if imgs:
            user_msg["images"] = imgs
        raw = _ollama_sync(
            base_url,
            model,
            [
                {"role": "system", "content": STRONGER_SYSTEM},
                user_msg,
            ],
            timeout=timeout,
        )
    except Exception as e:
        return None, str(e)
    parsed = parse_planner_json(raw)
    if not parsed:
        parsed = repair_planner_json(base_url, model, raw, timeout=60)
    return parsed, raw


def build_planner_research_snack(
    user_goal: str,
    code_excerpt: str,
    error_excerpt: str,
    max_chars: int = 2800,
) -> str:
    """Lightweight DDG snippets for planner context (no deep fetch)."""
    try:
        import web_research_coding as wrc
    except ImportError:
        return ""

    queries = wrc.heuristic_search_queries(user_goal, code_excerpt, error_excerpt)[:2]
    if not queries:
        g = (user_goal or "").strip()[:120]
        queries = [f"{g} python documentation"] if g else []

    parts = []
    for query in queries[:2]:
        hits = wrc.search_web(query, max_results=3)
        block, _ = wrc.format_results_block(
            hits,
            deep_fetch=False,
            deep_fetch_top_k=0,
            label=f"Snack: {query[:80]}",
        )
        if block:
            parts.append(block)
    combined = "\n\n".join(parts)
    return combined[:max_chars] if combined else ""


def call_planner_sync(
    base_url: str,
    model: str,
    user_goal: str,
    *,
    filename: str,
    file_path: str = "",
    code_excerpt: str,
    error_excerpt: str,
    research_snack: str = "",
    snippet_library_context: str = "",
    referenced_files_context: str = "",
    messages_for_digest: Optional[List[Dict[str, Any]]] = None,
    planner_images: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 90,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Plan + optional JSON repair pass. Returns (parsed_dict_or_None, raw_text)."""
    code_excerpt = (code_excerpt or "")[:8000]
    error_excerpt = (error_excerpt or "")[:3000]
    imp = summarize_imports(code_excerpt)

    user_parts = [
        f"Open file (basename): {filename}",
    ]
    if file_path:
        user_parts.append(f"Full path: {file_path}")
    user_parts.append(f"User request:\n{user_goal}")

    if referenced_files_context and referenced_files_context.strip():
        user_parts.append(referenced_files_context.strip())

    digest = build_chat_digest(messages_for_digest or [])
    if digest:
        user_parts.append(
            "--- Recent chat (prior turns; current request is above) ---\n" + digest
        )

    if snippet_library_context and snippet_library_context.strip():
        user_parts.append(snippet_library_context.strip())

    user_parts.append(f"Detected import lines:\n{imp}")
    user_parts.append(f"Recent run error (if any):\n{error_excerpt or '(none)'}")
    user_parts.append(f"Current code (truncated):\n```\n{code_excerpt or '(empty)'}\n```")
    if research_snack and research_snack.strip():
        user_parts.append(
            "--- Brief web findings (snippets only) ---\n" + research_snack.strip()
        )
    imgs = _image_b64_list(planner_images)
    if imgs:
        user_parts.append(
            f"--- Attached: {len(imgs)} image(s) (screenshots/UI/diagrams). "
            "Use them together with the code and request above. ---"
        )
    user = "\n\n".join(user_parts)

    if imgs:
        timeout = max(timeout, 180)

    user_msg: Dict[str, Any] = {"role": "user", "content": user}
    if imgs:
        user_msg["images"] = imgs

    try:
        raw = _ollama_sync(
            base_url,
            model,
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                user_msg,
            ],
            timeout=timeout,
        )
    except Exception as e:
        return None, str(e)

    parsed = parse_planner_json(raw)
    if parsed is None:
        parsed = repair_planner_json(base_url, model, raw, timeout=min(75, timeout))
        if parsed is not None:
            raw = raw + "\n\n[repaired_json_ok]"
    return parsed, raw


def format_plan_markdown(plan: Dict[str, Any]) -> str:
    """Human-readable plan for the review UI."""
    lines = [f"**Goal:** {plan.get('goal', '')}", ""]
    dod = (plan.get("definition_of_done") or "").strip()
    if dod:
        lines.append(f"**Definition of done:** {dod}")
        lines.append("")

    warns = plan.get("_validation_warnings") or []
    if warns:
        lines.append("**Planner notes (auto-checks)**")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")

    ng = plan.get("non_goals") or []
    if ng:
        lines.append("**Non-goals**")
        for x in ng:
            lines.append(f"- {x}")
        lines.append("")

    ass = plan.get("assumptions") or []
    if ass:
        lines.append("**Assumptions**")
        for a in ass:
            lines.append(f"- {a}")
        lines.append("")

    con = plan.get("constraints") or []
    if con:
        lines.append("**Constraints**")
        for c in con:
            lines.append(f"- {c}")
        lines.append("")

    dep = plan.get("dependencies") or []
    if dep:
        lines.append("**Dependencies**")
        for d in dep:
            lines.append(f"- {d}")
        lines.append("")

    ft = plan.get("files_touched") or []
    if ft:
        lines.append("**Files**")
        for f in ft:
            if isinstance(f, dict):
                lines.append(f"- `{f.get('path', '')}` ({f.get('role', '')})")
        lines.append("")

    rk = plan.get("risks") or []
    if rk:
        lines.append("**Risks**")
        for r in rk:
            lines.append(f"- {r}")
        lines.append("")

    tp = plan.get("test_plan", "")
    if tp:
        lines.append(f"**Test plan:** {tp}")
        lines.append("")

    if plan.get("needs_research"):
        lines.append("_Planner flagged: may need extra research._")
        lines.append("")

    lines.append("**Steps**")
    for st in plan.get("steps") or []:
        tid = st.get("id", "")
        task = st.get("task", "")
        acc = st.get("acceptance", "")
        lines.append(f"{tid}. **{task}**  \n   _Done when:_ {acc}")

    oq = plan.get("open_questions") or []
    if oq:
        lines.append("")
        lines.append("**Needs your input**")
        for q in oq:
            rec = q.get("recommended_default", "")
            extra = f" _(suggested: {rec})_" if rec else ""
            lines.append(f"- **{q.get('id', '?')}** {q.get('question', '')}{extra}")
    return "\n".join(lines)


def format_plan_for_system_prompt(
    plan: Dict[str, Any],
    user_answers: Optional[Dict[str, str]] = None,
) -> str:
    """Block injected into the writer system prompt."""
    user_answers = user_answers or {}
    parts = [
        "### Approved development plan (follow in order)",
        f"**Goal:** {plan.get('goal', '')}",
    ]
    dod = (plan.get("definition_of_done") or "").strip()
    if dod:
        parts.append(f"**Definition of done:** {dod}")
    parts.append("")

    ng = plan.get("non_goals") or []
    if ng:
        parts.append("**Non-goals:** " + "; ".join(str(x) for x in ng))

    if user_answers:
        parts.append("**User clarified:**")
        for qid, ans in user_answers.items():
            parts.append(f"- {qid}: {ans}")
        parts.append("")

    ass = plan.get("assumptions") or []
    if ass:
        parts.append("**Assumptions:** " + "; ".join(str(a) for a in ass))
    con = plan.get("constraints") or []
    if con:
        parts.append("**Constraints:** " + "; ".join(str(c) for c in con))
    dep = plan.get("dependencies") or []
    if dep:
        parts.append("**Dependencies:** " + "; ".join(str(d) for d in dep))

    ft = plan.get("files_touched") or []
    if ft:
        parts.append("**Files:** " + "; ".join(
            f"{x.get('path', '')} ({x.get('role', '')})" for x in ft if isinstance(x, dict)
        ))

    rk = plan.get("risks") or []
    if rk:
        parts.append("**Risks:** " + "; ".join(str(r) for r in rk))

    tp = plan.get("test_plan", "")
    if tp:
        parts.append(f"**Test plan:** {tp}")

    parts.append("")
    parts.append("**Steps (do not skip):**")
    for st in plan.get("steps") or []:
        parts.append(
            f"{st.get('id')}. {st.get('task', '')} — Acceptance: {st.get('acceptance', '')}"
        )
    parts.append("")
    parts.append(
        "Implement according to this plan. If a step is impossible, state which step "
        "and why, then suggest the smallest change to the plan."
    )
    return "\n".join(parts)


def _lint_steps(steps_raw: List[Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Preserve step order; ensure non-empty task/acceptance; sequential ids 1..n; collect warnings."""
    warnings: List[str] = []
    parsed: List[Dict[str, Any]] = []
    for i, st in enumerate(steps_raw):
        if not isinstance(st, dict):
            warnings.append(f"Step entry {i + 1} ignored (not an object).")
            continue
        task = str(st.get("task", "")).strip()
        acc = str(st.get("acceptance", "")).strip()
        sid = st.get("id", i + 1)
        try:
            orig_id = int(sid)
        except (TypeError, ValueError):
            orig_id = i + 1
            warnings.append(f"Step {i + 1}: non-numeric id; order preserved for renumbering.")
        if not task:
            task = "(unnamed step)"
            warnings.append(f"Step (order {i + 1}): empty task filled with placeholder.")
        if not acc:
            acc = "Define an observable check (e.g. test passes, behavior verified)."
            warnings.append(f"Step (order {i + 1}): empty acceptance filled with placeholder.")
        parsed.append({"_orig_id": orig_id, "task": task, "acceptance": acc})

    n = len(parsed)
    want = list(range(1, n + 1))
    orig_ids = [x["_orig_id"] for x in parsed]
    if orig_ids != want or len(set(orig_ids)) != n:
        warnings.append("Step ids normalized to sequential 1..n (array order preserved).")
    for j, st in enumerate(parsed, start=1):
        st["id"] = j
        del st["_orig_id"]

    return parsed, warnings


def validate_plan_shape(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure lists exist; coerce step ids; definition_of_done; step lint."""
    out = dict(plan)
    out.setdefault("assumptions", [])
    out.setdefault("constraints", [])
    out.setdefault("non_goals", [])
    out.setdefault("dependencies", [])
    out.setdefault("files_touched", [])
    out.setdefault("risks", [])
    out.setdefault("test_plan", "")
    out.setdefault("needs_research", False)
    out.setdefault("research_queries", [])
    out.setdefault("steps", [])
    out.setdefault("open_questions", [])

    dod = str(out.get("definition_of_done", "") or "").strip()
    out["definition_of_done"] = dod
    val_warnings: List[str] = []
    if not dod:
        val_warnings.append(
            "definition_of_done was empty — add observable completion criteria in the manual override or regenerate."
        )

    steps, step_warnings = _lint_steps(out["steps"])
    out["steps"] = steps
    val_warnings.extend(step_warnings)

    if val_warnings:
        out["_validation_warnings"] = val_warnings
    else:
        out.pop("_validation_warnings", None)

    oq = []
    for q in out["open_questions"]:
        if isinstance(q, dict) and q.get("question"):
            opts = list(q.get("options") or [])
            rec = str(q.get("recommended_default", "") or "")
            if rec and opts and rec not in opts:
                rec = opts[0] if opts else ""
            oq.append(
                {
                    "id": str(q.get("id", f"q{len(oq)+1}")),
                    "question": str(q["question"]),
                    "options": opts,
                    "recommended_default": rec,
                }
            )
    out["open_questions"] = oq[:3]

    ft = []
    for f in out.get("files_touched") or []:
        if isinstance(f, dict) and f.get("path"):
            ft.append(
                {"path": str(f["path"]), "role": str(f.get("role", "modify"))}
            )
    out["files_touched"] = ft[:12]

    out["needs_research"] = bool(out.get("needs_research"))
    rq = out.get("research_queries") or []
    out["research_queries"] = [str(x) for x in rq if str(x).strip()][:5]

    return out


def radio_index_for_question(q: Dict[str, Any]) -> int:
    """Default radio index from recommended_default."""
    opts = q.get("options") or []
    rec = q.get("recommended_default") or ""
    if rec and rec in opts:
        return opts.index(rec)
    return 0
