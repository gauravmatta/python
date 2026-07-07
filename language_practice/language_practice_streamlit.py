# Language practice — hear a sentence (TTS), then speak it (speech-to-text).
#
# Install:
#   pip install streamlit pyttsx3 SpeechRecognition sounddevice numpy requests edge-tts deep-translator
#   pip install openai-whisper torch   # local STT (Whisper base); CPU: fp16 disabled in code
#   pip install "spacy>=3.8" && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl
#   (Python 3.14 needs spaCy 3.8+ wheels; spaCy 3.7.x builds from source and fails on blis/Cython/NumPy.)
#   pip install pypinyin   # Chinese practice line: ruby pinyin above hanzi
#
# Run:
#   streamlit run language_practice_streamlit.py
#
# Notes:
#   - "Listen" uses Microsoft Edge neural TTS (edge-tts, needs internet); playback is hidden (no audio bar).
#     MP3 is cached in session per (sentence + voice + speed) so repeat clicks skip synthesis.
#     Fallback: pyttsx3 (robotic) if edge-tts fails. Speed slider maps to Edge rate; voice in sidebar.
#   - "Speak" / "Stop" records from the PC microphone (sounddevice). When target is English, sidebar chooses
#     Whisper (local) or Google via SpeechRecognition; Whisper can fall back to Google. Non-English: Google STT only.
#   - LLM outputs English only. UI shows target language + optional mother-tongue gloss (deep-translator).
#     Chinese target: pinyin above characters (pip install pypinyin).
#   - Similarity: spaCy (en_core_web_md) compares the practice line to the transcript in the target language.
#     Punctuation is stripped before comparison (STT often omits periods, etc.). Non-English scores are approximate.
#   - Next sentences: local Ollama (default model qwen3.5:9b); run ollama serve and pull the model.
#     Qwen3.5 is a reasoning model: API requests must send think=false (top-level) or it spends
#     most time on hidden reasoning — like CLI `ollama run ... --think=false`.
#   - Next-sentence prompts include the last RECENT_SENTENCES_MAX practice lines to reduce repeated topics.
#   - Settings + last practice sentence + sentences_studied_count + English STT choice (Whisper vs Google)
#     are saved per (target, mother) pair in language_practice_user_state.json.

from __future__ import annotations

import asyncio
import base64
import html
import io
import json
import os
import re
import tempfile
import unicodedata
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import speech_recognition as sr
import streamlit as st
import streamlit.components.v1 as components

# (Google Translate / deep-translator code, label) — practice & gloss languages.
LANG_OPTIONS: List[Tuple[str, str]] = [
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("tr", "Turkish"),
    ("ru", "Russian"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("uk", "Ukrainian"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("zh-CN", "Chinese (Simplified)"),
]

SAMPLE_RATE = 16_000
CHANNELS = 1

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 10
# Last N practice lines sent to Ollama so it avoids repeating topics.
RECENT_SENTENCES_MAX = 5
# Max distinct Edge-TTS clips kept in session (sentence + voice + speed key).
LISTEN_TTS_CACHE_MAX = 32

# Per-level specs for Ollama prompts: word bands, allowed/forbidden structures (one sentence only).
LEVEL_SPECS: Dict[int, str] = {
    1: (
        "Level 1 — Survival chunk. "
        "Word count: 2–5 words (strict). "
        "FORBIDDEN: subordinate clauses, relative clauses, semicolons, second sentence. "
        "ALLOWED: fixed phrases, greeting/thanks/apology/very short request (e.g. “Water, please.”). "
        "Vocabulary: highest-frequency only."
    ),
    2: (
        "Level 2 — Minimal line. "
        "Word count: 5–8 words. "
        "FORBIDDEN: subordination (because/when/if/although), relative clauses (who/which/that). "
        "ALLOWED: one simple clause; “and” or “please/thanks” at most once. "
        "Grammar: present or imperative; avoid perfect unless trivial (“I’m here”)."
    ),
    3: (
        "Level 3 — Routine micro-sentence. "
        "Word count: 8–12 words. "
        "FORBIDDEN: relative clauses; nested subordination. "
        "ALLOWED: one main clause plus at most ONE light add-on: a short prep phrase (time/place) OR a very short “because” (reason ≤5 words)."
    ),
    4: (
        "Level 4 — Clear everyday. "
        "Word count: 12–16 words. "
        "FORBIDDEN: relative clauses; more than one subordinator. "
        "ALLOWED: coordination with and/but/so OR one short because-clause. "
        "Vocabulary: common collocations; no opaque idioms."
    ),
    5: (
        "Level 5 — Connected idea. "
        "Word count: 16–20 words. "
        "FORBIDDEN: nested subordination (subclause inside subclause). "
        "ALLOWED: exactly ONE subordinate link among when/if/because, OR one very short relative clause if essential. "
        "Vocabulary: broader everyday; at most one less common word if context-clear."
    ),
    6: (
        "Level 6 — Natural dialog-style. "
        "Word count: 20–25 words. "
        "ALLOWED: two linked clauses with contrast (although/but), condition (if), reason (because/since), or time (while/after) — use at most two such devices total. "
        "Light transparent phrases only (“on the way”). "
        "FORBIDDEN: stacking many clauses with only “and”."
    ),
    7: (
        "Level 7 — Dense but controlled. "
        "Word count: 25–30 words. "
        "ALLOWED: one embedded clause (short which/who OR one subordinate link), not both heavy. "
        "Vocabulary: precise verbs/adjectives; abstract nouns allowed sparingly with clear context. "
        "FORBIDDEN: more than one independent clause unless properly subordinated."
    ),
    8: (
        "Level 8 — Advanced single sentence. "
        "Word count: 30–34 words. "
        "ALLOWED: layered syntax — pick at most TWO of: embedded clause, contrast/concession, purpose/result. "
        "Vocabulary: lower-frequency items OK if natural spoken English. "
        "FORBIDDEN: list-like enumeration; complexity must be grammatical, not comma-piling."
    ),
    9: (
        "Level 9 — Near-native compression. "
        "Word count: 34–40 words (aim ~38). "
        "ALLOWED: heavy embedding with clear scope (who did what); relative + subordination if still speakable. "
        "Vocabulary: nuanced; light metaphor at most one image. "
        "Hedging allowed (“might”, “tends to”)."
    ),
    10: (
        "Level 10 — Maximum band. "
        "Word count: 38–45 words (aim ~40–42; do not exceed 45 words). "
        "ALLOWED: highly complex single sentence — mix subordination, relative clause(s), concession, and/or non-finite structure only if still speakable aloud. "
        "Vocabulary: advanced/precise; formal connectives (nevertheless, whereas, insofar as) sparingly. "
        "Must integrate multiple propositions (cause + contrast + implication) without becoming a paragraph."
    ),
}


def level_spec_text(level: int) -> str:
    lv = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(level)))
    return LEVEL_SPECS.get(lv, LEVEL_SPECS[MAX_DIFFICULTY])


def num_predict_for_level(level: int) -> int:
    """Enough output tokens for long band-10 sentences (~45 words) without huge latency on low bands."""
    lv = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(level)))
    return min(320, 72 + lv * 22)


def get_recent_sentences(st_session: Any) -> List[str]:
    """Last RECENT_SENTENCES_MAX practice lines, oldest first."""
    raw = getattr(st_session, "recent_sentences", None)
    out: List[str] = []
    if isinstance(raw, list):
        for s in raw:
            t = (s or "").strip()
            if t:
                out.append(t)
    return out[-RECENT_SENTENCES_MAX:]


def push_recent_sentence(st_session: Any, new_sentence: str) -> None:
    """Append new line and keep only the last RECENT_SENTENCES_MAX entries."""
    t = (new_sentence or "").strip()
    if not t:
        return
    cur = get_recent_sentences(st_session)
    if cur and cur[-1] == t:
        return
    st_session.recent_sentences = (cur + [t])[-RECENT_SENTENCES_MAX:]


# pyttsx3 speech rate (words/min); ~200 is often default — lower = slower, easier to follow.
TTS_DEFAULT_WPM = 130
EDGE_TTS_DEFAULT_VOICE = "en-US-AriaNeural"
WHISPER_MODEL_NAME = "base"

# --- Persisted JSON (per target/mother pair); file next to this script ---
LP_STATE_FILENAME = "language_practice_user_state.json"


def _lp_state_file_path() -> Path:
    return Path(__file__).resolve().parent / LP_STATE_FILENAME


def pair_key(target: str, mother: str) -> str:
    """Stable key for a (target, mother) couple."""

    def norm(x: str) -> str:
        c = (x or "en").strip().lower()
        if c in ("zh-cn", "zh"):
            return "zh-cn"
        return c

    return f"{norm(target)}|{norm(mother)}"


def _lp_default_pair_state() -> Dict[str, Any]:
    ps = "Hello"
    return {
        "practice_sentence": ps,
        "difficulty_level": MIN_DIFFICULTY,
        "recent_sentences": [ps],
        "sentences_studied_count": 1,
        "tts_wpm": TTS_DEFAULT_WPM,
        "lp_edge_voice": None,
        "lp_ollama_base": DEFAULT_OLLAMA_URL,
        "lp_ollama_model": DEFAULT_OLLAMA_MODEL,
        "lp_en_stt": "whisper",
        "last_heard": None,
        "last_heard_en": None,
        "last_target": None,
    }


def _lp_clamp_difficulty(n: Any) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = MIN_DIFFICULTY
    return max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, v))


def _lp_clamp_wpm(n: Any) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = TTS_DEFAULT_WPM
    return max(80, min(220, v))


def _lp_clamp_study_count(n: Any) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 1
    return max(1, v)


def load_store() -> Dict[str, Any]:
    p = _lp_state_file_path()
    if not p.is_file():
        return {"pairs": {}}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("pairs"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"pairs": {}}


def _lp_save_store(store: Dict[str, Any]) -> None:
    p = _lp_state_file_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    text = json.dumps(store, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(p)


def _lp_merge_pair_data(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = _lp_default_pair_state()
    if not isinstance(raw, dict):
        return d
    known = set(d.keys())
    for k, v in raw.items():
        if k in known:
            d[k] = v
    ps = (d.get("practice_sentence") or "").strip() or "Hello"
    d["practice_sentence"] = ps
    d["difficulty_level"] = _lp_clamp_difficulty(d.get("difficulty_level"))
    rs = d.get("recent_sentences")
    if not isinstance(rs, list):
        rs = []
    cleaned: List[str] = []
    for s in rs:
        t = (s or "").strip()
        if t:
            cleaned.append(t)
    if not cleaned:
        cleaned = [ps]
    d["recent_sentences"] = cleaned[-RECENT_SENTENCES_MAX:]
    d["sentences_studied_count"] = _lp_clamp_study_count(d.get("sentences_studied_count"))
    d["tts_wpm"] = _lp_clamp_wpm(d.get("tts_wpm"))
    d["lp_ollama_base"] = (str(d.get("lp_ollama_base") or DEFAULT_OLLAMA_URL)).strip() or DEFAULT_OLLAMA_URL
    d["lp_ollama_model"] = (str(d.get("lp_ollama_model") or DEFAULT_OLLAMA_MODEL)).strip() or DEFAULT_OLLAMA_MODEL
    ev = d.get("lp_edge_voice")
    d["lp_edge_voice"] = ev if isinstance(ev, str) and ev.strip() else None
    _estt = d.get("lp_en_stt")
    d["lp_en_stt"] = _estt if _estt in ("whisper", "google") else "whisper"
    for k in ("last_heard", "last_heard_en", "last_target"):
        v = d.get(k)
        d[k] = v if isinstance(v, str) and v.strip() else None
    return d


def apply_pair_to_session(st: Any, raw: Optional[Dict[str, Any]]) -> None:
    """Write persisted fields into Streamlit session_state (not target/mother codes)."""
    d = _lp_merge_pair_data(raw)
    st["practice_sentence"] = d["practice_sentence"]
    st["difficulty_level"] = d["difficulty_level"]
    st["recent_sentences"] = d["recent_sentences"]
    st["tts_wpm"] = d["tts_wpm"]
    if d["lp_edge_voice"]:
        st["lp_edge_voice"] = d["lp_edge_voice"]
    else:
        st.pop("lp_edge_voice", None)
    st["lp_ollama_base"] = d["lp_ollama_base"]
    st["lp_ollama_model"] = d["lp_ollama_model"]
    st["sentences_studied_count"] = d["sentences_studied_count"]
    st["lp_en_stt"] = d["lp_en_stt"]
    if d["last_heard"]:
        st["last_heard"] = d["last_heard"]
    else:
        st.pop("last_heard", None)
    if d["last_heard_en"]:
        st["last_heard_en"] = d["last_heard_en"]
    else:
        st.pop("last_heard_en", None)
    if d["last_target"]:
        st["last_target"] = d["last_target"]
    else:
        st.pop("last_target", None)


def _lp_snapshot_from_session(st: Any) -> Dict[str, Any]:
    ps = ((getattr(st, "practice_sentence", None) or "") or "").strip() or "Hello"
    rs = getattr(st, "recent_sentences", None)
    if not isinstance(rs, list):
        rs = [ps]
    cleaned: List[str] = []
    for s in rs:
        t = (s or "").strip()
        if t:
            cleaned.append(t)
    if not cleaned:
        cleaned = [ps]
    out = {
        "practice_sentence": ps,
        "difficulty_level": _lp_clamp_difficulty(getattr(st, "difficulty_level", None)),
        "recent_sentences": cleaned[-RECENT_SENTENCES_MAX:],
        "sentences_studied_count": _lp_clamp_study_count(getattr(st, "sentences_studied_count", None)),
        "tts_wpm": _lp_clamp_wpm(getattr(st, "tts_wpm", None)),
        "lp_edge_voice": getattr(st, "lp_edge_voice", None),
        "lp_ollama_base": (str(getattr(st, "lp_ollama_base", None) or DEFAULT_OLLAMA_URL)).strip()
        or DEFAULT_OLLAMA_URL,
        "lp_ollama_model": (str(getattr(st, "lp_ollama_model", None) or DEFAULT_OLLAMA_MODEL)).strip()
        or DEFAULT_OLLAMA_MODEL,
        "lp_en_stt": (
            "google"
            if str(getattr(st, "lp_en_stt", None) or "whisper").strip().lower() == "google"
            else "whisper"
        ),
        "last_heard": getattr(st, "last_heard", None),
        "last_heard_en": getattr(st, "last_heard_en", None),
        "last_target": getattr(st, "last_target", None),
    }
    for k in ("last_heard", "last_heard_en", "last_target"):
        v = out[k]
        out[k] = v if isinstance(v, str) and v.strip() else None
    ev = out["lp_edge_voice"]
    out["lp_edge_voice"] = ev if isinstance(ev, str) and ev.strip() else None
    return out


def persist_current_pair(store: Dict[str, Any], st: Any, pk: str) -> None:
    store.setdefault("pairs", {})[pk] = _lp_snapshot_from_session(st)
    _lp_save_store(store)


# --- Chinese practice line: ruby pinyin (requires pypinyin) ---
def target_lang_is_chinese(code: str) -> bool:
    """True when sidebar target is Chinese (Simplified UI code zh-CN)."""
    c = (code or "").strip().lower()
    return c in ("zh-cn", "zh") or c.startswith("zh")


def _is_cjk_ideograph_for_pinyin(ch: str) -> bool:
    """Han characters that should get ruby pinyin (not punctuation or Latin)."""
    if len(ch) != 1:
        return False
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF:
        return True
    if 0x3400 <= o <= 0x4DBF:
        return True
    if 0x20000 <= o <= 0x323AF:
        return True
    return False


def _chinese_line_html_with_ruby_pinyin(text: str) -> str:
    """HTML <ruby> per Han run; phrase-level pinyin via pypinyin."""
    from pypinyin import Style, pinyin

    def _rubify_han_run(run: str) -> str:
        if not run:
            return ""
        py_result = pinyin(run, style=Style.TONE, heteronym=False)
        if len(py_result) != len(run):
            chunks: List[str] = []
            for ch in run:
                pr = pinyin(ch, style=Style.TONE, heteronym=False)
                py = pr[0][0] if pr and pr[0] else ch
                chunks.append(
                    f"<ruby>{html.escape(ch)}<rt>{html.escape(py)}</rt></ruby>"
                )
            return "".join(chunks)
        out: List[str] = []
        for ch, cell in zip(run, py_result):
            py = cell[0] if cell else ch
            out.append(f"<ruby>{html.escape(ch)}<rt>{html.escape(py)}</rt></ruby>")
        return "".join(out)

    parts: List[str] = []
    s = text or ""
    i = 0
    n = len(s)
    while i < n:
        if _is_cjk_ideograph_for_pinyin(s[i]):
            j = i + 1
            while j < n and _is_cjk_ideograph_for_pinyin(s[j]):
                j += 1
            parts.append(_rubify_han_run(s[i:j]))
            i = j
        else:
            parts.append(html.escape(s[i]))
            i += 1
    return "".join(parts)


_LP_RUBY_STYLE = """
<style>
.lp-ruby-line ruby { ruby-align: center; }
.lp-ruby-line rt {
  font-size: 0.52em;
  font-weight: 500;
  color: #5a5a5a;
  letter-spacing: 0.02em;
}
</style>
"""


def say_this_chinese_heading_html(line_main: str) -> str:
    """Scoped style + h2 with ruby pinyin for the practice line."""
    inner = _chinese_line_html_with_ruby_pinyin(line_main)
    return f'{_LP_RUBY_STYLE}<h2 class="lp-ruby-line">{inner}</h2>'


def edge_rate_from_wpm(wpm: int) -> str:
    """Map sidebar WPM to edge-tts rate string (roughly slower = negative %)."""
    w = max(80, min(220, int(wpm)))
    # ~80 wpm -> -40%, ~150 -> ~0%, ~220 -> +35%
    pct = int(round((w - 150) / 70.0 * 35))
    pct = max(-50, min(50, pct))
    return f"{pct:+d}%"


async def _edge_tts_mp3_bytes(text: str, voice: str, rate: str) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(
        text.strip(),
        voice,
        rate=rate,
        volume="+0%",
        pitch="+0Hz",
    )
    chunks: List[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def edge_tts_mp3_bytes_sync(text: str, voice: str, rate: str) -> bytes:
    return asyncio.run(_edge_tts_mp3_bytes(text, voice, rate))


def play_mp3_hidden(mp3: bytes) -> None:
    """Play MP3 in the browser without showing Streamlit's audio widget (user-gesture: Listen click)."""
    if not mp3:
        return
    b64 = base64.b64encode(mp3).decode("ascii")
    uid = time.time_ns()
    components.html(
        f'<audio autoplay src="data:audio/mpeg;base64,{b64}" '
        f'style="display:none;width:0;height:0;position:absolute;opacity:0;" '
        f'id="lp_tts_{uid}"></audio>',
        height=0,
    )


def listen_tts_cache_key(sentence: str, voice: str, rate: str) -> str:
    """Stable key: same text + voice + speed → reuse MP3."""
    return f"{(sentence or '').strip()}\x1f{voice}\x1f{rate}"


def cache_get_listen_mp3(st_session: Any, key: str) -> Optional[bytes]:
    c = st_session.get("listen_mp3_cache")
    if not isinstance(c, dict):
        return None
    v = c.get(key)
    return v if isinstance(v, (bytes, bytearray)) else None


def cache_put_listen_mp3(st_session: Any, key: str, mp3: bytes) -> None:
    c = dict(st_session.get("listen_mp3_cache") or {})
    if key not in c and len(c) >= LISTEN_TTS_CACHE_MAX:
        c.pop(next(iter(c)))
    c[key] = bytes(mp3)
    st_session.listen_mp3_cache = c


def target_lang_to_speech_locale(iso: str) -> str:
    """Map target language code to a locale for Whisper / Google STT."""
    iso = (iso or "en").strip().lower()
    m = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-PT",
        "tr": "tr-TR",
        "ru": "ru-RU",
        "nl": "nl-NL",
        "pl": "pl-PL",
        "uk": "uk-UA",
        "ar": "ar-SA",
        "hi": "hi-IN",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "zh-cn": "zh-CN",
    }
    if iso in m:
        return m[iso]
    if iso.startswith("zh"):
        return "zh-CN"
    return f"{iso}-{iso.upper()}" if len(iso) == 2 else "en-US"


def _google_lang_code(code: str) -> str:
    """Normalize codes for GoogleTranslator (e.g. zh-cn → zh-CN)."""
    c = (code or "en").strip()
    cl = c.lower()
    if cl in ("zh-cn", "zh"):
        return "zh-CN"
    if len(cl) == 2:
        return cl
    return c


def translate_en_to(text_en: str, dest_lang: str) -> str:
    from deep_translator import GoogleTranslator

    if not (text_en or "").strip():
        return ""
    d = _google_lang_code(dest_lang)
    if d.lower() == "en":
        return text_en.strip()
    return (GoogleTranslator(source="en", target=d).translate(text_en) or "").strip()


def _lang_eq(a: str, b: str) -> bool:
    x = (a or "").strip().lower()
    y = (b or "").strip().lower()
    if x == y:
        return True
    if x in ("zh-cn", "zh") and y in ("zh-cn", "zh"):
        return True
    return False


def practice_display_lines(
    english_canonical: str,
    target_lang: str,
    mother_lang: str,
) -> Tuple[str, Optional[str]]:
    """
    Main line = English translated to target (or English if target is en).
    Gloss = English translated to mother tongue; hidden if mother == target.
    """
    en = (english_canonical or "").strip()
    if not en:
        return "", None
    tl = (target_lang or "en").strip()
    ml = (mother_lang or "en").strip()
    if _google_lang_code(tl).lower() == "en":
        main = en
    else:
        main = translate_en_to(en, tl)
    if _lang_eq(tl, ml):
        return main, None
    if _google_lang_code(ml).lower() == "en" and _google_lang_code(tl).lower() != "en":
        gloss = en
    else:
        gloss = translate_en_to(en, ml)
    if gloss and main.strip().lower() == (gloss or "").strip().lower():
        return main, None
    return main, gloss if gloss else None


@st.cache_data(ttl=86400)
def edge_voice_options_for_target_lang(lang_code: str) -> List[Tuple[str, str]]:
    """Neural Edge voices whose ShortName matches the target language."""
    try:
        import edge_tts
    except ImportError:
        return [(EDGE_TTS_DEFAULT_VOICE, "Default (install edge-tts)")]

    async def _list() -> List[Dict[str, Any]]:
        return await edge_tts.list_voices()

    try:
        voices = asyncio.run(_list())
    except Exception:
        return [(EDGE_TTS_DEFAULT_VOICE, "Default")]

    lc = (lang_code or "en").strip().lower()
    opts: List[Tuple[str, str]] = []
    for v in voices:
        sn = (v.get("ShortName") or "").strip()
        if "Neural" not in sn:
            continue
        ok = False
        if lc == "en":
            ok = sn.startswith("en-")
        elif lc.startswith("zh"):
            ok = sn.startswith("zh-CN") or sn.startswith("zh-TW")
        elif len(lc) == 2:
            ok = sn.startswith(f"{lc}-")
        else:
            ok = sn.startswith(lc)
        if ok:
            fn = (v.get("FriendlyName") or sn).strip()
            opts.append((sn, fn))
    opts.sort(key=lambda x: x[1])
    if opts:
        return opts
    if lc == "en":
        return [(EDGE_TTS_DEFAULT_VOICE, "Aria (US) — neural")]
    return [(EDGE_TTS_DEFAULT_VOICE, "Fallback — pick English voices in sidebar if empty")]


DifficultyChoice = Literal["Easier", "Same", "Harder"]


def compute_new_level(current: int, choice: DifficultyChoice) -> int:
    if choice == "Easier":
        return max(MIN_DIFFICULTY, current - 1)
    if choice == "Harder":
        return min(MAX_DIFFICULTY, current + 1)
    return current


_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/" + "think" + ">"
_THINK_BLOCK = re.compile(
    re.escape(_THINK_OPEN) + r".*?" + re.escape(_THINK_CLOSE),
    re.DOTALL | re.IGNORECASE,
)


def clean_llm_sentence(raw: str) -> str:
    t = _THINK_BLOCK.sub("", (raw or "")).strip()
    if t.startswith("```"):
        lines = t.split("\n")
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    t = t.strip().strip("\"'“”")
    line = (t.split("\n") or [""])[0].strip()
    return line


def ollama_chat_sync(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    timeout: float = 120.0,
    options: Optional[Dict[str, Any]] = None,
    think: bool = False,
) -> str:
    """POST /api/chat. `think` must be top-level (not inside `options`) for Qwen3/Qwen3.5 — otherwise
    reasoning tokens fill the budget and responses are slow or empty. See Ollama thinking docs.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
    }
    if options:
        payload["options"] = options
    resp = requests.post(f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return (resp.json().get("message") or {}).get("content", "") or ""


def messages_for_next_sentence(
    previous: str,
    choice: DifficultyChoice,
    new_level: int,
    avoid_topics: List[str],
) -> List[Dict[str, str]]:
    new_level = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(new_level)))
    band_desc = level_spec_text(new_level)
    difficulty_hint = {
        "Easier": (
            "Relative to the previous sentence: make it EASIER — fewer words toward the band minimum, simpler "
            "vocabulary, and/or easier grammar. Do NOT merely simplify the same topic or paraphrase the same idea. "
            "Use a different sentence PATTERN than the previous line (not the same template with a swapped noun)."
        ),
        "Same": (
            "Relative to the previous sentence: match the SAME overall difficulty — comparable length (within the "
            "band’s word range), vocabulary tier, and structural complexity — not easier, not harder. "
            "“Same difficulty” does NOT mean the same grammatical scaffolding: vary the clause shape and opening."
        ),
        "Harder": (
            "Relative to the previous sentence: make it HARDER — more words toward the band maximum, richer "
            "vocabulary, and/or more complex grammar, while staying inside the target level rules below. "
            "Still choose a fresh syntactic pattern — do not stack difficulty onto an identical frame as before."
        ),
    }[choice]
    system = (
        "You create English speaking-practice lines for learners. "
        "You MUST output exactly ONE English sentence: one period (.) at the end, no second sentence, no bullet, "
        "no title, no quotes around the line, no explanation. "
        "Do not split into two sentences. Do not build complexity only by chaining many short clauses with "
        "\"and\"; use the allowed syntax for the target level. "
        "Each new line must use a FRESH topic and situation versus every sentence in the recent list below. "
        "Do not paraphrase or continue any of those scenarios. "
        "STRUCTURE / SYNTAX: The new sentence must NOT reuse the same grammatical template or scaffold as any "
        "recent line. Forbidden: repeating the same opening phrase or frame (e.g. all lines starting "
        "\"I would like to … please\", or only swapping the object inside \"Can I get …\", \"I need to …\"). "
        "Vary how the sentence begins (subject, adverb, question form, imperative, expletive \"There is/are\", "
        "subordinate-first, etc.) as allowed by the target level. "
        "Avoid reusing those sentences' content words; ordinary function words (the, a, is, …) are fine."
    )
    recent_block = ""
    if avoid_topics:
        lines = "\n".join(f"- {s!r}" for s in avoid_topics)
        recent_block = (
            f"### Recent practice sentences (up to {RECENT_SENTENCES_MAX})\n"
            "For EACH of these lines: do NOT repeat its topic, setting, OR its sentence structure (pattern, "
            "scaffold, typical opening). The next line must sound like a different *kind* of sentence, not the same "
            "mould with different nouns.\n"
            f"{lines}\n\n"
        )
    user = (
        "### Target level (follow strictly)\n"
        f"{band_desc}\n\n"
        f"{recent_block}"
        "### Context (difficulty reference only — ignore topic of “Previous sentence” for subject matter)\n"
        f"Previous sentence (for relative difficulty vs last line): {previous.strip()!r}\n"
        f"Button pressed: {choice}.\n"
        f"Target level number: {new_level} (allowed range {MIN_DIFFICULTY}–{MAX_DIFFICULTY}).\n\n"
        "### Relative adjustment\n"
        f"{difficulty_hint}\n\n"
        "### Requirements\n"
        "- Meet the word-count range and ALLOWED/FORBIDDEN rules for this level.\n"
        "- Topic: must differ from ALL sentences listed under “Recent practice sentences” above (if any).\n"
        "- Structure: must differ from ALL of those lines — different syntactic pattern (not the same clause frame "
        "with only the object or verb swapped). Change how the sentence is built, not just the words inside one "
        "fixed pattern.\n"
        "- Different concrete nouns and main verbs; no synonym-swap of the same situation.\n"
        "- Exactly one sentence, natural to speak aloud.\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def fetch_next_sentence_from_ollama(
    base_url: str,
    model: str,
    previous: str,
    choice: DifficultyChoice,
    new_level: int,
    avoid_topics: List[str],
) -> str:
    messages = messages_for_next_sentence(previous, choice, new_level, avoid_topics)
    nl = max(MIN_DIFFICULTY, min(MAX_DIFFICULTY, int(new_level)))
    raw = ollama_chat_sync(
        base_url,
        model,
        messages,
        think=False,
        options={
            # Slightly higher helps avoid the same syntactic template every turn (e.g. "I would like to … please").
            "temperature": 0.75,
            "top_p": 0.9,
            "num_predict": num_predict_for_level(nl),
        },
    )
    sentence = clean_llm_sentence(raw)
    if not sentence:
        raise ValueError("Model returned an empty sentence.")
    return sentence


def run_difficulty_button(choice: DifficultyChoice, *, base_url: str, model: str) -> None:
    prev = (st.session_state.get("practice_sentence") or "").strip()
    level = int(st.session_state.get("difficulty_level") or MIN_DIFFICULTY)
    new_level = compute_new_level(level, choice)
    avoid = get_recent_sentences(st.session_state)
    nxt = fetch_next_sentence_from_ollama(base_url, model, prev, choice, new_level, avoid_topics=avoid)
    st.session_state.practice_sentence = nxt
    st.session_state.difficulty_level = new_level
    push_recent_sentence(st.session_state, nxt)
    st.session_state.sentences_studied_count = int(st.session_state.get("sentences_studied_count") or 1) + 1
    st.session_state.pop("last_heard", None)
    st.session_state.pop("last_heard_en", None)
    st.session_state.pop("last_target", None)
    st.session_state.lp_practice_line_hidden = False
    # One script run draws "Say this" before these buttons run; rerun so the heading matches new state.
    st.rerun()


def _recording_worker(stop_event: threading.Event, chunks: list) -> None:
    def callback(indata, frames, time, status) -> None:  # noqa: ARG001
        chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=callback,
    ):
        stop_event.wait()


def numpy_audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    if audio.size == 0:
        return b""
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.astype(np.int16, copy=False).tobytes())
    return buf.getvalue()


def speak_aloud_pyttsx3(text: str, *, rate_wpm: int = TTS_DEFAULT_WPM) -> None:
    """Fallback TTS: local SAPI (robotic)."""
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", max(80, min(400, int(rate_wpm))))
        engine.say(text.strip())
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def speech_locale_to_whisper_language(locale: str) -> Optional[str]:
    """Whisper expects ISO 639-1 (e.g. en, es). Sidebar uses en-US → en."""
    if not (locale or "").strip():
        return None
    base = locale.split("-")[0].strip().lower()
    return base if len(base) == 2 else None


@st.cache_resource(show_spinner=True)
def _load_whisper_model():
    import whisper

    return whisper.load_model(WHISPER_MODEL_NAME)


def transcribe_wav_whisper(wav_bytes: bytes, language_locale: str) -> str:
    """Local Whisper STT. Requires openai-whisper + torch."""
    if not wav_bytes:
        return ""
    model = _load_whisper_model()
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(wav_bytes)
        lang = speech_locale_to_whisper_language(language_locale)
        kwargs: Dict[str, Any] = {"fp16": False}
        if lang:
            kwargs["language"] = lang
        result = model.transcribe(tmp_path, **kwargs)
        return (result.get("text") or "").strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def transcribe_wav_google(wav_bytes: bytes, language: str) -> str:
    r = sr.Recognizer()
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(wav_bytes)
        with sr.AudioFile(tmp_path) as source:
            audio = r.record(source)
        return r.recognize_google(audio, language=language)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _target_lang_code_is_english(code: str) -> bool:
    """True when practice target is English (STT engine is user-selectable in the sidebar)."""
    return (code or "en").strip().lower() == "en"


def transcribe_wav(
    wav_bytes: bytes,
    language_locale: str,
    *,
    target_lang_code: str = "en",
    english_stt: str = "whisper",
) -> str:
    """English target: Whisper and/or Google per `english_stt`. Other targets: Google STT only."""
    if _target_lang_code_is_english(target_lang_code):
        if (english_stt or "").strip().lower() == "google":
            return transcribe_wav_google(wav_bytes, language_locale)
        try:
            text = transcribe_wav_whisper(wav_bytes, language_locale)
            if text:
                return text
        except ImportError:
            pass
        except Exception:
            pass
        return transcribe_wav_google(wav_bytes, language_locale)
    return transcribe_wav_google(wav_bytes, language_locale)


SPACY_MODEL = "en_core_web_md"


def strip_punctuation_for_similarity(text: str) -> str:
    """Remove punctuation (ASCII and Unicode, e.g. 。，！) so similarity ignores annotation gaps from STT."""
    if not (text or "").strip():
        return ""
    out: List[str] = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch)
    return " ".join("".join(out).split())


@st.cache_resource(show_spinner=False)
def _load_spacy_nlp():
    import spacy

    return spacy.load(SPACY_MODEL)


def spacy_text_similarity(a: str, b: str) -> float:
    """Cosine-style similarity in [0, 1] from spaCy doc vectors (en_core_web_md).

    Compares the practice line to the spoken transcript (same language as the UI). The model is
    English-centric; non-Latin or multilingual text may yield approximate scores.
    """
    nlp = _load_spacy_nlp()
    sa = strip_punctuation_for_similarity((a or "").strip().lower())
    sb = strip_punctuation_for_similarity((b or "").strip().lower())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    da = nlp(sa)
    db = nlp(sb)
    return float(da.similarity(db))


def main() -> None:
    st.set_page_config(page_title="Language practice", layout="centered")
    st.title("Language practice")

    if "mic_listening" not in st.session_state:
        st.session_state.mic_listening = False
    if "lp_json_store" not in st.session_state:
        st.session_state.lp_json_store = load_store()
    if "lp_target_lang" not in st.session_state:
        st.session_state.lp_target_lang = "en"
    if "lp_mother_lang" not in st.session_state:
        st.session_state.lp_mother_lang = "en"

    _pk = pair_key(st.session_state.lp_target_lang, st.session_state.lp_mother_lang)
    if st.session_state.get("lp_active_pair") != _pk:
        _raw = (st.session_state.lp_json_store.get("pairs") or {}).get(_pk)
        apply_pair_to_session(st.session_state, _raw)
        st.session_state.lp_active_pair = _pk
        st.session_state.listen_mp3_cache = {}
        for _k in ("lp_disp_lines_key", "lp_disp_main", "lp_disp_gloss"):
            st.session_state.pop(_k, None)
        st.session_state.lp_practice_line_hidden = False

    if "sentences_studied_count" not in st.session_state:
        st.session_state.sentences_studied_count = 1
    if "lp_en_stt" not in st.session_state:
        st.session_state.lp_en_stt = "whisper"
    if "lp_practice_line_hidden" not in st.session_state:
        st.session_state.lp_practice_line_hidden = False

    with st.sidebar:
        st.header("Settings")
        _lang_codes = [x[0] for x in LANG_OPTIONS]
        _lang_lbl = {x[0]: x[1] for x in LANG_OPTIONS}
        st.selectbox(
            "Target language (practice line)",
            options=_lang_codes,
            format_func=lambda c: _lang_lbl.get(c, c),
            key="lp_target_lang",
            help="Sentence is shown in this language (translated from English).",
        )
        st.selectbox(
            "Mother tongue (meaning below)",
            options=_lang_codes,
            format_func=lambda c: _lang_lbl.get(c, c),
            key="lp_mother_lang",
            help="Gloss under the practice line; hidden if same as target or redundant.",
        )
        st.slider(
            "Listen speed (words/min)",
            min_value=80,
            max_value=220,
            step=5,
            key="tts_wpm",
            help="Maps to Edge TTS playback rate (slower ← → faster). Used for neural TTS; also applies to offline fallback.",
        )
        _tl = (st.session_state.get("lp_target_lang") or "en").strip()
        _vo = edge_voice_options_for_target_lang(_tl)
        _vo_ids = [r[0] for r in _vo]
        _vo_lbl = {r[0]: r[1] for r in _vo}
        if st.session_state.get("lp_edge_voice") not in _vo_ids:
            st.session_state.lp_edge_voice = _vo_ids[0]
        st.selectbox(
            "Listen voice (Edge TTS)",
            options=_vo_ids,
            format_func=lambda vid: _vo_lbl.get(vid, vid),
            key="lp_edge_voice",
            help="Voices match target language. Neural voices need internet; offline pyttsx3 is fallback.",
        )
        speech_lang = target_lang_to_speech_locale(_tl)
        _tl_lower = (_tl or "en").strip().lower()
        if _tl_lower == "en":
            st.radio(
                "English speech-to-text",
                options=["whisper", "google"],
                format_func=lambda x: (
                    f"Whisper ({WHISPER_MODEL_NAME}, local)"
                    if x == "whisper"
                    else "Google (SpeechRecognition)"
                ),
                key="lp_en_stt",
                horizontal=True,
                help="When target is English: local Whisper + torch, or cloud Google STT via SpeechRecognition.",
            )
            _en_m = str(st.session_state.get("lp_en_stt") or "whisper").strip().lower()
            if _en_m == "google":
                _stt_hint = "Google SpeechRecognition (`recognize_google`)"
            else:
                _stt_hint = f"Whisper **{WHISPER_MODEL_NAME}** (local), falls back to Google if needed"
        else:
            _stt_hint = "Google SpeechRecognition only (Whisper option applies when target is English)"
        st.caption(f"STT follows **target** (`{speech_lang}`): {_stt_hint}.")

        st.subheader("Ollama")
        ollama_base = st.text_input("Base URL", value=DEFAULT_OLLAMA_URL, key="lp_ollama_base")
        ollama_model = st.text_input("Model", value=DEFAULT_OLLAMA_MODEL, key="lp_ollama_model")
        st.caption(
            f"Difficulty band: **{int(st.session_state.difficulty_level)}** "
            f"({MIN_DIFFICULTY} = simplest … {MAX_DIFFICULTY} = hardest)"
        )

    sentence_en = (st.session_state.practice_sentence or "").strip()
    _tl_disp = str(st.session_state.get("lp_target_lang") or "en")
    _ml_disp = str(st.session_state.get("lp_mother_lang") or "en")
    # Cache translated lines so mic Speak/Stop reruns do not block on Google Translate again.
    _disp_key = f"{sentence_en}\x00{_tl_disp}\x00{_ml_disp}"
    try:
        if st.session_state.get("lp_disp_lines_key") == _disp_key:
            line_main = st.session_state["lp_disp_main"]
            line_gloss = st.session_state.get("lp_disp_gloss")
        else:
            line_main, line_gloss = practice_display_lines(sentence_en, _tl_disp, _ml_disp)
            st.session_state["lp_disp_lines_key"] = _disp_key
            st.session_state["lp_disp_main"] = line_main
            st.session_state["lp_disp_gloss"] = line_gloss
    except Exception as ex:
        line_main, line_gloss = sentence_en, None
        st.error(f"Translation failed ({ex}). Check `pip install deep-translator` and internet.")

    _study_n = max(1, int(st.session_state.get("sentences_studied_count") or 1))
    _col_say_l, _col_say_r = st.columns([3, 2])
    with _col_say_l:
        st.markdown("### Say this")
    with _col_say_r:
        st.markdown(
            f'<p style="text-align:right;margin:0.55rem 0 0 0;color:#5f6368;font-size:0.95rem;line-height:1.35;">'
            f"<b>{_study_n}</b> sentences studied<br/>"
            f'<span style="font-size:0.82rem;opacity:0.88;">(this language pair)</span></p>',
            unsafe_allow_html=True,
        )
    _line_hidden = bool(st.session_state.get("lp_practice_line_hidden"))
    if _line_hidden:
        st.caption("_Practice line is hidden — use **Listen** / **Speak** to practice from memory._")
    else:
        if target_lang_is_chinese(_tl_disp):
            try:
                st.markdown(say_this_chinese_heading_html(line_main), unsafe_allow_html=True)
            except ImportError:
                st.markdown(f"## {line_main}")
                st.caption("Install **pypinyin** for pinyin above characters: `pip install pypinyin`")
        else:
            st.markdown(f"## {line_main}")
        if line_gloss:
            st.markdown(
                f'<p style="font-size:1.22rem;line-height:1.5;font-style:italic;color:#5f6368;margin:0.35rem 0 0 0;">'
                f"{html.escape(line_gloss)}</p>",
                unsafe_allow_html=True,
            )

    col_listen, col_speak, col_hide = st.columns(3)
    with col_listen:
        if st.button("Listen", type="primary", use_container_width=True, key="btn_listen"):
            if not line_main.strip():
                st.warning("Enter or choose a sentence first.")
            else:
                try:
                    _voice = str(st.session_state.get("lp_edge_voice") or EDGE_TTS_DEFAULT_VOICE)
                    _rate = edge_rate_from_wpm(int(st.session_state.tts_wpm))
                    _ck = listen_tts_cache_key(line_main, _voice, _rate)
                    mp3 = cache_get_listen_mp3(st.session_state, _ck)
                    if mp3:
                        play_mp3_hidden(mp3)
                    else:
                        mp3 = edge_tts_mp3_bytes_sync(line_main, _voice, _rate)
                        if not mp3:
                            raise ValueError("Empty audio")
                        cache_put_listen_mp3(st.session_state, _ck, mp3)
                        play_mp3_hidden(mp3)
                except ImportError:
                    speak_aloud_pyttsx3(line_main, rate_wpm=int(st.session_state.tts_wpm))
                    st.info("Install **edge-tts** for neural voices: `pip install edge-tts`")
                except Exception as e_edge:
                    try:
                        speak_aloud_pyttsx3(line_main, rate_wpm=int(st.session_state.tts_wpm))
                        st.warning(f"Edge TTS failed ({e_edge}); played offline voice instead.")
                    except Exception as e2:
                        st.error(f"Could not play speech: {e2}")
    with col_speak:
        if st.session_state.mic_listening:
            if st.button("Stop", type="primary", key="mic_stop", use_container_width=True):
                ev: threading.Event = st.session_state.mic_stop_event
                th: threading.Thread = st.session_state.mic_thread
                chunks: list = st.session_state.mic_chunks
                ev.set()
                th.join(timeout=60)
                if th.is_alive():
                    st.error("Recording did not stop cleanly. Try refreshing the page if the mic stays busy.")
                st.session_state.mic_listening = False
                audio = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=np.int16)
                wav_bytes = numpy_audio_to_wav_bytes(audio)
                if not wav_bytes:
                    st.warning("No audio captured. Press Speak and try again.")
                else:
                    try:
                        _tlg = (st.session_state.get("lp_target_lang") or "en").strip().lower()
                        _en_stt = str(st.session_state.get("lp_en_stt") or "whisper").strip().lower()
                        text = transcribe_wav(
                            wav_bytes,
                            speech_lang,
                            target_lang_code=_tlg,
                            english_stt=_en_stt if _tlg == "en" else "google",
                        )
                        st.session_state["last_heard"] = text
                        st.session_state.pop("last_heard_en", None)
                        st.session_state["last_target"] = (st.session_state.practice_sentence or "").strip()
                    except sr.UnknownValueError:
                        st.error("Could not understand the audio. Speak more clearly or check the language.")
                        st.session_state.pop("last_heard", None)
                        st.session_state.pop("last_heard_en", None)
                        st.session_state.pop("last_target", None)
                    except sr.RequestError as e:
                        st.error(f"Speech service error (check internet): {e}")
                        st.session_state.pop("last_heard", None)
                        st.session_state.pop("last_heard_en", None)
                        st.session_state.pop("last_target", None)
                st.session_state["lp_mic_stop_rerun"] = True
        else:
            if st.button("Speak", type="primary", key="mic_speak", use_container_width=True):
                try:
                    chunks = []
                    stop_event = threading.Event()
                    worker = threading.Thread(
                        target=_recording_worker,
                        args=(stop_event, chunks),
                        daemon=True,
                    )
                    st.session_state.mic_chunks = chunks
                    st.session_state.mic_stop_event = stop_event
                    st.session_state.mic_thread = worker
                    worker.start()
                    st.session_state.mic_listening = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not open microphone: {e}")
    with col_hide:
        _hide_lbl = "Show" if st.session_state.get("lp_practice_line_hidden") else "Hide"
        if st.button(_hide_lbl, use_container_width=True, key="btn_toggle_practice_line"):
            st.session_state.lp_practice_line_hidden = not bool(
                st.session_state.get("lp_practice_line_hidden")
            )
            st.rerun()

    heard = st.session_state.get("last_heard")
    if heard is not None:
        st.markdown("**You said:**")
        st.markdown(
            f'<div style="font-size:1.28rem;line-height:1.55;padding:0.85rem 1.1rem;border-radius:0.35rem;'
            f"background:rgba(38,39,48,0.06);border-left:4px solid rgb(49,130,206);color:rgb(38,39,48);\">"
            f"{html.escape(str(heard))}</div>",
            unsafe_allow_html=True,
        )
        try:
            sim = spacy_text_similarity((line_main or "").strip(), str(heard).strip())
            st.metric("Similarity", f"{sim * 100:.1f}%")
        except Exception as e:
            st.warning(
                f"Could not compute similarity ({e}). Same setup as 47.py: "
                f"`pip install spacy` then `python -m spacy download {SPACY_MODEL}`."
            )

    st.divider()
    st.markdown("### Next sentence")
    mic_busy = st.session_state.mic_listening
    ob = (ollama_base or DEFAULT_OLLAMA_URL).strip()
    om = (ollama_model or DEFAULT_OLLAMA_MODEL).strip()
    col_e, col_s, col_h = st.columns(3)
    with col_e:
        if st.button("Easier", disabled=mic_busy, use_container_width=True, key="diff_easier"):
            try:
                with st.spinner("Asking Ollama…"):
                    run_difficulty_button("Easier", base_url=ob, model=om)
            except requests.RequestException as e:
                st.error(f"Could not reach Ollama ({e}). Start `ollama serve` and pull `{om}`.")
            except Exception as e:
                st.error(f"Could not get next sentence: {e}")
    with col_s:
        if st.button("Same", disabled=mic_busy, use_container_width=True, key="diff_same"):
            try:
                with st.spinner("Asking Ollama…"):
                    run_difficulty_button("Same", base_url=ob, model=om)
            except requests.RequestException as e:
                st.error(f"Could not reach Ollama ({e}). Start `ollama serve` and pull `{om}`.")
            except Exception as e:
                st.error(f"Could not get next sentence: {e}")
    with col_h:
        if st.button("Harder", disabled=mic_busy, use_container_width=True, key="diff_harder"):
            try:
                with st.spinner("Asking Ollama…"):
                    run_difficulty_button("Harder", base_url=ob, model=om)
            except requests.RequestException as e:
                st.error(f"Could not reach Ollama ({e}). Start `ollama serve` and pull `{om}`.")
            except Exception as e:
                st.error(f"Could not get next sentence: {e}")

    _pk_save = pair_key(st.session_state.lp_target_lang, st.session_state.lp_mother_lang)
    persist_current_pair(st.session_state.lp_json_store, st.session_state, _pk_save)

    if st.session_state.pop("lp_mic_stop_rerun", False):
        st.rerun()


if __name__ == "__main__":
    main()
