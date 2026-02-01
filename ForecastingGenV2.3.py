# QuestionGenV2.py
# Streamlit all-in-one (incremental + formatter retries on validation errors):
# Topics CSV -> protos -> select k -> (optional canonicalize selected protos)
# -> build full -> verify -> (NEW: ensure resolution URL present) -> format single
# -> append to draft CSV incrementally
# -> optional rebalance -> (optional canonicalize post-rebalance) -> optional final clean -> export CSV
#
# Key additions in this version:
# 1) Multi-level web enrichment where Call 2 builds on Call 1, and Call 3 builds on Call 2.
# 2) URL accessibility filtering (drops 404/410/401/403/5xx; normalizes redirects) to avoid dead/unreachable pages.
# 3) Pre-format check that resolution_criteria contains at least one URL other than Metaculus credible sources policy;
#    if missing, inject a best accessible URL from web results into resolution_criteria AND description (minimal edit).
# 4) Stronger date-discipline instructions to reduce the “today/tomorrow” window bug (e.g., “between Jan 22 and Jan 23”).
# 5) Rebalance target updated to 60% binary / 20% numeric / 20% MCQ (per request).
#
# Install:
#   pip install streamlit requests pydantic
# Run:
#   streamlit run QuestionGenV2.py

from __future__ import annotations

import ast
import csv
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
from zoneinfo import ZoneInfo

import requests
import streamlit as st

# --- Pydantic v2/v1 compatibility ---
try:
    from pydantic import BaseModel, Field, ValidationError
    from pydantic import field_validator  # pydantic v2
    PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, ValidationError, validator  # type: ignore
    PYDANTIC_V2 = False

    def field_validator(*fields, **kwargs):  # type: ignore
        def deco(fn):
            return validator(*fields, **kwargs, allow_reuse=True)(fn)
        return deco


PARIS_TZ = ZoneInfo("Europe/Paris")

# Metaculus “credible sources” policy (often sufficient vs enumerating outlets)
METACULUS_CREDIBLE_SOURCES_URL = "https://www.metaculus.com/faq/#definitions"

CSV_COLUMNS = [
    "title",
    "type",
    "resolution_criteria",
    "fine_print",
    "description",
    "question_weight",
    "open_time",
    "scheduled_close_time",
    "scheduled_resolve_time",
    "range_min",
    "range_max",
    "zero_point",
    "open_lower_bound",
    "open_upper_bound",
    "unit",
    "group_variable",
    "options",
    "categories",
    "ai_rating",
    "ai_rationale",
]

ALLOWED_TYPES = {"binary", "numeric", "multiple_choice"}
ALLOWED_RATINGS = {"hard_reject", "soft_reject", "accept_for_aib", "accept_for_main_site"}
ALLOWED_VERIFY_STATUS = {"pass", "fix"}


# ---------------------------
# Schemas
# ---------------------------
class ProtoQuestion(BaseModel):
    title: str
    suggested_type: str = Field(description="binary | numeric | multiple_choice")
    description: str
    why_informative: str
    candidate_sources: List[str] = Field(default_factory=list)

    @field_validator("suggested_type")
    def _type_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_TYPES:
            raise ValueError(f"suggested_type must be one of {sorted(ALLOWED_TYPES)}")
        return v2


class FullQuestion(BaseModel):
    title: str
    type: str
    resolution_criteria: str
    fine_print: str
    description: str
    question_weight: float

    open_time: str
    scheduled_close_time: str
    scheduled_resolve_time: str

    range_min: Optional[float] = None
    range_max: Optional[float] = None
    zero_point: Optional[float] = None
    open_lower_bound: Optional[float] = None
    open_upper_bound: Optional[float] = None
    unit: Optional[str] = None
    group_variable: Optional[str] = None
    options: Optional[List[str]] = None
    categories: List[str]

    ai_rating: str
    ai_rationale: str

    @field_validator("type")
    def _type_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_TYPES:
            raise ValueError(f"type must be one of {sorted(ALLOWED_TYPES)}")
        return v2

    @field_validator("ai_rating")
    def _rating_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_RATINGS:
            raise ValueError(f"ai_rating must be one of {sorted(ALLOWED_RATINGS)}")
        return v2

    @field_validator("categories")
    def _categories_ok(cls, v: List[str]):  # type: ignore
        vv = [str(c).strip() for c in (v or []) if str(c).strip()]
        if not vv:
            raise ValueError("categories must be a non-empty list")
        return vv

    @field_validator("options")
    def _options_ok(cls, v: Optional[List[str]]):  # type: ignore
        if v is None:
            return None
        vv = [str(o).strip() for o in v if str(o).strip()]
        return vv or None


class VerifyResponse(BaseModel):
    status: str = Field(description="pass | fix")
    issues: List[str] = Field(default_factory=list)
    question: FullQuestion

    @field_validator("status")
    def _status_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_VERIFY_STATUS:
            raise ValueError("status must be 'pass' or 'fix'")
        return v2


# Web search schema (produced by a cheap :online model)
class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResponse(BaseModel):
    query: str
    results: List[WebSearchResult] = Field(default_factory=list)


# ---------------------------
# Utils
# ---------------------------
def iso_dt(d: date, t: dtime = dtime(0, 0)) -> str:
    dt = datetime.combine(d, t).replace(tzinfo=PARIS_TZ)
    return dt.isoformat()


def dt_range_defaults(start_d: date, end_d: date) -> Tuple[str, str, str]:
    open_time = iso_dt(start_d, dtime(0, 0))
    close_time = iso_dt(end_d, dtime(23, 59))
    resolve_time = iso_dt(end_d, dtime(23, 59))
    return open_time, close_time, resolve_time


def date_window_phrase_inclusive(start_d: date, end_d: date) -> str:
    """
    Produces an unambiguous inclusive window phrase without 'between'/'by'.
    Convention: [start_d, end_d] inclusive is described as:
      "after (start_d - 1 day) and on or before end_d"
    """
    s0 = (start_d - timedelta(days=1)).isoformat()
    e = end_d.isoformat()
    return f"after {s0} and on or before {e}"


def read_topics_csv(uploaded_file) -> Tuple[List[str], List[Dict[str, Any]]]:
    data = uploaded_file.getvalue()
    text = data.decode("utf-8", errors="replace")
    sio = io.StringIO(text)
    reader = csv.DictReader(sio)
    if not reader.fieldnames:
        raise ValueError("CSV appears to have no header row.")
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has a header but no data rows.")
    return reader.fieldnames, rows


def pick_topic_column(fieldnames: List[str], rows: List[Dict[str, Any]]) -> str:
    sample = rows[:50]
    best_col, best_score = fieldnames[0], -1
    for col in fieldnames:
        score = sum(1 for r in sample if str(r.get(col, "")).strip())
        if score > best_score:
            best_col, best_score = col, score
    return best_col


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_first_json_block(s: str) -> Optional[str]:
    s = s.strip()
    start = None
    opener = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None or opener is None:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : j + 1]
    return None


def try_parse_json(s: str) -> Any:
    s0 = strip_code_fences(s)

    try:
        return json.loads(s0)
    except Exception:
        pass

    block = extract_first_json_block(s0)
    if block:
        try:
            return json.loads(block)
        except Exception:
            try:
                obj = ast.literal_eval(block)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    try:
        obj = ast.literal_eval(s0)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    raise ValueError("Could not parse valid JSON from the model output.")


def to_model_dict(obj: Any) -> Dict[str, Any]:
    if PYDANTIC_V2 and hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def normalize_full_question_fields(q: FullQuestion) -> FullQuestion:
    d = to_model_dict(q)

    if d.get("type") != "multiple_choice":
        d["options"] = None

    if d.get("type") != "numeric":
        d["range_min"] = None
        d["range_max"] = None
        d["zero_point"] = None
        d["open_lower_bound"] = None
        d["open_upper_bound"] = None
        d["unit"] = None

    if d.get("question_weight") is None:
        d["question_weight"] = 1.0

    return FullQuestion(**d)


def full_question_to_row(q: FullQuestion) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}
    row["title"] = q.title
    row["type"] = q.type
    row["resolution_criteria"] = q.resolution_criteria
    row["fine_print"] = q.fine_print
    row["description"] = q.description
    row["question_weight"] = q.question_weight
    row["open_time"] = q.open_time
    row["scheduled_close_time"] = q.scheduled_close_time
    row["scheduled_resolve_time"] = q.scheduled_resolve_time

    row["range_min"] = "" if q.range_min is None else q.range_min
    row["range_max"] = "" if q.range_max is None else q.range_max
    row["zero_point"] = "" if q.zero_point is None else q.zero_point
    row["open_lower_bound"] = "" if q.open_lower_bound is None else q.open_lower_bound
    row["open_upper_bound"] = "" if q.open_upper_bound is None else q.open_upper_bound
    row["unit"] = "" if q.unit is None else q.unit
    row["group_variable"] = "" if q.group_variable is None else q.group_variable

    row["options"] = "" if not q.options else json.dumps(q.options, ensure_ascii=False)
    row["categories"] = json.dumps(q.categories, ensure_ascii=False)

    row["ai_rating"] = q.ai_rating
    row["ai_rationale"] = q.ai_rationale
    return row


def write_csv(rows: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return output.getvalue()


def ensure_online_variant(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return m
    return m if m.endswith(":online") else f"{m}:online"


def inject_context(messages: List[Dict[str, str]], context: str) -> List[Dict[str, str]]:
    """
    Do not modify prompt strings. Instead add a separate user message containing context.
    Insert after the system message (if present) and before the main user prompt.
    """
    ctx = (context or "").strip()
    if not ctx:
        return messages

    if messages and messages[0].get("role") == "system":
        if len(messages) >= 2 and messages[1].get("role") == "user":
            return [messages[0], {"role": "user", "content": ctx}, messages[1]] + messages[2:]
        return [messages[0], {"role": "user", "content": ctx}] + messages[1:]
    return [{"role": "user", "content": ctx}] + messages


def format_web_context(topic: str, calls: List[WebSearchResponse]) -> str:
    lines: List[str] = []
    lines.append("WEB SEARCH CONTEXT (use ONLY to pick candidate sources/resolution links; do not copy snippets verbatim):")
    lines.append(f"Topic: {topic}")
    lines.append("")
    for idx, c in enumerate(calls, start=1):
        lines.append(f"Call {idx} query: {c.query}")
        for j, r in enumerate(c.results, start=1):
            u = (r.url or "").strip()
            t = (r.title or "").strip()
            s = (r.snippet or "").strip()
            lines.append(f"  {j}. {t} — {u}")
            if s:
                lines.append(f"     Snippet: {s}")
        lines.append("")
    return "\n".join(lines).strip()


def dedupe_web_results(responses: List[WebSearchResponse], max_total: int = 30) -> List[WebSearchResponse]:
    """
    Preserve per-call grouping, but dedupe URLs across calls in-place.
    """
    seen: set[str] = set()
    out: List[WebSearchResponse] = []
    total = 0
    for r in responses:
        kept: List[WebSearchResult] = []
        for it in r.results:
            url = (it.url or "").strip()
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            kept.append(it)
            total += 1
            if total >= max_total:
                break
        out.append(WebSearchResponse(query=r.query, results=kept))
        if total >= max_total:
            break
    return out


def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    # Basic URL regex; good enough for our use-case
    urls = re.findall(r"https?://[^\s\)\]\}<>\"']+", text)
    # Strip trailing punctuation
    cleaned = []
    for u in urls:
        cleaned.append(u.rstrip(".,;:!?)\"]}"))
    return cleaned


def has_non_metaculus_url(text: str) -> bool:
    urls = extract_urls(text or "")
    for u in urls:
        if u.startswith(METACULUS_CREDIBLE_SOURCES_URL):
            continue
        return True
    return False


# ---------------------------
# URL accessibility checks (prevents 404 / dead links / restricted pages)
# ---------------------------
URL_CHECK_TIMEOUT_S = 10
URL_CHECK_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": URL_CHECK_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return s

_URL_SESSION = _requests_session()

def check_url_accessible(url: str) -> Tuple[bool, int, str]:
    """
    Returns (ok, status_code, final_url).
    ok=True means the URL is likely reachable by a normal user (HTTP 200-399 after redirects).
    We treat 401/403/404/410/451 and other 4xx/5xx as not accessible by default.
    """
    u = (url or "").strip()
    if not u.startswith("http"):
        return (False, 0, u)

    try:
        r = _URL_SESSION.head(u, allow_redirects=True, timeout=URL_CHECK_TIMEOUT_S)
        code = int(getattr(r, "status_code", 0) or 0)
        final = str(getattr(r, "url", u) or u)

        if code in (405, 400, 0):
            raise RuntimeError("HEAD not usable, fallback to GET")

        if 200 <= code <= 399:
            return (True, code, final)

        if code in (401, 403, 404, 410, 451):
            return (False, code, final)

        if 400 <= code:
            return (False, code, final)

        return (False, code, final)

    except Exception:
        try:
            r2 = _URL_SESSION.get(u, allow_redirects=True, timeout=URL_CHECK_TIMEOUT_S, stream=True)
            code2 = int(getattr(r2, "status_code", 0) or 0)
            final2 = str(getattr(r2, "url", u) or u)

            if 200 <= code2 <= 399:
                return (True, code2, final2)
            if code2 in (401, 403, 404, 410, 451):
                return (False, code2, final2)
            if 400 <= code2:
                return (False, code2, final2)
            return (False, code2, final2)
        except Exception:
            return (False, 0, u)

def url_check_cached(url: str, ttl_s: int = 6 * 3600) -> Tuple[bool, int, str]:
    """
    Cache in Streamlit session_state to avoid re-checking same URLs.
    """
    u = (url or "").strip()
    if not u:
        return (False, 0, u)

    st.session_state.setdefault("_url_check_cache", {})
    cache: Dict[str, Any] = st.session_state["_url_check_cache"]

    now = time.time()
    entry = cache.get(u)
    if entry and (now - float(entry.get("ts", 0))) < ttl_s:
        return (bool(entry["ok"]), int(entry["code"]), str(entry["final"]))

    ok, code, final = check_url_accessible(u)
    cache[u] = {"ok": ok, "code": code, "final": final, "ts": now}
    return (ok, code, final)

def filter_accessible_results(
    results: List[WebSearchResult],
    *,
    max_keep: int,
    strict: bool = True,
) -> List[WebSearchResult]:
    """
    strict=True drops 401/403/404/410/451/5xx.
    strict=False keeps 401/403 as fallback (last resort).
    """
    kept: List[WebSearchResult] = []
    fallback: List[WebSearchResult] = []

    for r in results or []:
        ok, code, final = url_check_cached(r.url)
        rr = WebSearchResult(title=r.title, url=final or r.url, snippet=r.snippet)

        if ok:
            kept.append(rr)
        else:
            if not strict and code in (401, 403):
                fallback.append(rr)

        if len(kept) >= max_keep:
            break

    if len(kept) < max_keep and not strict:
        for rr in fallback:
            kept.append(rr)
            if len(kept) >= max_keep:
                break

    return kept


def score_url_for_resolution(url: str) -> float:
    """
    Heuristic ranking of URLs for resolution use. Higher is better.
    """
    u = (url or "").lower()
    score = 0.0

    # Strong signals: official/stat pages
    for kw in [
        "/statistics", "/stat", "/data", "/dataset", "/series", "/release", "/press",
        "api", "download", "csv", "xlsx", "pdf",
        "bank", "centralbank", "ons.gov.uk", "bea.gov", "bls.gov", "ecb.europa.eu",
        "who.int", "imf.org", "worldbank.org", "oecd.org", "eurostat",
        "gov", ".gouv.", ".gov.", ".int",
    ]:
        if kw in u:
            score += 3.0

    # Slight preference for shorter, “clean” pages
    if len(u) < 120:
        score += 0.5

    # Penalize obvious news/homepages without stable series pages
    for bad in ["?utm_", "facebook.com", "twitter.com", "t.co", "instagram.com", "linkedin.com", "reddit.com"]:
        if bad in u:
            score -= 2.0

    # Penalize metaculus policy link (we want a real resolution URL too)
    if u.startswith(METACULUS_CREDIBLE_SOURCES_URL):
        score -= 10.0

    return score


def pick_best_resolution_urls(web_calls: List[WebSearchResponse], max_urls: int = 8) -> List[str]:
    """
    Select accessible, non-metaculus URLs from web_calls, rank them, and return best.
    """
    urls: List[str] = []
    for c in web_calls or []:
        for r in c.results or []:
            u = (r.url or "").strip()
            if not u:
                continue
            if u.startswith(METACULUS_CREDIBLE_SOURCES_URL):
                continue
            ok, _, final = url_check_cached(u)
            if not ok:
                continue
            urls.append(final or u)

    uniq: Dict[str, float] = {}
    for u in urls:
        if u not in uniq:
            uniq[u] = score_url_for_resolution(u)

    ranked = sorted(uniq.items(), key=lambda kv: kv[1], reverse=True)
    return [u for (u, _) in ranked[:max_urls]]


# ---------------------------
# OpenRouter client
# ---------------------------
@dataclass
class OpenRouterConfig:
    api_key: str
    primary_model: str
    light_model: str
    formatter_model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    use_response_format_json: bool = True
    app_title: str = "Forecast Question Generator"
    http_referer: str = "http://localhost:8501"


class OpenRouterError(RuntimeError):
    pass


def openrouter_chat(
    cfg: OpenRouterConfig,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    plugins: Optional[List[Dict[str, Any]]] = None,
    response_format_override: Optional[Optional[Dict[str, Any]]] = None,
) -> str:
    if not cfg.api_key or len(cfg.api_key.strip()) < 20:
        raise OpenRouterError("OpenRouter API key missing/too short. Paste a valid OpenRouter key.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg.http_referer,
        "X-Title": cfg.app_title,
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(cfg.temperature if temperature_override is None else temperature_override),
        "max_tokens": int(cfg.max_tokens if max_tokens_override is None else max_tokens_override),
    }
    if plugins:
        payload["plugins"] = plugins

    if response_format_override is None:
        if cfg.use_response_format_json:
            payload["response_format"] = {"type": "json_object"}
    else:
        if response_format_override is not None:
            payload["response_format"] = response_format_override

    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    if resp.status_code != 200:
        raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1200]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise OpenRouterError(f"Unexpected response format: {json.dumps(data)[:1500]}")


def repair_to_json(cfg: OpenRouterConfig, raw_text: str, schema_hint: str) -> Any:
    system = (
        "You are a strict JSON repair tool.\n"
        "Return ONLY valid JSON (no markdown, no prose).\n"
        "Use double quotes, no trailing commas.\n"
        "Extract and return the single best JSON object that matches the schema hint."
    )
    user = (
        f"Schema hint:\n{schema_hint}\n\n"
        f"Input text:\n{raw_text}\n\n"
        "Return ONLY the repaired JSON."
    )
    text = openrouter_chat(
        cfg,
        cfg.formatter_model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens_override=min(cfg.max_tokens, 1200),
        temperature_override=0.0,
    )
    return try_parse_json(text)


def call_json_strict(
    cfg: OpenRouterConfig,
    model: str,
    messages: List[Dict[str, str]],
    schema_hint: str,
    *,
    retries: int = 2,
    sleep_s: float = 0.6,
    plugins: Optional[List[Dict[str, Any]]] = None,
    response_format_override: Optional[Optional[Dict[str, Any]]] = None,
) -> Any:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        raw = openrouter_chat(
            cfg,
            model,
            messages,
            plugins=plugins,
            response_format_override=response_format_override,
        )
        try:
            return try_parse_json(raw)
        except Exception as e:
            last_err = e

        try:
            return repair_to_json(cfg, raw_text=raw, schema_hint=schema_hint)
        except Exception as e2:
            last_err = e2

        messages = messages + [
            {
                "role": "user",
                "content": (
                    "INVALID OUTPUT. Return ONLY a SINGLE valid JSON object. "
                    "No markdown. No commentary. Use double quotes. Follow the schema exactly."
                ),
            }
        ]
        time.sleep(sleep_s)

    raise ValueError(f"Model did not return valid JSON after retries. Last error: {last_err}")


# ---------------------------
# Multilevel web enrichment
# ---------------------------
def openrouter_web_search_level(
    cfg: OpenRouterConfig,
    *,
    web_model: str,
    query: str,
    max_results_each: int = 10,
    strict_url_filter: bool = True,
) -> WebSearchResponse:
    web_model2 = ensure_online_variant(web_model)
    plugins = [{"id": "web", "max_results": int(max_results_each)}]

    schema_hint = r"""{
  "query": "string",
  "results": [
    {"title":"string","url":"https://...","snippet":"string"}
  ]
}"""

    system = "Return ONLY a SINGLE valid JSON object. No markdown, no prose. Use double quotes."
    user = (
        f"Web search query: {query}\n\n"
        f"Return exactly the top {max_results_each} results as JSON.\n"
        "Rules:\n"
        "- results[i].url must be a direct clickable URL.\n"
        "- snippet should be short.\n"
        "- Do not include extra keys.\n"
        f"Schema hint:\n{schema_hint}"
    )

    raw = call_json_strict(
        cfg,
        web_model2,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema_hint=schema_hint,
        retries=1,
        plugins=plugins,
        response_format_override=None,
    )

    try:
        resp = WebSearchResponse(**raw)
    except ValidationError:
        q2 = str(raw.get("query", query))
        res_raw = raw.get("results", []) or []
        res2: List[WebSearchResult] = []
        for rr in res_raw:
            try:
                res2.append(WebSearchResult(**rr))
            except Exception:
                continue
        resp = WebSearchResponse(query=q2, results=res2)

    # Filter unreachable and normalize redirects
    resp.results = filter_accessible_results(resp.results, max_keep=max_results_each, strict=bool(strict_url_filter))
    return resp


def propose_next_query_from_prev(
    cfg: OpenRouterConfig,
    web_model: str,
    topic: str,
    prev: WebSearchResponse,
    *,
    level: int,
    hint: str,
    max_results_each: int = 10,
) -> str:
    """
    Uses the online model (no web plugin needed here) to propose a better follow-up query
    based on previous results (titles/urls/snippets).
    """
    web_model2 = ensure_online_variant(web_model)

    schema_hint = r"""{"query":"string"}"""
    prev_json = json.dumps(to_model_dict(prev), ensure_ascii=False)

    system = "Return ONLY a SINGLE valid JSON object. No markdown, no prose. Use double quotes."
    user = (
        f"Task: propose a follow-up web search query for level {level}.\n"
        f"Topic: {topic}\n"
        f"Hint: {hint}\n\n"
        "Constraints:\n"
        "- The query should aim to find stable, resolvable, official or canonical sources.\n"
        "- Prefer official statistical offices, central banks, international organizations, government releases, or stable datasets.\n"
        "- Avoid overly broad queries; add one or two discriminating terms.\n"
        "- Output ONLY JSON: {\"query\":\"...\"}\n\n"
        "Previous web results (JSON):\n"
        f"{prev_json}\n\n"
        "Return ONLY the JSON."
    )

    out = call_json_strict(
        cfg,
        web_model2,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        schema_hint=schema_hint,
        retries=1,
        plugins=None,
        response_format_override=None,
    )
    q = str(out.get("query", "")).strip()
    if not q:
        # Fallback deterministic improvement
        q = f"{topic} official statistics release dataset"
    return q


def openrouter_web_search_multilevel(
    cfg: OpenRouterConfig,
    *,
    web_model: str,
    topic: str,
    level1_query_tpl: str,
    do_level2: bool,
    do_level3: bool,
    level2_hint: str = "Find the primary official resolution source or dataset page.",
    level3_hint: str = "Find a direct release/series page that definitively resolves the question.",
    max_results_each: int = 10,
    strict_url_filter: bool = True,
) -> Tuple[List[WebSearchResponse], str]:
    calls: List[WebSearchResponse] = []

    q1 = (level1_query_tpl or "{topic}").replace("{topic}", topic).strip() or topic.strip()
    r1 = openrouter_web_search_level(
        cfg, web_model=web_model, query=q1, max_results_each=max_results_each, strict_url_filter=strict_url_filter
    )
    calls.append(r1)

    if do_level2:
        q2 = propose_next_query_from_prev(
            cfg, web_model, topic, r1, level=2, hint=level2_hint, max_results_each=max_results_each
        )
        r2 = openrouter_web_search_level(
            cfg, web_model=web_model, query=q2, max_results_each=max_results_each, strict_url_filter=strict_url_filter
        )
        calls.append(r2)

        if do_level3:
            q3 = propose_next_query_from_prev(
                cfg, web_model, topic, r2, level=3, hint=level3_hint, max_results_each=max_results_each
            )
            r3 = openrouter_web_search_level(
                cfg, web_model=web_model, query=q3, max_results_each=max_results_each, strict_url_filter=strict_url_filter
            )
            calls.append(r3)

    calls = dedupe_web_results(calls, max_total=30)
    ctx = format_web_context(topic, calls) if calls else ""
    return calls, ctx


# ---------------------------
# Validation-reformat loop
# ---------------------------
T = TypeVar("T", bound=BaseModel)

SYSTEM_FORMATTER = (
    "You are a schema enforcer.\n"
    "Return ONLY a SINGLE valid JSON object. No markdown, no prose.\n"
    "Use double quotes. Ensure it parses with json.loads().\n"
    "Fix ONLY what is needed for schema compliance; do not change meaning."
)

def validate_or_reformat(
    cfg: OpenRouterConfig,
    data: Any,
    model_cls: Type[T],
    schema_hint: str,
    context_label: str,
    *,
    max_attempts: int = 3,
) -> T:
    cur = data
    last_err: Optional[Exception] = None

    for _attempt in range(1, max_attempts + 1):
        try:
            return model_cls(**cur)  # type: ignore[arg-type]
        except ValidationError as e:
            last_err = e
            err_txt = str(e)

            user = (
                f"Context: {context_label}\n"
                f"Validation error:\n{err_txt}\n\n"
                f"Required schema hint:\n{schema_hint}\n\n"
                "Important constraints:\n"
                f"- For FullQuestion.ai_rating, allowed: {sorted(ALLOWED_RATINGS)}\n"
                f"- VerifyResponse.status allowed: {sorted(ALLOWED_VERIFY_STATUS)}\n"
                "- Do NOT set ai_rating to 'pass' or 'fix'. Those belong only to VerifyResponse.status.\n"
                "- Return ONLY JSON.\n\n"
                "Input JSON:\n"
                f"{json.dumps(cur, ensure_ascii=False)}"
            )
            cur = call_json_strict(
                cfg,
                cfg.formatter_model,
                [{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}],
                schema_hint=schema_hint,
                retries=1,
            )

        except Exception as e:
            last_err = e
            break

    raise ValueError(f"Failed to validate after {max_attempts} formatter attempts ({context_label}). Last error: {last_err}")


# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PRIMARY = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."
SYSTEM_LIGHT = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."


def prompt_generate_protos(topic: str, n: int, start_d: date, end_d: date) -> Tuple[List[Dict[str, str]], str]:
    today_iso = date.today().isoformat()
    _, _, resolve_time = dt_range_defaults(start_d, end_d)
    window_phrase = date_window_phrase_inclusive(start_d, end_d)

    schema_hint = r"""{
  "protos": [
    {
      "title": "Will country X's inflation rate exceed 10% before May 1, 2026?",
      "suggested_type": "binary",
      "description": "Tracks whether inflation in country X reaches a double-digit level by the resolution date.",
      "why_informative": "High inflation has implications for monetary policy, living standards, and financial markets.",
      "candidate_sources": [
        "IMF World Economic Outlook – https://www.imf.org/",
        "World Bank Data – https://data.worldbank.org/"
      ]
    }
  ]
}"""

    user = (
        f"Topic: {topic}\n"
        f"Generate exactly {n} proto-questions.\n"
        f"Each question must be resolvable unambiguously by: {resolve_time}\n"
        f"Current date is: {today_iso}.\n\n"
        "Core goals:\n"
        "- Each question must be decision-relevant or insight-generating, not definitional trivia.\n"
        "- Each question must pass Tetlock's clairvoyance test.\n"
        "- Do NOT write questions about events whose outcomes are already determined before the current date.\n"
        "- Avoid vague language unless tied to a numeric threshold or an official category.\n\n"
        "Date discipline (critical):\n"
        "- Avoid using 'between' and 'by'.\n"
        "- Use explicit inequalities such as 'after <DATE>' / 'before <DATE>' / 'on or before <DATE>'.\n"
        f"- The ONLY acceptable end bound for the event window is the resolution deadline: {end_d.isoformat()}.\n"
        f"- Do NOT accidentally use the current date ({today_iso}) or 'tomorrow' as the end bound.\n"
        f"- If you need an inclusive window from {start_d.isoformat()} to {end_d.isoformat()}, prefer: '{window_phrase}'.\n"
        "- Avoid giving the start date of the question itself; use 'by the time the question is published and before <resolution date>'.\n\n"
        "Avoid overspecification:\n"
        "- Do NOT restrict outcomes to a narrow publication channel unless essential.\n"
        "- Do NOT enumerate media outlets as gatekeepers; instead rely on credible-sources principle.\n"
        f"- If disputes are plausible, prefer a short reference to Metaculus credible sources policy: {METACULUS_CREDIBLE_SOURCES_URL}\n"
        "- Exception (high-salience outcomes like national election winners): prefer an official authority as primary, but if unavailable, allow resolution via general credible-source consensus consistent with Metaculus guidance.\n\n"
        "Type rules:\n"
        "- suggested_type must be one of: binary | numeric | multiple_choice.\n"
        "- Avoid disguised-binary numeric.\n"
        "- Use numeric type only when a range of values is genuinely informative.\n"
        "- Use multiple_choice only when there are >= 4 genuinely distinct, plausible options.\n\n"
        "Source rules:\n"
        "- candidate_sources should contain named sources + URLs whenever feasible.\n"
        "- Prefer reputable and stable sources (official stats, intl orgs, etc.).\n\n"
        "Output format:\n"
        "- Return ONLY valid JSON, with a single top-level object having a 'protos' array.\n"
        "- Each entry must match the structure shown in the schema hint.\n\n"
        f"Schema hint example:\n{schema_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PRIMARY},
        {"role": "user", "content": user},
    ]
    return messages, schema_hint


def prompt_select_k(topic: str, protos: List[ProtoQuestion], k: int) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"selected":[{PROTO_OBJECTS_SUBSET}],"selection_rationale":"..."}"""
    protos_json = json.dumps([to_model_dict(p) for p in protos], ensure_ascii=False)
    user = (
        f"Topic: {topic}\n"
        f"Select the best {k} proto-questions from the list.\n\n"
        "Criteria: info value, resolvability, non-redundant.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Proto list:\n{protos_json}"
    )
    return ([{"role": "system", "content": SYSTEM_PRIMARY}, {"role": "user", "content": user}], schema_hint)


def prompt_canonicalize_selected_protos(selected: List[ProtoQuestion]) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"selected":[{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}]}"""
    payload = json.dumps([to_model_dict(p) for p in selected], ensure_ascii=False)
    user = (
        "Canonicalize the selected proto-questions.\n"
        "Do NOT change meaning; only ensure schema compliance.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Selected protos:\n{payload}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


def prompt_build_full(topic: str, proto: ProtoQuestion, start_d: date, end_d: date) -> Tuple[List[Dict[str, str]], str]:
    today_iso = date.today().isoformat()
    open_time, close_time, resolve_time = dt_range_defaults(start_d, end_d)
    window_phrase = date_window_phrase_inclusive(start_d, end_d)

    schema_hint = r"""{
  "title": "Will country X's inflation rate exceed 10% before May 1, 2026?",
  "type": "binary",
  "resolution_criteria": "Resolves Yes if <SOURCE> reports <CONDITION> on or before 2026-05-01. Otherwise No. Primary source: https://example.com/series",
  "fine_print": "Handle revisions and disputes per Metaculus credible sources guidance.",
  "description": "Tracks whether country X reaches double-digit inflation by the resolution date. Resolution source: https://example.com/series",
  "question_weight": 1.0,
  "open_time": "2026-01-01T00:00:00+01:00",
  "scheduled_close_time": "2026-05-01T23:59:00+01:00",
  "scheduled_resolve_time": "2026-05-01T23:59:00+01:00",
  "range_min": null,
  "range_max": null,
  "zero_point": null,
  "open_lower_bound": null,
  "open_upper_bound": null,
  "unit": null,
  "group_variable": null,
  "options": null,
  "categories": ["Macroeconomics", "Inflation"],
  "ai_rating": "accept_for_aib",
  "ai_rationale": "Falsifiable, decision-relevant, clear data sources, no major ambiguity."
}"""

    proto_json = json.dumps(to_model_dict(proto), ensure_ascii=False)

    user = (
        f"Topic: {topic}\n"
        f"Current date is: {today_iso}.\n"
        "Build a single full forecasting question object (FullQuestion) from the proto.\n\n"
        "Conceptual requirements:\n"
        "- Must pass Tetlock's clairvoyance test.\n"
        "- The event must NOT already be decided before the current date.\n"
        "- Avoid vague language unless tied to precise thresholds or official categories.\n\n"
        "Date discipline (critical):\n"
        "- Avoid 'between' and 'by'. Use explicit 'after <DATE>', 'before <DATE>', 'on or before <DATE>'.\n"
        f"- The ONLY acceptable end bound for the event window is {end_d.isoformat()}.\n"
        f"- Do NOT use the current date ({today_iso}) or 'tomorrow' as the end bound.\n"
        f"- If you need an inclusive window from {start_d.isoformat()} to {end_d.isoformat()}, prefer: '{window_phrase}'.\n\n"
        "Avoid overspecification:\n"
        "- Do NOT unnecessarily restrict the event to a specific announcement channel.\n"
        "- Do NOT list specific media outlets as mandatory/forbidden.\n"
        f"- If needed, reference Metaculus credible sources policy: {METACULUS_CREDIBLE_SOURCES_URL}\n"
        "- Exception (high-salience outcomes like election winners): official authority primary; credible-source consensus only as fallback.\n\n"
        "Resolution criteria & sources:\n"
        "- resolution_criteria must be executable.\n"
        "- Include at least one direct URL to a primary resolution source when feasible.\n"
        "- If sources conflict, specify precedence.\n\n"
        "Time rules:\n"
        f"- open_time must be >= {open_time}.\n"
        f"- scheduled_close_time must be <= {close_time}.\n"
        f"- scheduled_resolve_time must be <= {resolve_time}.\n\n"
        "Type-specific rules:\n"
        "- type must be one of: binary | numeric | multiple_choice.\n"
        "- Numeric must have meaningful range_min/range_max and unit.\n"
        "- MCQ must have >=4 meaningful options.\n\n"
        "AI rating:\n"
        f"- ai_rating must be one of: {sorted(ALLOWED_RATINGS)}.\n"
        "- Do NOT use 'pass' or 'fix' as ai_rating.\n\n"
        "Output format:\n"
        "- Return ONLY a single JSON object matching FullQuestion.\n"
        "- Use null for missing optional fields.\n\n"
        f"Proto object:\n{proto_json}\n\n"
        f"Schema hint example:\n{schema_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PRIMARY},
        {"role": "user", "content": user},
    ]
    return messages, schema_hint


def prompt_verify(full_q: FullQuestion, start_d: date, end_d: date) -> Tuple[List[Dict[str, str]], str]:
    end_iso = iso_dt(end_d, dtime(23, 59))
    today_iso = date.today().isoformat()

    schema_hint = r"""{
  "status": "pass",
  "issues": ["..."],
  "question": { FULL_QUESTION_OBJECT }
}"""

    full_json = json.dumps(to_model_dict(full_q), ensure_ascii=False)

    user = (
        "You are a strict validator and light editor for forecasting questions.\n\n"
        "Task:\n"
        "- Inspect the provided FullQuestion.\n"
        "- If it satisfies all constraints, return status='pass' and the (possibly lightly cleaned) question.\n"
        "- If it violates any constraint, minimally fix it and return status='fix' plus issues.\n\n"
        "Critical date-discipline check (this is a common failure):\n"
        f"- The resolution deadline is {end_d.isoformat()} (scheduled_resolve_time must be <= {end_iso}).\n"
        f"- If any text uses the current date ({today_iso}) or 'tomorrow' or a 1-2 day window as the END bound, replace it with the resolution deadline.\n"
        f"- If an inclusive window is needed for {start_d.isoformat()}..{end_d.isoformat()}, use: '{date_window_phrase_inclusive(start_d, end_d)}'.\n"
        "- Avoid 'between'/'by'. Prefer explicit inequalities.\n\n"
        "Overspecification checks:\n"
        "- Remove unnecessary restrictions on how an event must be announced.\n"
        "- Avoid enumerations of outlets; use credible-sources guidance if needed.\n\n"
        "Resolution URL check:\n"
        "- Prefer having at least one direct primary URL in resolution_criteria. If missing, do not invent; keep text minimal.\n\n"
        "Output format:\n"
        "- Return ONLY one JSON object matching VerifyResponse structure.\n\n"
        f"Question to validate:\n{full_json}\n\n"
        f"Schema hint:\n{schema_hint}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_LIGHT},
        {"role": "user", "content": user},
    ]
    return messages, schema_hint


def prompt_format_single(full_q: FullQuestion) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"question":FULL_QUESTION_OBJECT}"""
    full_json = json.dumps(to_model_dict(full_q), ensure_ascii=False)
    user = (
        "Canonicalize this single FullQuestion into the exact schema.\n"
        "Do not change meaning; only fix formatting/schema issues.\n\n"
        "Rules:\n"
        "- options must be null for non-MCQ; list for MCQ\n"
        "- categories must be a non-empty list\n"
        "- use null for missing optional fields\n"
        f"- ai_rating must be one of {sorted(ALLOWED_RATINGS)}\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Input:\n{full_json}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


def prompt_rebalance(
    questions: List[FullQuestion],
    target_binary: int,
    target_numeric: int,
    target_mcq: int,
    end_d: date,
) -> Tuple[List[Dict[str, str]], str]:
    end_iso = iso_dt(end_d, dtime(23, 59))
    schema_hint = """{"questions":[FULL_QUESTION_OBJECTS],"change_log":["..."]}"""
    qs_json = json.dumps([to_model_dict(q) for q in questions], ensure_ascii=False)
    user = (
        "Minimally edit questions to match target type distribution.\n\n"
        f"Targets: binary={target_binary}, numeric={target_numeric}, multiple_choice={target_mcq}\n"
        f"Constraint: scheduled_resolve_time <= {end_iso}\n"
        "No disguised-binary numeric. MCQ >=4 meaningful options.\n\n"
        "Keep edits disciplined:\n"
        "- Avoid introducing overspecification.\n"
        "- Keep date language explicit (avoid 'between'/'by').\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Questions:\n{qs_json}"
    )
    return ([{"role": "system", "content": SYSTEM_LIGHT}, {"role": "user", "content": user}], schema_hint)


def prompt_canonicalize_questions_list(questions: List[FullQuestion]) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"questions":[FULL_QUESTION_OBJECTS]}"""
    qs_json = json.dumps([to_model_dict(q) for q in questions], ensure_ascii=False)
    user = (
        "Canonicalize the list of FullQuestion objects.\n"
        "Do NOT change meaning; only ensure schema compliance.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Input:\n{qs_json}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


def prompt_clean_questions(questions: List[FullQuestion]) -> Tuple[List[Dict[str, str]], str]:
    schema_hint = """{"questions":[FULL_QUESTION_OBJECTS],"notes":["optional"]}"""
    qs_json = json.dumps([to_model_dict(q) for q in questions], ensure_ascii=False)
    user = (
        "Return cleaned strict JSON with the same questions. Minimal changes only for schema correctness.\n\n"
        "Cleanup goals (minimal):\n"
        "- Keep date language explicit (avoid 'between'/'by').\n"
        "- Remove accidental overspecification if present.\n"
        "- Keep terminology aligned with cited sources.\n\n"
        f"Return JSON exactly matching schema:\n{schema_hint}\n\n"
        f"Input:\n{qs_json}"
    )
    return ([{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}], schema_hint)


def prompt_inject_resolution_url(
    full_q: FullQuestion,
    best_urls: List[str],
) -> Tuple[List[Dict[str, str]], str]:
    """
    Minimal edit: ensure resolution_criteria includes one accessible primary URL (not Metaculus policy),
    and ensure description also includes that URL (as "Resolution source: <url>").
    """
    schema_hint = r"""{ FULL_QUESTION_OBJECT }"""
    qjson = json.dumps(to_model_dict(full_q), ensure_ascii=False)
    urls_txt = "\n".join(f"- {u}" for u in best_urls[:8])

    system = SYSTEM_FORMATTER
    user = (
        "You are a careful editor.\n"
        "Goal: ensure the question contains at least one primary resolution URL.\n\n"
        "Rules (strict):\n"
        "- Make the MINIMUM possible textual changes.\n"
        "- Do NOT change the meaning or thresholds.\n"
        "- Add exactly ONE best URL (choose the most canonical/official) from the provided list.\n"
        "- Insert it into resolution_criteria (e.g., 'Primary source: <url>') and also into description "
        "(e.g., 'Resolution source: <url>').\n"
        "- Do NOT add Metaculus credible sources policy URL as the only URL.\n"
        "- Return ONLY a single JSON object matching the FullQuestion schema.\n\n"
        "Candidate accessible URLs:\n"
        f"{urls_txt}\n\n"
        "Input FullQuestion JSON:\n"
        f"{qjson}\n\n"
        "Return ONLY JSON."
    )
    return ([{"role": "system", "content": system}, {"role": "user", "content": user}], schema_hint)


# ---------------------------
# Orchestration
# ---------------------------
def compute_targets(total: int) -> Tuple[int, int, int]:
    """
    Targets: 60% binary, 20% numeric, 20% MCQ
    """
    b = int(round(total * 0.60))
    n = int(round(total * 0.20))
    m = total - b - n
    if m < 0:
        m = 0
        while b + n + m > total and n > 0:
            n -= 1
        while b + n + m > total and b > 0:
            b -= 1
    return b, n, m


def count_types(questions: List[FullQuestion]) -> Dict[str, int]:
    c = {"binary": 0, "numeric": 0, "multiple_choice": 0}
    for q in questions:
        c[q.type] = c.get(q.type, 0) + 1
    return c


def ensure_resolution_url_present(
    cfg: OpenRouterConfig,
    fq: FullQuestion,
    web_calls: List[WebSearchResponse],
) -> FullQuestion:
    """
    Before formatting/export, enforce: at least one non-metaculus URL exists in resolution_criteria.
    If missing, inject the best accessible URL from web_calls into resolution_criteria and description.
    """
    if has_non_metaculus_url(fq.resolution_criteria or ""):
        return fq

    best = pick_best_resolution_urls(web_calls, max_urls=8)
    if not best:
        # Nothing to inject; keep as-is
        return fq

    msgs, hint = prompt_inject_resolution_url(fq, best)
    raw = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
    fq2 = validate_or_reformat(cfg, raw, FullQuestion, """FULL_QUESTION_OBJECT""", "Inject resolution URL", max_attempts=3)
    return normalize_full_question_fields(fq2)


def generate_for_topic_iter(
    cfg: OpenRouterConfig,
    topic: str,
    n: int,
    k: int,
    start_d: date,
    end_d: date,
    use_formatter_after_selection: bool,
    progress_cb=None,
    *,
    enable_web: bool = False,
    web_model: str = "openai/gpt-4.1-mini:online",
    level1_query_tpl: str = "{topic}",
    do_level2: bool = True,
    do_level3: bool = True,
    web_max_results_each: int = 10,
    strict_url_filter: bool = True,
):
    # 0) multilevel web enrichment
    web_ctx = ""
    web_calls: List[WebSearchResponse] = []
    if enable_web:
        if progress_cb:
            progress_cb(f"Web enrichment: multilevel (1→2→3) for topic: {topic}")
        web_calls, web_ctx = openrouter_web_search_multilevel(
            cfg,
            web_model=web_model,
            topic=topic,
            level1_query_tpl=level1_query_tpl,
            do_level2=do_level2,
            do_level3=do_level3,
            max_results_each=int(web_max_results_each),
            strict_url_filter=bool(strict_url_filter),
        )
        st.session_state.setdefault("web_log", {})
        st.session_state["web_log"][topic] = {
            "calls": [to_model_dict(c) for c in web_calls],
            "context": web_ctx,
        }

    # 1) protos
    if progress_cb:
        progress_cb(f"Generating {n} protos for topic: {topic}")
    msgs, hint = prompt_generate_protos(topic, n, start_d, end_d)
    msgs = inject_context(msgs, web_ctx)
    raw = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    proto_hint = """{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}"""
    protos = [validate_or_reformat(cfg, p, ProtoQuestion, proto_hint, "ProtoQuestion") for p in (raw.get("protos", []) or [])][:n]
    if not protos:
        raise ValueError(f"No protos generated for topic: {topic}")

    # 2) select k
    if progress_cb:
        progress_cb(f"Selecting top {k} protos for topic: {topic}")
    msgs, hint = prompt_select_k(topic, protos, k)
    raw2 = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    selected_raw = raw2.get("selected", []) or []
    selected = [validate_or_reformat(cfg, p, ProtoQuestion, proto_hint, "Selected ProtoQuestion") for p in selected_raw][:k]
    if not selected:
        selected = protos[:k]

    # 3) optional canonicalize selected protos
    if use_formatter_after_selection:
        if progress_cb:
            progress_cb(f"Canonicalizing selected protos (formatter) for topic: {topic}")
        msgs, hint = prompt_canonicalize_selected_protos(selected)
        canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
        canon_list = canon.get("selected", []) or []
        if canon_list:
            selected = [validate_or_reformat(cfg, p, ProtoQuestion, hint, "Canonicalized ProtoQuestion") for p in canon_list]

    # 4) per selected: build -> verify -> ensure URL -> format -> yield
    for i, proto in enumerate(selected, start=1):
        if progress_cb:
            progress_cb(f"Building full card {i}/{len(selected)} for topic: {topic}")
        msgs, hint = prompt_build_full(topic, proto, start_d, end_d)
        msgs = inject_context(msgs, web_ctx)
        raw3 = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)
        fq = validate_or_reformat(cfg, raw3, FullQuestion, hint, f"Build FullQuestion ({topic} #{i})")
        fq = normalize_full_question_fields(fq)

        if progress_cb:
            progress_cb(f"Verifying card {i}/{len(selected)} for topic: {topic}")
        msgs, hint_v = prompt_verify(fq, start_d, end_d)
        raw4 = call_json_strict(cfg, cfg.light_model, msgs, schema_hint=hint_v)
        vr = validate_or_reformat(cfg, raw4, VerifyResponse, hint_v, f"VerifyResponse ({topic} #{i})", max_attempts=3)
        fq2 = normalize_full_question_fields(vr.question)

        # NEW: ensure at least one non-metaculus URL in resolution_criteria (inject from web results if missing)
        if progress_cb:
            progress_cb(f"Ensuring resolution URL present {i}/{len(selected)} for topic: {topic}")
        fq2b = ensure_resolution_url_present(cfg, fq2, web_calls)

        if progress_cb:
            progress_cb(f"Formatting card {i}/{len(selected)} for topic: {topic}")
        msgs, hint_f = prompt_format_single(fq2b)
        raw5 = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint_f, retries=1)
        fq_final = validate_or_reformat(
            cfg, raw5.get("question", raw5), FullQuestion, """FULL_QUESTION_OBJECT""", f"Formatted FullQuestion ({topic} #{i})", max_attempts=3
        )
        fq_final = normalize_full_question_fields(fq_final)

        yield fq_final


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Topics -> Forecast Questions CSV", layout="wide")
st.title("Topics CSV → Forecast Questions → CSV Export (multilevel web + URL filtering + URL injection)")

colA, colB = st.columns(2)

with colA:
    api_key = st.text_input("OpenRouter API key", type="password").strip()
    primary_model = st.text_input("Primary model (generation)", value="openai/gpt-4.1-mini")
    light_model = st.text_input("Light model (verify / rebalance)", value="openai/gpt-4.1-mini")
    formatter_model = st.text_input("Formatter model (schema enforcement / repair)", value="openai/gpt-4.1-nano")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.number_input("Max tokens per call", min_value=500, max_value=8000, value=2000, step=100)
    use_rf = st.checkbox("Use response_format=json_object (if supported)", value=True)

with colB:
    uploaded = st.file_uploader("Upload topics CSV", type=["csv"])
    n = st.number_input("n = proto questions per topic", min_value=1, max_value=50, value=6, step=1)
    k = st.number_input("k = keep per topic", min_value=1, max_value=20, value=3, step=1)
    start_d = st.date_input("Questions start date", value=date.today())
    end_d = st.date_input("Questions end date (must resolve by this date)", value=date.today())
    do_rebalance = st.checkbox("Rebalance to ~60% binary / 20% numeric / 20% MCQ", value=True)
    use_formatter_after_selection = st.checkbox("Formatter after selecting k protos (recommended)", value=True)
    use_formatter_after_rebalance = st.checkbox("Formatter after rebalancing (recommended)", value=True)
    do_final_clean = st.checkbox("Final clean pass (formatter)", value=True)

st.divider()

st.subheader("Optional: Web enrichment (multilevel 1→2→3, max_results=10 each)")

web_enable = st.checkbox("Enable web enrichment", value=True)
web_model = st.text_input("Web model (must support :online)", value="openai/gpt-4.1-mini:online")

strict_url_filter = st.checkbox("Filter inaccessible URLs (drop 404/410/401/403/5xx)", value=True)

st.caption(
    "Multilevel behavior: Call 2 query is generated from Call 1 results; Call 3 from Call 2. "
    "Objective: reliably obtain an accessible primary resolution URL."
)

level1_tpl = st.text_input("Level 1 query template", value="{topic}")
do_level2 = st.checkbox("Run Level 2 (follow-up query)", value=True)
do_level3 = st.checkbox("Run Level 3 (follow-up query)", value=True)

web_max_each = 10  # fixed per request

topics: List[str] = []
if uploaded:
    try:
        fieldnames, rows = read_topics_csv(uploaded)
        default_col = pick_topic_column(fieldnames, rows)
        chosen_col = st.selectbox("Topic column", options=fieldnames, index=fieldnames.index(default_col))
        topics_raw = [str(r.get(chosen_col, "")).strip() for r in rows]
        topics = [t for t in topics_raw if t]
        st.write(f"Detected {len(topics)} topics.")
        if topics:
            st.dataframe({"topic": topics[: min(30, len(topics))]})
    except Exception as e:
        st.error(f"Failed to read topics CSV: {e}")
        topics = []

run = st.button("Generate CSV", type="primary", disabled=not (api_key and topics and start_d and end_d))

status_box = st.empty()
progress = st.progress(0)

draft_info_ph = st.empty()
draft_preview_ph = st.empty()
final_info_ph = st.empty()
final_preview_ph = st.empty()

web_preview_ph = st.empty()

def progress_cb(msg: str):
    status_box.info(msg)

if run:
    if end_d < start_d:
        st.error("End date must be >= start date.")
        st.stop()
    if int(k) > int(n):
        st.error("k must be <= n.")
        st.stop()

    cfg = OpenRouterConfig(
        api_key=api_key,
        primary_model=primary_model.strip(),
        light_model=light_model.strip(),
        formatter_model=formatter_model.strip(),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        use_response_format_json=bool(use_rf),
    )

    try:
        # Rough step count for UI progress
        per_topic = 2 + (1 if use_formatter_after_selection else 0) + int(k) * 4  # build/verify/ensureURL/format
        if web_enable:
            per_topic += 1  # web enrichment bucket

        total_steps = max(1, len(topics)) * per_topic
        if do_rebalance:
            total_steps += 1
            if use_formatter_after_rebalance:
                total_steps += 1
        if do_final_clean:
            total_steps += 1

        done = {"value": 0}
        def tick(msg: str):
            done["value"] += 1
            progress.progress(min(1.0, done["value"] / max(1, total_steps)))
            progress_cb(msg)

        draft_questions: List[FullQuestion] = []
        draft_rows: List[Dict[str, Any]] = []

        for t_i, topic in enumerate(topics, start=1):
            tick(f"=== Topic {t_i}/{len(topics)}: {topic} ===")

            for q in generate_for_topic_iter(
                cfg=cfg,
                topic=topic,
                n=int(n),
                k=int(k),
                start_d=start_d,
                end_d=end_d,
                use_formatter_after_selection=use_formatter_after_selection,
                progress_cb=tick,
                enable_web=bool(web_enable),
                web_model=web_model.strip(),
                level1_query_tpl=level1_tpl.strip() or "{topic}",
                do_level2=bool(do_level2),
                do_level3=bool(do_level3),
                web_max_results_each=web_max_each,
                strict_url_filter=bool(strict_url_filter),
            ):
                draft_questions.append(q)
                draft_rows.append(full_question_to_row(q))

                draft_info_ph.markdown(
                    f"**Draft rows:** {len(draft_rows)} | **Draft type counts:** {count_types(draft_questions)}"
                )
                draft_preview_ph.dataframe(draft_rows[-min(20, len(draft_rows)) :], use_container_width=True)

            if web_enable:
                wlog = st.session_state.get("web_log", {}).get(topic)
                if wlog and wlog.get("context"):
                    with web_preview_ph.container():
                        with st.expander(f"Web enrichment preview — {topic}", expanded=False):
                            st.text(wlog["context"])

        draft_csv_text = write_csv(draft_rows)

        # Final passes
        final_questions = list(draft_questions)
        log: Dict[str, Any] = {"type_counts_draft": count_types(draft_questions)}

        if do_rebalance and final_questions:
            tick("Rebalancing types to ~60/20/20.")
            total = len(final_questions)
            tb, tn, tm = compute_targets(total)
            msgs, hint = prompt_rebalance(final_questions, tb, tn, tm, end_d)
            raw = call_json_strict(cfg, cfg.light_model, msgs, schema_hint=hint)
            edited = raw.get("questions", []) or []

            tmp: List[FullQuestion] = []
            for idx, qd in enumerate(edited, start=1):
                fq = validate_or_reformat(cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Rebalance FullQuestion #{idx}", max_attempts=3)
                tmp.append(normalize_full_question_fields(fq))
            final_questions = tmp
            log["rebalance_change_log"] = raw.get("change_log", [])
            log["type_counts_after_rebalance"] = count_types(final_questions)

            if use_formatter_after_rebalance:
                tick("Canonicalizing post-rebalance list (formatter).")
                msgs, hint2 = prompt_canonicalize_questions_list(final_questions)
                canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint2, retries=1)
                canon_list = canon.get("questions", []) or []
                if canon_list:
                    tmp2: List[FullQuestion] = []
                    for idx, qd in enumerate(canon_list, start=1):
                        fq = validate_or_reformat(cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Post-rebalance canonical FullQuestion #{idx}", max_attempts=3)
                        tmp2.append(normalize_full_question_fields(fq))
                    final_questions = tmp2

        if do_final_clean and final_questions:
            tick("Final clean pass (formatter).")
            msgs, hint = prompt_clean_questions(final_questions)
            raw = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
            cleaned = raw.get("questions", []) or []
            tmp3: List[FullQuestion] = []
            for idx, qd in enumerate(cleaned, start=1):
                fq = validate_or_reformat(cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Final clean FullQuestion #{idx}", max_attempts=3)
                tmp3.append(normalize_full_question_fields(fq))
            final_questions = tmp3
            log["clean_notes"] = raw.get("notes", [])
            log["type_counts_final"] = count_types(final_questions)

        final_rows = [full_question_to_row(q) for q in final_questions]
        final_csv_text = write_csv(final_rows)

        final_info_ph.markdown(
            f"**Final rows:** {len(final_rows)} | **Final type counts:** {count_types(final_questions)}"
        )
        final_preview_ph.dataframe(final_rows[: min(20, len(final_rows))], use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="Download DRAFT CSV (incremental)",
                data=draft_csv_text.encode("utf-8"),
                file_name="forecast_questions_DRAFT.csv",
                mime="text/csv",
            )
        with col_dl2:
            st.download_button(
                label="Download FINAL CSV",
                data=final_csv_text.encode("utf-8"),
                file_name="forecast_questions_FINAL.csv",
                mime="text/csv",
            )

        with st.expander("Logs / change log", expanded=False):
            st.json(log)

        st.success("Done.")

    except (OpenRouterError, ValidationError, ValueError) as e:
        st.error(str(e))
        if "draft_rows" in locals() and draft_rows:
            salvage_csv = write_csv(draft_rows)
            st.download_button(
                label="Download SALVAGED DRAFT CSV (partial)",
                data=salvage_csv.encode("utf-8"),
                file_name="forecast_questions_SALVAGED_DRAFT.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        if "draft_rows" in locals() and draft_rows:
            salvage_csv = write_csv(draft_rows)
            st.download_button(
                label="Download SALVAGED DRAFT CSV (partial)",
                data=salvage_csv.encode("utf-8"),
                file_name="forecast_questions_SALVAGED_DRAFT.csv",
                mime="text/csv",
            )
