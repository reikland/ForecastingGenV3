import re
import time
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st

from forecasting_app.constants import METACULUS_CREDIBLE_SOURCES_URL
from forecasting_app.schemas import WebSearchResponse, WebSearchResult


def format_web_context(topic: str, calls: List[WebSearchResponse]) -> str:
    lines: List[str] = []
    lines.append("WEB SEARCH CONTEXT (use ONLY to pick candidate sources/resolution links; do not copy snippets verbatim):")
    lines.append(f"Topic: {topic}")
    for i, c in enumerate(calls, start=1):
        lines.append(f"\nCall {i} query: {c.query}")
        for r in c.results or []:
            lines.append(f"- {r.title} | {r.url} | {r.snippet}")
    return "\n".join(lines)


def dedupe_web_results(responses: List[WebSearchResponse], max_total: int = 30) -> List[WebSearchResponse]:
    seen = set()
    out: List[WebSearchResponse] = []
    for r in responses:
        rr = WebSearchResponse(query=r.query, results=[])
        for it in r.results:
            u = (it.url or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            rr.results.append(it)
        if rr.results:
            out.append(rr)
        if sum(len(x.results) for x in out) >= max_total:
            break
    return out


def extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s)]+", text or "")


def has_non_metaculus_url(text: str) -> bool:
    urls = extract_urls(text)
    for u in urls:
        if not u.startswith(METACULUS_CREDIBLE_SOURCES_URL):
            return True
    return False


def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ForecastGen/1.0)"})
    return s


def check_url_accessible(url: str) -> Tuple[bool, int, str]:
    """
    Returns (ok, status_code, final_url). ok if status 200-399; rejects 401/403/404/410/5xx.
    """
    try:
        s = _requests_session()
        resp = s.head(url, allow_redirects=True, timeout=12)
        status = resp.status_code
        final = resp.url or url
        if status >= 500 or status in (401, 403, 404, 410):
            return False, status, final
        if status < 400:
            return True, status, final

        resp = s.get(url, allow_redirects=True, timeout=12)
        status = resp.status_code
        final = resp.url or url
        if status >= 500 or status in (401, 403, 404, 410):
            return False, status, final
        return status < 400, status, final
    except Exception:
        return False, 0, url


def url_check_cached(url: str, ttl_s: int = 6 * 3600) -> Tuple[bool, int, str]:
    """
    Cache URL accessibility checks in session_state to avoid repeated HEAD/GETs.
    """
    st.session_state.setdefault("_url_check_cache", {})
    cache: Dict[str, Any] = st.session_state["_url_check_cache"]
    now = time.time()
    if url in cache:
        ok, status, final, ts = cache[url]
        if now - ts < ttl_s:
            return ok, status, final
    ok, status, final = check_url_accessible(url)
    cache[url] = (ok, status, final, now)
    return ok, status, final


def filter_accessible_results(
    results: List[WebSearchResult],
    max_keep: int = 10,
    strict: bool = True,
) -> List[WebSearchResult]:
    if not strict:
        return results[:max_keep]

    kept: List[WebSearchResult] = []
    for r in results:
        u = (r.url or "").strip()
        if not u:
            continue
        ok, _, final = url_check_cached(u)
        if not ok:
            continue
        if final and final != u:
            r = WebSearchResult(title=r.title, url=final, snippet=r.snippet)
        kept.append(r)
        if len(kept) >= max_keep:
            break
    return kept


def score_url_for_resolution(url: str) -> float:
    """
    Heuristic: prioritize official-looking sources.
    """
    score = 0.0
    u = url.lower()
    if re.search(r"\.gov|\.gouv|\.govt|\.stat", u):
        score += 2.0
    if any(k in u for k in ["worldbank", "imf.org", "oecd", "who.int", "un.org", "ecb.europa.eu", "federalreserve"]):
        score += 1.5
    if "data" in u or "dataset" in u or "statistics" in u:
        score += 0.5
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
