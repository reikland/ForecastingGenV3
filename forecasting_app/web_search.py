import json
from typing import List, Tuple

from pydantic import ValidationError

from forecasting_app.model_utils import ensure_online_variant, to_model_dict
from forecasting_app.openrouter_client import OpenRouterConfig, call_json_strict
from forecasting_app.schemas import WebSearchResponse, WebSearchResult
from forecasting_app.web_utils import dedupe_web_results, filter_accessible_results, format_web_context


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
