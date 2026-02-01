from datetime import date
from typing import Dict, List, Tuple

import streamlit as st

from forecasting_app.model_utils import inject_context, normalize_full_question_fields, to_model_dict
from forecasting_app.openrouter_client import OpenRouterConfig, call_json_strict
from forecasting_app.prompts.lists import (
    prompt_canonicalize_questions_list,
    prompt_clean_questions,
    prompt_rebalance,
)
from forecasting_app.prompts.protos import (
    prompt_canonicalize_selected_protos,
    prompt_generate_protos,
    prompt_select_k,
)
from forecasting_app.prompts.questions import (
    prompt_build_full,
    prompt_format_single,
    prompt_inject_resolution_url,
    prompt_verify,
)
from forecasting_app.schemas import FullQuestion, ProtoQuestion, VerifyResponse, WebSearchResponse
from forecasting_app.validation import validate_or_reformat
from forecasting_app.web_search import openrouter_web_search_multilevel
from forecasting_app.web_utils import has_non_metaculus_url, pick_best_resolution_urls


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

    if progress_cb:
        progress_cb(f"Generating {n} protos for topic: {topic}")
    msgs, hint = prompt_generate_protos(topic, n, start_d, end_d)
    msgs = inject_context(msgs, web_ctx)
    raw = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    proto_hint = """{"title":"...","suggested_type":"binary|numeric|multiple_choice","description":"...","why_informative":"...","candidate_sources":["..."]}"""
    protos = [validate_or_reformat(cfg, p, ProtoQuestion, proto_hint, "ProtoQuestion") for p in (raw.get("protos", []) or [])][:n]
    if not protos:
        raise ValueError(f"No protos generated for topic: {topic}")

    if progress_cb:
        progress_cb(f"Selecting top {k} protos for topic: {topic}")
    msgs, hint = prompt_select_k(topic, protos, k)
    raw2 = call_json_strict(cfg, cfg.primary_model, msgs, schema_hint=hint)

    selected_raw = raw2.get("selected", []) or []
    selected = [validate_or_reformat(cfg, p, ProtoQuestion, proto_hint, "Selected ProtoQuestion") for p in selected_raw][:k]
    if not selected:
        selected = protos[:k]

    if use_formatter_after_selection:
        if progress_cb:
            progress_cb(f"Canonicalizing selected protos (formatter) for topic: {topic}")
        msgs, hint = prompt_canonicalize_selected_protos(selected)
        canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
        canon_list = canon.get("selected", []) or []
        if canon_list:
            selected = [validate_or_reformat(cfg, p, ProtoQuestion, hint, "Canonicalized ProtoQuestion") for p in canon_list]

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
