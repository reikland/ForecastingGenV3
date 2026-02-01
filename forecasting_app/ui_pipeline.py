from typing import Any, Dict, List

import streamlit as st
from pydantic import ValidationError

from forecasting_app.csv_utils import full_question_to_row, write_csv
from forecasting_app.model_utils import normalize_full_question_fields
from forecasting_app.openrouter_client import OpenRouterConfig, OpenRouterError, call_json_strict
from forecasting_app.orchestration import compute_targets, count_types, generate_for_topic_iter
from forecasting_app.prompts.lists import (
    prompt_canonicalize_questions_list,
    prompt_clean_questions,
    prompt_rebalance,
)
from forecasting_app.schemas import FullQuestion
from forecasting_app.validation import validate_or_reformat


def run_generation_pipeline(
    cfg: OpenRouterConfig,
    model_inputs: Dict[str, Any],
    web_inputs: Dict[str, Any],
    topics: List[str],
    *,
    status_box,
    progress,
    draft_info_ph,
    draft_preview_ph,
    final_info_ph,
    final_preview_ph,
    web_preview_ph,
) -> None:
    def progress_cb(msg: str) -> None:
        status_box.info(msg)

    try:
        per_topic = 2 + (1 if model_inputs["use_formatter_after_selection"] else 0) + int(model_inputs["k"]) * 4
        if web_inputs["web_enable"]:
            per_topic += 1

        total_steps = max(1, len(topics)) * per_topic
        if model_inputs["do_rebalance"]:
            total_steps += 1
            if model_inputs["use_formatter_after_rebalance"]:
                total_steps += 1
        if model_inputs["do_final_clean"]:
            total_steps += 1

        done = {"value": 0}

        def tick(msg: str) -> None:
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
                n=int(model_inputs["n"]),
                k=int(model_inputs["k"]),
                start_d=model_inputs["start_d"],
                end_d=model_inputs["end_d"],
                use_formatter_after_selection=model_inputs["use_formatter_after_selection"],
                progress_cb=tick,
                enable_web=bool(web_inputs["web_enable"]),
                web_model=web_inputs["web_model"].strip(),
                level1_query_tpl=web_inputs["level1_tpl"].strip() or "{topic}",
                do_level2=bool(web_inputs["do_level2"]),
                do_level3=bool(web_inputs["do_level3"]),
                web_max_results_each=10,
                strict_url_filter=bool(web_inputs["strict_url_filter"]),
            ):
                draft_questions.append(q)
                draft_rows.append(full_question_to_row(q))

                draft_info_ph.markdown(
                    f"**Draft rows:** {len(draft_rows)} | **Draft type counts:** {count_types(draft_questions)}"
                )
                draft_preview_ph.dataframe(draft_rows[-min(20, len(draft_rows)) :], use_container_width=True)

            if web_inputs["web_enable"]:
                wlog = st.session_state.get("web_log", {}).get(topic)
                if wlog and wlog.get("context"):
                    with web_preview_ph.container():
                        with st.expander(f"Web enrichment preview — {topic}", expanded=False):
                            st.text(wlog["context"])

        draft_csv_text = write_csv(draft_rows)

        final_questions = list(draft_questions)
        log: Dict[str, Any] = {"type_counts_draft": count_types(draft_questions)}

        if model_inputs["do_rebalance"] and final_questions:
            tick("Rebalancing types to ~60/20/20.")
            total = len(final_questions)
            tb, tn, tm = compute_targets(total)
            msgs, hint = prompt_rebalance(final_questions, tb, tn, tm, model_inputs["end_d"])
            raw = call_json_strict(cfg, cfg.light_model, msgs, schema_hint=hint)
            edited = raw.get("questions", []) or []

            tmp: List[FullQuestion] = []
            for idx, qd in enumerate(edited, start=1):
                fq = validate_or_reformat(
                    cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Rebalance FullQuestion #{idx}", max_attempts=3
                )
                tmp.append(normalize_full_question_fields(fq))
            final_questions = tmp
            log["rebalance_change_log"] = raw.get("change_log", [])
            log["type_counts_after_rebalance"] = count_types(final_questions)

            if model_inputs["use_formatter_after_rebalance"]:
                tick("Canonicalizing post-rebalance list (formatter).")
                msgs, hint2 = prompt_canonicalize_questions_list(final_questions)
                canon = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint2, retries=1)
                canon_list = canon.get("questions", []) or []
                if canon_list:
                    tmp2: List[FullQuestion] = []
                    for idx, qd in enumerate(canon_list, start=1):
                        fq = validate_or_reformat(
                            cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Post-rebalance canonical FullQuestion #{idx}", max_attempts=3
                        )
                        tmp2.append(normalize_full_question_fields(fq))
                    final_questions = tmp2

        if model_inputs["do_final_clean"] and final_questions:
            tick("Final clean pass (formatter).")
            msgs, hint = prompt_clean_questions(final_questions)
            raw = call_json_strict(cfg, cfg.formatter_model, msgs, schema_hint=hint, retries=1)
            cleaned = raw.get("questions", []) or []
            tmp3: List[FullQuestion] = []
            for idx, qd in enumerate(cleaned, start=1):
                fq = validate_or_reformat(
                    cfg, qd, FullQuestion, """FULL_QUESTION_OBJECT""", f"Final clean FullQuestion #{idx}", max_attempts=3
                )
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
