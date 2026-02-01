import streamlit as st

from forecasting_app.openrouter_client import OpenRouterConfig
from forecasting_app.ui_inputs import load_topics, render_model_inputs, render_web_inputs, setup_page
from forecasting_app.ui_pipeline import run_generation_pipeline


def run_app() -> None:
    setup_page()

    model_inputs = render_model_inputs()
    web_inputs = render_web_inputs()

    topics, _rows = load_topics(model_inputs["uploaded"])

    run = st.button(
        "Generate CSV",
        type="primary",
        disabled=not (model_inputs["api_key"] and topics and model_inputs["start_d"] and model_inputs["end_d"]),
    )

    if not run:
        return

    if model_inputs["end_d"] < model_inputs["start_d"]:
        st.error("End date must be >= start date.")
        st.stop()
    if int(model_inputs["k"]) > int(model_inputs["n"]):
        st.error("k must be <= n.")
        st.stop()

    cfg = OpenRouterConfig(
        api_key=model_inputs["api_key"],
        primary_model=model_inputs["primary_model"].strip(),
        light_model=model_inputs["light_model"].strip(),
        formatter_model=model_inputs["formatter_model"].strip(),
        temperature=float(model_inputs["temperature"]),
        max_tokens=int(model_inputs["max_tokens"]),
        use_response_format_json=bool(model_inputs["use_rf"]),
    )

    status_box = st.empty()
    progress = st.progress(0)

    run_generation_pipeline(
        cfg,
        model_inputs,
        web_inputs,
        topics,
        status_box=status_box,
        progress=progress,
        draft_info_ph=st.empty(),
        draft_preview_ph=st.empty(),
        final_info_ph=st.empty(),
        final_preview_ph=st.empty(),
        web_preview_ph=st.empty(),
    )
