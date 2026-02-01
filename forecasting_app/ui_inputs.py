from datetime import date
from typing import Any, Dict, List, Tuple

import streamlit as st

from forecasting_app.csv_utils import pick_topic_column, read_topics_csv


def setup_page() -> None:
    st.set_page_config(page_title="Topics -> Forecast Questions CSV", layout="wide")
    st.title("Topics CSV → Forecast Questions → CSV Export (multilevel web + URL filtering + URL injection)")


def render_model_inputs() -> Dict[str, Any]:
    col_a, col_b = st.columns(2)

    with col_a:
        api_key = st.text_input("OpenRouter API key", type="password").strip()
        primary_model = st.text_input("Primary model (generation)", value="openai/gpt-4.1-mini")
        light_model = st.text_input("Light model (verify / rebalance)", value="openai/gpt-4.1-mini")
        formatter_model = st.text_input("Formatter model (schema enforcement / repair)", value="openai/gpt-4.1-nano")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.number_input("Max tokens per call", min_value=500, max_value=8000, value=2000, step=100)
        use_rf = st.checkbox("Use response_format=json_object (if supported)", value=True)

    with col_b:
        uploaded = st.file_uploader("Upload topics CSV", type=["csv"])
        n = st.number_input("n = proto questions per topic", min_value=1, max_value=50, value=6, step=1)
        k = st.number_input("k = keep per topic", min_value=1, max_value=20, value=3, step=1)
        start_d = st.date_input("Questions start date", value=date.today())
        end_d = st.date_input("Questions end date (must resolve by this date)", value=date.today())
        do_rebalance = st.checkbox("Rebalance to ~60% binary / 20% numeric / 20% MCQ", value=True)
        use_formatter_after_selection = st.checkbox("Formatter after selecting k protos (recommended)", value=True)
        use_formatter_after_rebalance = st.checkbox("Formatter after rebalancing (recommended)", value=True)
        do_final_clean = st.checkbox("Final clean pass (formatter)", value=True)

    return {
        "api_key": api_key,
        "primary_model": primary_model,
        "light_model": light_model,
        "formatter_model": formatter_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "use_rf": use_rf,
        "uploaded": uploaded,
        "n": n,
        "k": k,
        "start_d": start_d,
        "end_d": end_d,
        "do_rebalance": do_rebalance,
        "use_formatter_after_selection": use_formatter_after_selection,
        "use_formatter_after_rebalance": use_formatter_after_rebalance,
        "do_final_clean": do_final_clean,
    }


def render_web_inputs() -> Dict[str, Any]:
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

    return {
        "web_enable": web_enable,
        "web_model": web_model,
        "strict_url_filter": strict_url_filter,
        "level1_tpl": level1_tpl,
        "do_level2": do_level2,
        "do_level3": do_level3,
    }


def load_topics(uploaded) -> Tuple[List[str], List[Dict[str, Any]]]:
    topics: List[str] = []
    rows: List[Dict[str, Any]] = []
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
    return topics, rows
