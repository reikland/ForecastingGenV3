import json
from datetime import date
from typing import Dict, List, Tuple

from forecasting_app.constants import METACULUS_CREDIBLE_SOURCES_URL
from forecasting_app.date_utils import date_window_phrase_inclusive, dt_range_defaults
from forecasting_app.model_utils import to_model_dict
from forecasting_app.schemas import ProtoQuestion
from forecasting_app.validation import SYSTEM_FORMATTER

SYSTEM_PRIMARY = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."


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
