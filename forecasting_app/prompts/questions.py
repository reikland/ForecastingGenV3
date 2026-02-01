import json
from datetime import date, time as dtime
from typing import Dict, List, Tuple

from forecasting_app.constants import ALLOWED_RATINGS, METACULUS_CREDIBLE_SOURCES_URL
from forecasting_app.date_utils import date_window_phrase_inclusive, dt_range_defaults, iso_dt
from forecasting_app.model_utils import to_model_dict
from forecasting_app.schemas import FullQuestion, ProtoQuestion
from forecasting_app.validation import SYSTEM_FORMATTER

SYSTEM_PRIMARY = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."
SYSTEM_LIGHT = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."


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
