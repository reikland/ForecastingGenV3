import json
from datetime import date, time as dtime
from typing import Dict, List, Tuple

from forecasting_app.date_utils import iso_dt
from forecasting_app.model_utils import to_model_dict
from forecasting_app.schemas import FullQuestion
from forecasting_app.validation import SYSTEM_FORMATTER

SYSTEM_LIGHT = "Return ONLY a SINGLE valid JSON object. No markdown, no code fences, no prose. Use double quotes."


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
