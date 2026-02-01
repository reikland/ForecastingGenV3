from typing import Any, Dict, List

from forecasting_app.schemas import FullQuestion, PYDANTIC_V2


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
