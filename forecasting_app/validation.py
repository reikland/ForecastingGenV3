import json
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from forecasting_app.constants import ALLOWED_RATINGS, ALLOWED_VERIFY_STATUS
from forecasting_app.openrouter_client import OpenRouterConfig, call_json_strict

T = TypeVar("T", bound=BaseModel)

SYSTEM_FORMATTER = (
    "You are a schema enforcer.\n"
    "Return ONLY a SINGLE valid JSON object. No markdown, no prose.\n"
    "Use double quotes. Ensure it parses with json.loads().\n"
    "Fix ONLY what is needed for schema compliance; do not change meaning."
)


def validate_or_reformat(
    cfg: OpenRouterConfig,
    data: Any,
    model_cls: Type[T],
    schema_hint: str,
    context_label: str,
    *,
    max_attempts: int = 3,
) -> T:
    cur = data
    last_err: Optional[Exception] = None

    for _attempt in range(1, max_attempts + 1):
        try:
            return model_cls(**cur)  # type: ignore[arg-type]
        except ValidationError as e:
            last_err = e
            err_txt = str(e)

            user = (
                f"Context: {context_label}\n"
                f"Validation error:\n{err_txt}\n\n"
                f"Required schema hint:\n{schema_hint}\n\n"
                "Important constraints:\n"
                f"- For FullQuestion.ai_rating, allowed: {sorted(ALLOWED_RATINGS)}\n"
                f"- VerifyResponse.status allowed: {sorted(ALLOWED_VERIFY_STATUS)}\n"
                "- Do NOT set ai_rating to 'pass' or 'fix'. Those belong only to VerifyResponse.status.\n"
                "- Return ONLY JSON.\n\n"
                "Input JSON:\n"
                f"{json.dumps(cur, ensure_ascii=False)}"
            )
            cur = call_json_strict(
                cfg,
                cfg.formatter_model,
                [{"role": "system", "content": SYSTEM_FORMATTER}, {"role": "user", "content": user}],
                schema_hint=schema_hint,
                retries=1,
            )

        except Exception as e:
            last_err = e
            break

    raise ValueError(f"Failed to validate after {max_attempts} formatter attempts ({context_label}). Last error: {last_err}")
