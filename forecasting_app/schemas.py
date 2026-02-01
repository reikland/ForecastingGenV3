from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from forecasting_app.constants import ALLOWED_RATINGS, ALLOWED_TYPES, ALLOWED_VERIFY_STATUS

try:
    from pydantic import field_validator  # pydantic v2
    PYDANTIC_V2 = True
except Exception:  # pragma: no cover
    from pydantic import validator  # type: ignore
    PYDANTIC_V2 = False

    def field_validator(*fields, **kwargs):  # type: ignore
        def deco(fn):
            return validator(*fields, **kwargs, allow_reuse=True)(fn)

        return deco


class ProtoQuestion(BaseModel):
    title: str
    suggested_type: str = Field(description="binary | numeric | multiple_choice")
    description: str
    why_informative: str
    candidate_sources: List[str] = Field(default_factory=list)

    @field_validator("suggested_type")
    def _type_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_TYPES:
            raise ValueError(f"suggested_type must be one of {sorted(ALLOWED_TYPES)}")
        return v2


class FullQuestion(BaseModel):
    title: str
    type: str
    resolution_criteria: str
    fine_print: str
    description: str
    question_weight: float

    open_time: str
    scheduled_close_time: str
    scheduled_resolve_time: str

    range_min: Optional[float] = None
    range_max: Optional[float] = None
    zero_point: Optional[float] = None
    open_lower_bound: Optional[float] = None
    open_upper_bound: Optional[float] = None
    unit: Optional[str] = None
    group_variable: Optional[str] = None
    options: Optional[List[str]] = None
    categories: List[str]

    ai_rating: str
    ai_rationale: str

    @field_validator("type")
    def _type_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_TYPES:
            raise ValueError(f"type must be one of {sorted(ALLOWED_TYPES)}")
        return v2

    @field_validator("ai_rating")
    def _rating_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_RATINGS:
            raise ValueError(f"ai_rating must be one of {sorted(ALLOWED_RATINGS)}")
        return v2

    @field_validator("categories")
    def _categories_ok(cls, v: List[str]):  # type: ignore
        vv = [str(c).strip() for c in (v or []) if str(c).strip()]
        if not vv:
            raise ValueError("categories must be a non-empty list")
        return vv

    @field_validator("options")
    def _options_ok(cls, v: Optional[List[str]]):  # type: ignore
        if v is None:
            return None
        vv = [str(o).strip() for o in v if str(o).strip()]
        return vv or None


class VerifyResponse(BaseModel):
    status: str = Field(description="pass | fix")
    issues: List[str] = Field(default_factory=list)
    question: FullQuestion

    @field_validator("status")
    def _status_ok(cls, v: str):  # type: ignore
        v2 = str(v).strip().lower()
        if v2 not in ALLOWED_VERIFY_STATUS:
            raise ValueError("status must be 'pass' or 'fix'")
        return v2


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class WebSearchResponse(BaseModel):
    query: str
    results: List[WebSearchResult] = Field(default_factory=list)
