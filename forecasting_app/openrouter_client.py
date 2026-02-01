import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from forecasting_app.json_utils import try_parse_json


@dataclass
class OpenRouterConfig:
    api_key: str
    primary_model: str
    light_model: str
    formatter_model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    use_response_format_json: bool = True
    app_title: str = "Forecast Question Generator"
    http_referer: str = "http://localhost:8501"


class OpenRouterError(RuntimeError):
    pass


def openrouter_chat(
    cfg: OpenRouterConfig,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    plugins: Optional[List[Dict[str, Any]]] = None,
    response_format_override: Optional[Optional[Dict[str, Any]]] = None,
) -> str:
    if not cfg.api_key or len(cfg.api_key.strip()) < 20:
        raise OpenRouterError("OpenRouter API key missing/too short. Paste a valid OpenRouter key.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg.http_referer,
        "X-Title": cfg.app_title,
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(cfg.temperature if temperature_override is None else temperature_override),
        "max_tokens": int(cfg.max_tokens if max_tokens_override is None else max_tokens_override),
    }
    if plugins:
        payload["plugins"] = plugins

    if response_format_override is None:
        if cfg.use_response_format_json:
            payload["response_format"] = {"type": "json_object"}
    else:
        if response_format_override is not None:
            payload["response_format"] = response_format_override

    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    if resp.status_code != 200:
        raise OpenRouterError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:1200]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise OpenRouterError(f"Unexpected response format: {json.dumps(data)[:1500]}")


def repair_to_json(cfg: OpenRouterConfig, raw_text: str, schema_hint: str) -> Any:
    system = (
        "You are a strict JSON repair tool.\n"
        "Return ONLY valid JSON (no markdown, no prose).\n"
        "Use double quotes, no trailing commas.\n"
        "Extract and return the single best JSON object that matches the schema hint."
    )
    user = (
        f"Schema hint:\n{schema_hint}\n\n"
        f"Input text:\n{raw_text}\n\n"
        "Return ONLY the repaired JSON."
    )
    text = openrouter_chat(
        cfg,
        cfg.formatter_model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens_override=min(cfg.max_tokens, 1200),
        temperature_override=0.0,
    )
    return try_parse_json(text)


def call_json_strict(
    cfg: OpenRouterConfig,
    model: str,
    messages: List[Dict[str, str]],
    schema_hint: str,
    *,
    retries: int = 2,
    sleep_s: float = 0.6,
    plugins: Optional[List[Dict[str, Any]]] = None,
    response_format_override: Optional[Optional[Dict[str, Any]]] = None,
) -> Any:
    last_err: Optional[Exception] = None
    for _ in range(retries + 1):
        raw = openrouter_chat(
            cfg,
            model,
            messages,
            plugins=plugins,
            response_format_override=response_format_override,
        )
        try:
            return try_parse_json(raw)
        except Exception as e:
            last_err = e

        try:
            return repair_to_json(cfg, raw_text=raw, schema_hint=schema_hint)
        except Exception as e2:
            last_err = e2

        messages = messages + [
            {
                "role": "user",
                "content": (
                    "INVALID OUTPUT. Return ONLY a SINGLE valid JSON object. "
                    "No markdown. No commentary. Use double quotes. Follow the schema exactly."
                ),
            }
        ]
        time.sleep(sleep_s)

    raise ValueError(f"Model did not return valid JSON after retries. Last error: {last_err}")
