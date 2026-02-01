import ast
import json
import re
from typing import Any, Optional


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_first_json_block(s: str) -> Optional[str]:
    s = s.strip()
    start = None
    opener = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None or opener is None:
        return None

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_str = False
    esc = False

    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : j + 1]
    return None


def try_parse_json(s: str) -> Any:
    s0 = strip_code_fences(s)

    try:
        return json.loads(s0)
    except Exception:
        pass

    block = extract_first_json_block(s0)
    if block:
        try:
            return json.loads(block)
        except Exception:
            try:
                obj = ast.literal_eval(block)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    try:
        obj = ast.literal_eval(s0)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    raise ValueError("Could not parse valid JSON from the model output.")
