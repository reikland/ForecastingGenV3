import csv
import io
import json
from typing import Any, Dict, List, Tuple

from forecasting_app.constants import CSV_COLUMNS
from forecasting_app.schemas import FullQuestion


def read_topics_csv(uploaded_file) -> Tuple[List[str], List[Dict[str, Any]]]:
    data = uploaded_file.getvalue()
    text = data.decode("utf-8", errors="replace")
    sio = io.StringIO(text)
    reader = csv.DictReader(sio)
    if not reader.fieldnames:
        raise ValueError("CSV appears to have no header row.")
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has a header but no data rows.")
    return reader.fieldnames, rows


def pick_topic_column(fieldnames: List[str], rows: List[Dict[str, Any]]) -> str:
    sample = rows[:50]
    best_col, best_score = fieldnames[0], -1
    for col in fieldnames:
        score = sum(1 for r in sample if str(r.get(col, "")).strip())
        if score > best_score:
            best_col, best_score = col, score
    return best_col


def full_question_to_row(q: FullQuestion) -> Dict[str, Any]:
    row: Dict[str, Any] = {k: "" for k in CSV_COLUMNS}
    row["title"] = q.title
    row["type"] = q.type
    row["resolution_criteria"] = q.resolution_criteria
    row["fine_print"] = q.fine_print
    row["description"] = q.description
    row["question_weight"] = q.question_weight
    row["open_time"] = q.open_time
    row["scheduled_close_time"] = q.scheduled_close_time
    row["scheduled_resolve_time"] = q.scheduled_resolve_time

    row["range_min"] = "" if q.range_min is None else q.range_min
    row["range_max"] = "" if q.range_max is None else q.range_max
    row["zero_point"] = "" if q.zero_point is None else q.zero_point
    row["open_lower_bound"] = "" if q.open_lower_bound is None else q.open_lower_bound
    row["open_upper_bound"] = "" if q.open_upper_bound is None else q.open_upper_bound
    row["unit"] = "" if q.unit is None else q.unit
    row["group_variable"] = "" if q.group_variable is None else q.group_variable

    row["options"] = "" if not q.options else json.dumps(q.options, ensure_ascii=False)
    row["categories"] = json.dumps(q.categories, ensure_ascii=False)

    row["ai_rating"] = q.ai_rating
    row["ai_rationale"] = q.ai_rationale
    return row


def write_csv(rows: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return output.getvalue()
