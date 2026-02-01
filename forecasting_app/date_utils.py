from datetime import date, datetime, time as dtime, timedelta
from typing import Tuple

from forecasting_app.constants import PARIS_TZ


def iso_dt(d: date, t: dtime = dtime(0, 0)) -> str:
    dt = datetime.combine(d, t).replace(tzinfo=PARIS_TZ)
    return dt.isoformat()


def dt_range_defaults(start_d: date, end_d: date) -> Tuple[str, str, str]:
    open_time = iso_dt(start_d, dtime(0, 0))
    close_time = iso_dt(end_d, dtime(23, 59))
    resolve_time = iso_dt(end_d, dtime(23, 59))
    return open_time, close_time, resolve_time


def date_window_phrase_inclusive(start_d: date, end_d: date) -> str:
    """
    Produces an unambiguous inclusive window phrase without 'between'/'by'.
    Convention: [start_d, end_d] inclusive is described as:
      "after (start_d - 1 day) and on or before end_d"
    """
    s0 = (start_d - timedelta(days=1)).isoformat()
    e = end_d.isoformat()
    return f"after {s0} and on or before {e}"
