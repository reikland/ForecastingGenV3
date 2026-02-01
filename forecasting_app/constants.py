from zoneinfo import ZoneInfo

PARIS_TZ = ZoneInfo("Europe/Paris")

# Metaculus “credible sources” policy (often sufficient vs enumerating outlets)
METACULUS_CREDIBLE_SOURCES_URL = "https://www.metaculus.com/faq/#definitions"

CSV_COLUMNS = [
    "title",
    "type",
    "resolution_criteria",
    "fine_print",
    "description",
    "question_weight",
    "open_time",
    "scheduled_close_time",
    "scheduled_resolve_time",
    "range_min",
    "range_max",
    "zero_point",
    "open_lower_bound",
    "open_upper_bound",
    "unit",
    "group_variable",
    "options",
    "categories",
    "ai_rating",
    "ai_rationale",
]

ALLOWED_TYPES = {"binary", "numeric", "multiple_choice"}
ALLOWED_RATINGS = {"hard_reject", "soft_reject", "accept_for_aib", "accept_for_main_site"}
ALLOWED_VERIFY_STATUS = {"pass", "fix"}
