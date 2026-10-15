"""Unit tests for Silver parsing and conforming."""

import json

import pandas as pd

from contracts import CONTRACTS, validate
from silver import conform_events, conform_reviews, parse_event_lines


def _bronze(lines):
    return pd.DataFrame({"raw_line": lines, "_line_number": range(1, len(lines) + 1),
                         "_source_key": "s3://b/landing/f.jsonl"})


GOOD = {"event_id": "E1", "employee_id": "1", "event_date": "2026-01-02T10:00:00",
        "event_type": "enrollment", "course_name": "Py", "text": "hi", "score": None}


def test_malformed_lines_are_separated():
    parsed, malformed = parse_event_lines(_bronze([json.dumps(GOOD), "{not json", "[1, 2]"]))
    assert len(parsed) == 1
    assert len(malformed) == 2
    assert set(malformed["_dq_reason"]) == {"malformed_json"}


def test_json_null_and_absent_keys_stay_null():
    """Regression: NaN from absent keys must not become the string 'nan' (it failed float casting)."""
    no_text = {k: v for k, v in GOOD.items() if k != "text"} | {"event_id": "E2"}
    parsed, _ = parse_event_lines(_bronze([json.dumps(GOOD), json.dumps(no_text)]))
    assert parsed["score"].isna().all()
    assert parsed.loc[parsed["event_id"] == "E2", "text"].isna().all()
    result = validate(parsed, CONTRACTS["learning_activity"], {"employees": {1}})
    assert result.stats["rows_valid"] == 2


def test_numeric_json_values_are_accepted():
    parsed, _ = parse_event_lines(_bronze([json.dumps(GOOD | {"employee_id": 1, "score": 71})]))
    result = validate(parsed, CONTRACTS["learning_activity"], {"employees": {1}})
    assert result.valid["score"].iloc[0] == 71.0


def test_exact_duplicate_events_are_deduplicated():
    parsed, _ = parse_event_lines(_bronze([json.dumps(GOOD)] * 3))
    result = validate(parsed, CONTRACTS["learning_activity"], {"employees": {1}})
    assert len(result.valid) == 1
    assert result.stats["quarantine_reasons"] == {"duplicate_pk": 2}


def test_conform_reviews_derives_period_columns():
    df = pd.DataFrame({"review_period": ["2024-H2", "2025-H1"]})
    out = conform_reviews(df)
    assert out["review_year"].tolist() == [2024, 2025]
    assert out["period_end_date"].tolist() == [pd.Timestamp("2024-12-31"), pd.Timestamp("2025-06-30")]


def test_conform_events_adds_partition_column():
    out = conform_events(pd.DataFrame({"event_date": pd.to_datetime(["2023-10-02 19:08"])}))
    assert out["event_year"].iloc[0] == 2023
    assert out["event_month"].iloc[0] == "2023-10"
