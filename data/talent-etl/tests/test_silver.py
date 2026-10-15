"""Unit tests for Silver parsing and conforming."""

import json

import pandas as pd

import pytest

from contracts import CONTRACTS, validate
from silver import conform, conform_events, conform_reviews, parse_event_lines, quarantine_samples, silver_checks


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


def _valid_teams():
    raw = pd.DataFrame([{"team_id": "1", "name": "A", "department": "Sales"},
                        {"team_id": "2", "name": "B", "department": "HR"}])
    return validate(raw, CONTRACTS["teams"]).valid


def test_silver_checks_pass_on_validated_output():
    assert all(silver_checks(_valid_teams(), CONTRACTS["teams"]).values())


@pytest.mark.parametrize("breakage,failed_check", [
    (lambda d: pd.concat([d, d.iloc[[0]]]), "primary_key_unique"),
    (lambda d: d.assign(team_id=pd.array([1, None], "Int64")), "primary_key_not_null"),
    (lambda d: d.assign(department=pd.array(["Sales", None], "string")), "required_fields_not_null"),
    (lambda d: d.assign(team_id=d["team_id"].astype("string")), "types_match_contract"),
])
def test_silver_checks_catch_broken_output(breakage, failed_check):
    checks = silver_checks(breakage(_valid_teams()), CONTRACTS["teams"])
    assert not checks[failed_check]


def test_conform_orders_contract_then_derived_then_audit_columns():
    raw = pd.DataFrame([{"employee_id": "1", "first_name": "Ann", "last_name": "Lee", "email": "A@X.IO",
                         "team_id": "1", "job_level": "IC1", "hire_date": "2020-01-01", "_source": "pg"}])
    out = conform("employees", validate(raw, CONTRACTS["employees"], {"teams": {1}}).valid)
    assert list(out.columns) == list(CONTRACTS["employees"].columns) + ["full_name", "_source"]
    assert out["email"].iloc[0] == "a@x.io"


def test_quarantine_samples_groups_by_first_reason():
    q = pd.DataFrame({"event_id": ["E1", "E2", "E3"], "_dq_reason": ["missing:employee_id",
                      "missing:employee_id;invalid_datetime:event_date", "duplicate_pk"],
                      "_bronze_run_id": ["r", "r", "r"]})
    samples = quarantine_samples(q, limit=1)
    assert set(samples) == {"missing:employee_id", "duplicate_pk"}  # grouped by rule + column
    assert len(samples["missing:employee_id"]) == 1
    assert "_bronze_run_id" not in samples["missing:employee_id"][0]
