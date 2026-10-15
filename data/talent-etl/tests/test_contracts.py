"""Unit tests for schema contracts and data-quality validation."""

import pandas as pd
import pytest

from contracts import CONTRACTS, SchemaError, validate


def _reviews(**overrides):
    row = {"review_id": "1", "employee_id": "1", "review_period": "2025-H1", "rating": "3",
           "manager_comment": "ok", "submitted_date": "2025-07-01"}
    row.update(overrides)
    return row


def test_valid_rows_are_typed():
    result = validate(pd.DataFrame([_reviews()]), CONTRACTS["performance_reviews"], {"employees": {1}})
    assert result.stats["rows_valid"] == 1
    assert str(result.valid["rating"].dtype) == "Int64"
    assert result.valid["submitted_date"].iloc[0] == pd.Timestamp("2025-07-01")


@pytest.mark.parametrize("override,reason", [
    ({"rating": "7"}, "out_of_range:rating"),
    ({"rating": "four"}, "invalid_int:rating"),
    ({"rating": "3.5"}, "invalid_int:rating"),
    ({"submitted_date": "yesterday"}, "invalid_date:submitted_date"),
    ({"employee_id": ""}, "missing:employee_id"),
    ({"employee_id": "999"}, "orphan_fk:employee_id->employees"),
])
def test_rule_violations_are_quarantined_with_reason(override, reason):
    result = validate(pd.DataFrame([_reviews(**override)]), CONTRACTS["performance_reviews"], {"employees": {1}})
    assert result.valid.empty
    assert reason in result.quarantined["_dq_reason"].iloc[0]


def test_domain_check():
    df = pd.DataFrame([{"competency_id": "1", "name": "X", "category": "Magic"}])
    result = validate(df, CONTRACTS["competencies"])
    assert result.quarantined["_dq_reason"].iloc[0] == "out_of_domain:category"


def test_duplicate_primary_key_keeps_first_and_quarantines_rest():
    df = pd.DataFrame([_reviews(manager_comment="first"), _reviews(manager_comment="second")])
    result = validate(df, CONTRACTS["performance_reviews"], {"employees": {1}})
    assert list(result.valid["manager_comment"]) == ["first"]
    assert result.quarantined["_dq_reason"].tolist() == ["duplicate_pk"]


def test_nullable_column_may_be_blank():
    result = validate(pd.DataFrame([_reviews(manager_comment="")]), CONTRACTS["performance_reviews"],
                      {"employees": {1}})
    assert result.stats["rows_valid"] == 1
    assert pd.isna(result.valid["manager_comment"].iloc[0])


def test_rows_are_conserved():
    df = pd.DataFrame([_reviews(), _reviews(review_id="2", rating="9"), _reviews(review_id="3")])
    result = validate(df, CONTRACTS["performance_reviews"], {"employees": {1}})
    assert len(result.valid) + len(result.quarantined) == len(df)


def test_missing_contract_column_raises_schema_error():
    df = pd.DataFrame([_reviews()]).drop(columns=["rating"])
    with pytest.raises(SchemaError):
        validate(df, CONTRACTS["performance_reviews"])


def test_unexpected_column_is_dropped_and_reported():
    result = validate(pd.DataFrame([_reviews(surprise="x")]), CONTRACTS["performance_reviews"], {"employees": {1}})
    assert "surprise" not in result.valid.columns
    assert result.stats["dropped_columns"] == ["surprise"]
