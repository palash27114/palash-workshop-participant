"""Unit tests for Bronze helpers and Silver's selection of Bronze loads."""

import hashlib

import pandas as pd
import pytest

from bronze import (
    BronzeIntegrityError,
    assert_checks,
    check,
    count_lines,
    event_object_stem,
    lines_frame,
    md5_hex,
)
from silver import latest_file_loads


def test_count_lines_separates_blank_lines():
    assert count_lines(b'{"a":1}\n\n   \n{"a":2}\n') == (2, 2)


def test_lines_frame_keeps_malformed_lines_and_line_numbers():
    body = b'{"event_id":"E1"}\n\n{not json\n'
    df = lines_frame("s3://b/landing/f.jsonl", '"abc"', body, "run-1")
    assert df["raw_line"].tolist() == ['{"event_id":"E1"}', "{not json"]
    assert df["_line_number"].tolist() == [1, 3]
    assert set(df["_source_etag"]) == {"abc"}
    assert {"_source", "_ingested_at", "_run_id"} <= set(df.columns)


def test_event_object_stem_is_version_unique():
    a = event_object_stem("year=2023/month=10/day=02/events-000.jsonl", '"0123456789abcdef"')
    b = event_object_stem("year=2023/month=10/day=02/events-000.jsonl", '"fedcba9876543210"')
    assert a == "year=2023/month=10/day=02/events-000__0123456789ab"
    assert a != b


def test_md5_matches_hashlib():
    assert md5_hex(b"bronze") == hashlib.md5(b"bronze").hexdigest()


def test_checks_pass_and_fail():
    checks: dict = {}
    check(checks, "rows", 10, 10)
    assert_checks(checks, "ok")
    check(checks, "bytes", 100, 99)
    with pytest.raises(BronzeIntegrityError, match="bytes"):
        assert_checks(checks, "t")


def test_latest_file_loads_keeps_one_load_per_file():
    bronze = pd.DataFrame({
        "_source_key": ["f1", "f1", "f1", "f2"],
        "_ingested_at": ["2026-10-14T10", "2026-10-15T09", "2026-10-15T09", "2026-10-14T10"],
        "raw_line": ["old", "new-1", "new-2", "only"],
    })
    current, superseded = latest_file_loads(bronze)
    assert sorted(current["raw_line"]) == ["new-1", "new-2", "only"]
    assert superseded == 1
