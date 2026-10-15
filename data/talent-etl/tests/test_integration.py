"""
End-to-end test against real local services (LocalStack S3 + PostgreSQL).

Uses a throwaway lake bucket so it never touches your working lake. Skipped
automatically when LocalStack or PostgreSQL is not reachable, or when the
source tables have not been seeded (run ``python seed_sources.py`` first).
"""

from __future__ import annotations

import uuid

import pytest

import job
from config import load_settings
from storage import Lake

pytestmark = pytest.mark.integration


@pytest.fixture
def scratch_lake(monkeypatch):
    """Point the pipeline at a unique bucket, copying the landing zone into it."""
    try:
        import psycopg
        settings = load_settings()
        with psycopg.connect(settings.postgres_conninfo) as conn:
            conn.execute(f"SELECT 1 FROM {settings.source_schema}.employees LIMIT 1")
        source = Lake(settings)
        landing = source.list_keys(settings.landing_prefix)
    except Exception as exc:  # noqa: BLE001 - any failure means "environment not ready"
        pytest.skip(f"local services not available: {exc}")
    if not landing:
        pytest.skip("landing zone empty - run seed_sources.py")

    bucket = f"talent-etl-test-{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("LAKE_BUCKET", bucket)
    monkeypatch.setenv("RUN_DATE", "2026-10-15")
    lake = Lake(load_settings())
    lake.ensure_bucket()
    for key, body in source.get_many_bytes(landing).items():
        lake.put_bytes(key, body)
    yield lake
    lake.delete_prefix("")
    lake.s3.delete_bucket(Bucket=bucket)


def _steps(lake):
    return {(s["layer"], s["dataset"]): s for s in lake.get_json("_lineage/latest.json")["steps"]}


def test_pipeline_end_to_end_reconciles_and_is_idempotent(scratch_lake):
    assert job.main([]) == 0
    first = _steps(scratch_lake)
    manifest = scratch_lake.get_json("_lineage/latest.json")
    assert manifest["status"] == "succeeded"

    # Bronze -> Silver: nothing lost.
    events = first[("silver", "learning_activity")]
    assert events["rows_read"] == events["rows_written"] + events["rows_quarantined"]
    assert events["rows_quarantined"] > 0  # the sample dataset contains planted bad records

    # Silver -> Gold: employee-grain tables have one row per employee.
    employees = first[("silver", "employees")]["rows_written"]
    assert first[("gold", "employee_performance_summary")]["rows_written"] == employees

    # Second run, same RUN_DATE: identical outputs, no new bronze files.
    keys_before = scratch_lake.list_keys("")
    assert job.main([]) == 0
    second = _steps(scratch_lake)
    assert second[("bronze", "lms/learning_activity")]["details"]["new_or_changed_files"] == 0
    for k in first:
        if k == ("bronze", "lms/learning_activity"):
            continue  # incremental: unchanged files are skipped, so it legitimately writes 0 rows
        assert first[k]["rows_written"] == second[k]["rows_written"], k
    data_keys = lambda keys: {k for k in keys if not k.startswith("_lineage/")}  # noqa: E731
    assert data_keys(scratch_lake.list_keys("")) == data_keys(keys_before)
