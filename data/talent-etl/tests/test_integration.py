"""
End-to-end tests against real local services (LocalStack S3 + PostgreSQL).

Each test uses a throwaway lake bucket so it never touches your working lake.
Skipped automatically when LocalStack or PostgreSQL is not reachable, or when
the source tables have not been seeded (run ``python seed_sources.py`` first).
"""

from __future__ import annotations

import json
import uuid

import pytest

import job
from config import load_settings
from storage import Lake

pytestmark = pytest.mark.integration
EVENTS_STEP = ("bronze", "lms/learning_activity")


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


def _bronze_objects(lake):
    return {o["Key"]: o["ETag"] for o in lake.list_objects("bronze/")}


def test_pipeline_end_to_end_reconciles_and_is_idempotent(scratch_lake):
    assert job.main([]) == 0
    first = _steps(scratch_lake)
    assert scratch_lake.get_json("_lineage/latest.json")["status"] == "succeeded"

    # Bronze reconciled to source: every landing line landed, every HR table check passed.
    assert first[EVENTS_STEP]["rows_read"] == first[EVENTS_STEP]["rows_written"] == 29868
    for table in ("employees", "performance_reviews", "talent_review"):
        assert all(first[("bronze", f"postgres/{table}")]["details"]["checks"].values())

    # Bronze -> Silver: nothing lost.
    events = first[("silver", "learning_activity")]
    assert events["rows_read"] == events["rows_written"] + events["rows_quarantined"]
    assert events["rows_quarantined"] > 0  # the sample dataset contains planted bad records

    # Silver: renamed "notes" values recovered, all structural checks passed, DQ report published.
    assert events["details"]["dq"]["recovered_from_aliases"] == {"notes->text": 39}
    assert events["details"]["dq"]["dropped_columns"] == []
    assert all(events["details"]["checks"].values())
    report = scratch_lake.get_json("quarantine/_reports/run_date=2026-10-15/dq_report.json")
    assert report["summary"]["learning_activity"]["rows_quarantined"] == events["rows_quarantined"]
    assert set(report["datasets"]["learning_activity"]["quarantine_samples"]) >= {"malformed_json", "duplicate_pk"}

    # Silver -> Gold: employee-grain tables have one row per employee.
    employees = first[("silver", "employees")]["rows_written"]
    assert first[("gold", "employee_performance_summary")]["rows_written"] == employees

    # Second run, same RUN_DATE: Bronze is untouched (write-once), Silver/Gold identical.
    bronze_before = _bronze_objects(scratch_lake)
    assert job.main([]) == 0
    second = _steps(scratch_lake)
    assert _bronze_objects(scratch_lake) == bronze_before
    assert second[EVENTS_STEP]["details"]["loaded_files"] == 0
    assert "skipped" in second[("bronze", "postgres/employees")]["details"]
    for k in first:
        if k[0] in ("silver", "gold"):
            assert first[k]["rows_written"] == second[k]["rows_written"], k


def test_bronze_manifests_record_source_bytes_and_counts(scratch_lake):
    assert job.main(["--layers", "bronze"]) == 0
    pg = scratch_lake.get_json("bronze/postgres/employees/ingest_date=2026-10-15/_manifest.json")
    assert pg["source_row_count"] == pg["parquet"]["rows"] == 2000
    raw = scratch_lake.head("bronze/postgres/employees/ingest_date=2026-10-15/employees.csv")
    assert raw["ContentLength"] == pg["raw_export"]["bytes"]

    ev = scratch_lake.get_json("bronze/lms/learning_activity/ingest_date=2026-10-15/_manifest.json")
    assert ev["totals"]["files"] == 783
    assert ev["totals"]["source_lines"] == ev["totals"]["bronze_rows"] == 29868
    entry = next(iter(ev["files"].values()))
    copy_key = entry["raw_copy"].split("/", 3)[3]
    assert scratch_lake.head(copy_key)["ContentLength"] == entry["source_bytes"]


def test_changed_landing_file_is_versioned_not_overwritten(scratch_lake):
    assert job.main([]) == 0
    silver_rows = _steps(scratch_lake)[("silver", "learning_activity")]["rows_written"]
    bronze_before = _bronze_objects(scratch_lake)

    # Upstream re-delivers one file with an extra valid event.
    key = scratch_lake.list_keys("landing/learning_activity/", ".jsonl")[0]
    extra = {"event_id": "EV-TEST-NEW", "employee_id": "1", "event_date": "2026-01-01T09:00:00",
             "event_type": "enrollment", "course_name": "Python Basics", "text": "late delivery", "score": None}
    scratch_lake.put_bytes(key, scratch_lake.get_bytes(key) + (json.dumps(extra) + "\n").encode())

    assert job.main([]) == 0
    steps = _steps(scratch_lake)
    assert steps[EVENTS_STEP]["details"]["changed_files"] == 1
    bronze_after = _bronze_objects(scratch_lake)
    # Old objects are still there, byte-for-byte (same ETag): nothing was overwritten.
    assert all(bronze_after.get(k) == etag for k, etag in bronze_before.items() if "_manifest" not in k)
    assert len(bronze_after) == len(bronze_before) + 2  # new .jsonl copy + new .parquet
    # Silver reads only the latest load of the changed file: exactly one new row, no duplicates.
    silver = steps[("silver", "learning_activity")]
    assert silver["rows_written"] == silver_rows + 1
    assert silver["details"]["superseded_bronze_rows"] > 0


def test_force_bronze_reloads_without_duplicating(scratch_lake):
    assert job.main([]) == 0
    silver_rows = _steps(scratch_lake)[("silver", "learning_activity")]["rows_written"]
    assert job.main(["--force-bronze"]) == 0
    steps = _steps(scratch_lake)
    assert steps[EVENTS_STEP]["details"]["loaded_files"] == 783
    assert "skipped" not in steps[("bronze", "postgres/employees")]["details"]
    assert steps[("silver", "learning_activity")]["rows_written"] == silver_rows


def test_critical_quarantine_rate_keeps_previous_silver_and_stops(scratch_lake, monkeypatch):
    """A batch whose quarantine rate reaches the critical threshold must not replace good Silver data."""
    assert job.main([]) == 0
    before = scratch_lake.read_parquet("silver/learning_activity/")

    # The real batch quarantines ~1% of events; lower the critical threshold so it counts as broken.
    monkeypatch.setenv("QUARANTINE_FAIL_PCT", "0.5")
    assert job.main(["--layers", "silver"]) == 1

    manifest = scratch_lake.get_json("_lineage/latest.json")
    assert manifest["status"] == "failed"
    assert [a["dataset"] for a in manifest["alerts"] if a["level"] == "critical"] == ["learning_activity"]
    report = scratch_lake.get_json("quarantine/_reports/run_date=2026-10-15/dq_report.json")
    assert report["summary"]["learning_activity"]["level"] == "critical"
    after = scratch_lake.read_parquet("silver/learning_activity/")
    assert len(after) == len(before)  # previous good Silver kept
    assert set(after["_silver_run_id"]) == set(before["_silver_run_id"])
