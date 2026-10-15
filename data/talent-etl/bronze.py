"""
Bronze layer: land source data in the lake exactly as received, write-once,
and prove it matches the source.

Layout::

    bronze/postgres/<table>/ingest_date=D/
        <table>.csv          byte-identical COPY export from PostgreSQL
        part-000.parquet     same rows, all columns as text, + audit columns
        _manifest.json       source count, bytes, md5, checks  (written last = commit marker)

    bronze/lms/learning_activity/ingest_date=D/year=Y/month=M/day=DD/
        events-000__<etag>.jsonl     byte-identical server-side copy of the landing file
        events-000__<etag>.parquet   one row per raw line (malformed lines included)
    bronze/lms/learning_activity/ingest_date=D/_manifest.json   one entry per file loaded that day

Rules enforced here:

* **Write-once.** A PostgreSQL partition with a manifest is never rewritten
  unless ``force=True``. Landing files are copied only when new or changed
  (ETag watermark in ``_state/``); a changed file lands as a *new* object
  named by its ETag, so earlier versions are never overwritten.
* **Reconciled to source.** Each load checks row counts and byte sizes against
  the source and fails the step (``BronzeIntegrityError``) on any mismatch.
* **No business logic.** Nothing is parsed, typed, filtered or deduplicated.
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from config import Settings
from lineage import RunContext
from sources import connect, export_table
from storage import Lake

log = logging.getLogger(__name__)

EVENTS_DATASET = "lms/learning_activity"
EVENTS_STATE_KEY = "learning_activity_ingested.json"
MANIFEST = "_manifest.json"


class BronzeIntegrityError(Exception):
    """Raised when what landed in Bronze does not match what the source provided."""


# --------------------------------------------------------------------------- pure helpers
def md5_hex(data: bytes) -> str:
    """Hex MD5 of ``data`` (matches the S3 ETag of a single-part, SSE-S3 object)."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def check(checks: dict[str, dict], name: str, expected: object, actual: object) -> None:
    """Record one source-vs-Bronze comparison."""
    checks[name] = {"expected": expected, "actual": actual, "ok": expected == actual}


def assert_checks(checks: dict[str, dict], what: str) -> None:
    """Raise if any recorded check failed."""
    failed = {k: v for k, v in checks.items() if not v["ok"]}
    if failed:
        raise BronzeIntegrityError(f"{what}: source/Bronze mismatch {failed}")


def count_lines(body: bytes) -> tuple[int, int]:
    """Return (non-blank lines, blank lines) of a text file."""
    lines = body.decode("utf-8", errors="replace").splitlines()
    non_blank = sum(1 for line in lines if line.strip())
    return non_blank, len(lines) - non_blank


def event_object_stem(relative_key: str, etag: str) -> str:
    """``year=…/day=…/events-000.jsonl`` + ETag -> ``year=…/day=…/events-000__<etag12>`` (version-unique)."""
    short_etag = etag.strip('"')[:12]  # kept outside the f-string: Python 3.11 (EKS image) can't nest quotes
    return f"{relative_key.removesuffix('.jsonl')}__{short_etag}"


def audit_columns(df: pd.DataFrame, source: str, run_id: str) -> pd.DataFrame:
    """Append the audit columns every Bronze row carries."""
    out = df.copy()
    out["_source"] = source
    out["_ingested_at"] = datetime.now(timezone.utc).isoformat()
    out["_run_id"] = run_id
    return out


def lines_frame(source_uri: str, etag: str, body: bytes, run_id: str) -> pd.DataFrame:
    """One Bronze row per non-blank raw line of a landing file, malformed lines included."""
    lines = body.decode("utf-8", errors="replace").splitlines()
    rows = [(i, line) for i, line in enumerate(lines, start=1) if line.strip()]
    df = pd.DataFrame(rows, columns=["_line_number", "raw_line"])
    df["_source_key"] = source_uri
    df["_source_etag"] = etag.strip('"')
    return audit_columns(df, "lms", run_id)


# --------------------------------------------------------------------------- PostgreSQL
def ingest_postgres(settings: Settings, lake: Lake, ctx: RunContext, force: bool = False) -> None:
    """Snapshot every HR table into today's write-once Bronze partition."""
    ingest_date = settings.run_date.isoformat()
    with connect(settings, snapshot=True) as conn:
        for table in settings.source_tables:
            with ctx.step("bronze", f"postgres/{table}") as rec:
                source = (f"postgresql://{settings.postgres_host}/{settings.postgres_name}"
                          f"/{settings.source_schema}.{table}")
                prefix = f"{settings.bronze_prefix}postgres/{table}/ingest_date={ingest_date}/"
                rec.inputs, rec.outputs = [source], [lake.uri(prefix)]

                if not force and lake.exists(prefix + MANIFEST):
                    rec.details["skipped"] = "partition already loaded (write-once); use --force-bronze to reload"
                    log.info("bronze postgres/%s ingest_date=%s already loaded - skipped", table, ingest_date)
                    continue

                export = export_table(conn, settings.source_schema, table)
                lake.delete_prefix(prefix)  # clears a partial earlier attempt, or the old copy on --force
                raw_key, parquet_key = f"{prefix}{table}.csv", f"{prefix}part-000.parquet"
                lake.put_bytes(raw_key, export.csv_bytes, "text/csv")
                lake.write_parquet(parquet_key, audit_columns(export.frame, source, ctx.run_id))

                checks: dict[str, dict] = {}
                check(checks, "rows_match_source_count", export.source_row_count, len(export.frame))
                check(checks, "raw_bytes_match_export", len(export.csv_bytes), lake.head(raw_key)["ContentLength"])
                check(checks, "parquet_rows_match_source_count", export.source_row_count,
                      len(lake.read_parquet(parquet_key)))
                assert_checks(checks, f"postgres/{table}")

                lake.put_json(prefix + MANIFEST, {
                    "dataset": f"postgres/{table}", "ingest_date": ingest_date, "run_id": ctx.run_id,
                    "source": source, "source_row_count": export.source_row_count,
                    "raw_export": {"uri": lake.uri(raw_key), "bytes": len(export.csv_bytes),
                                   "md5": md5_hex(export.csv_bytes)},
                    "parquet": {"uri": lake.uri(parquet_key), "rows": len(export.frame)},
                    "columns": list(export.frame.columns),
                    "checks": checks,
                    "committed_at": datetime.now(timezone.utc).isoformat(),
                })
                rec.rows_read = export.source_row_count
                rec.rows_written = len(export.frame)
                rec.details["checks"] = {k: v["ok"] for k, v in checks.items()}
                if export.source_row_count == 0:
                    log.warning("bronze postgres/%s: source returned zero rows", table)


# --------------------------------------------------------------------------- LMS events
def ingest_events(settings: Settings, lake: Lake, ctx: RunContext, force: bool = False) -> None:
    """Copy new or changed landing JSONL files into today's Bronze partition, verifying each one."""
    ingest_date = settings.run_date.isoformat()
    base = f"{settings.bronze_prefix}{EVENTS_DATASET}/ingest_date={ingest_date}/"

    with ctx.step("bronze", EVENTS_DATASET) as rec:
        state_key = f"{settings.state_prefix}{EVENTS_STATE_KEY}"
        state: dict[str, str] = lake.get_json(state_key, default={}) or {}
        landing = [o for o in lake.list_objects(settings.landing_prefix) if o["Key"].endswith(".jsonl")]
        todo = landing if force else [o for o in landing if state.get(o["Key"]) != o["ETag"]]
        changed = [o for o in todo if o["Key"] in state and state[o["Key"]] != o["ETag"]]
        if not landing:
            log.warning("bronze %s: no landing files under %s", EVENTS_DATASET, lake.uri(settings.landing_prefix))

        bodies = lake.get_many_bytes(o["Key"] for o in todo)

        def load(obj: dict) -> dict:
            body = bodies[obj["Key"]]
            stem = event_object_stem(obj["Key"][len(settings.landing_prefix):], obj["ETag"])
            raw_key, parquet_key = f"{base}{stem}.jsonl", f"{base}{stem}.parquet"
            source_uri = lake.uri(obj["Key"])

            lake.copy_object(obj["Key"], raw_key)
            frame = lines_frame(source_uri, obj["ETag"], body, ctx.run_id)
            lake.write_parquet(parquet_key, frame)

            non_blank, blank = count_lines(body)
            copy_head = lake.head(raw_key)
            checks: dict[str, dict] = {}
            check(checks, "downloaded_bytes_match_source", obj["Size"], len(body))
            check(checks, "raw_copy_bytes_match_source", obj["Size"], copy_head["ContentLength"])
            check(checks, "rows_match_source_lines", non_blank, len(frame))
            source_md5, source_etag = md5_hex(body), obj["ETag"].strip('"')
            if source_etag == source_md5:  # ETag is a plain MD5 (single-part, SSE-S3): compare content hashes
                check(checks, "raw_copy_md5_matches_source", source_md5, copy_head["ETag"].strip('"'))
            assert_checks(checks, obj["Key"])
            return {
                "source": source_uri, "source_etag": source_etag, "source_bytes": obj["Size"],
                "source_md5": source_md5, "source_lines": non_blank, "blank_lines": blank,
                "raw_copy": lake.uri(raw_key), "parquet": lake.uri(parquet_key), "bronze_rows": len(frame),
                "checks_passed": sorted(checks), "run_id": ctx.run_id,
            }

        with ThreadPoolExecutor(max_workers=16) as pool:
            entries = list(pool.map(load, todo))

        # Merge today's manifest (several runs may land files on the same day), then commit the watermark.
        manifest_key = base + MANIFEST
        manifest = lake.get_json(manifest_key, default=None) or {
            "dataset": EVENTS_DATASET, "ingest_date": ingest_date, "files": {}, "runs": []}
        for entry in entries:
            manifest["files"][f"{entry['source']}#{entry['source_etag']}"] = entry
        if entries:
            manifest["runs"].append({"run_id": ctx.run_id, "files": len(entries), "forced": force})
            manifest["totals"] = {
                "files": len(manifest["files"]),
                "source_lines": sum(e["source_lines"] for e in manifest["files"].values()),
                "bronze_rows": sum(e["bronze_rows"] for e in manifest["files"].values()),
                "source_bytes": sum(e["source_bytes"] for e in manifest["files"].values()),
            }
            lake.put_json(manifest_key, manifest)
        for obj in todo:
            state[obj["Key"]] = obj["ETag"]
        lake.put_json(state_key, state)

        rows = sum(e["bronze_rows"] for e in entries)
        rec.inputs = [lake.uri(settings.landing_prefix)]
        rec.outputs = [lake.uri(base)] if entries else []
        rec.rows_read = sum(e["source_lines"] for e in entries)
        rec.rows_written = rows
        rec.details.update(
            landing_files=len(landing), loaded_files=len(entries), new_files=len(entries) - len(changed),
            changed_files=len(changed), skipped_unchanged=len(landing) - len(todo), forced=force,
            bytes_copied=sum(e["source_bytes"] for e in entries),
        )
