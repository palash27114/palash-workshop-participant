"""
Bronze layer: land source data in the lake exactly as received.

* PostgreSQL tables -> ``bronze/postgres/<table>/ingest_date=<d>/part-000.parquet``
  (a full daily snapshot; re-running the same day overwrites only that day).
* Learning-activity JSONL files -> ``bronze/lms/learning_activity/year=/month=/day=/<file>.parquet``
  with one row per *raw line*, including malformed ones. Ingestion is
  incremental: a state file remembers each landing object's ETag, so only new
  or changed files are copied on the next run.

Bronze never parses, filters or deduplicates; that is Silver's job.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from config import Settings
from lineage import RunContext
from sources import connect, read_table_raw, with_retries
from storage import Lake

log = logging.getLogger(__name__)

EVENTS_DATASET = "lms/learning_activity"
EVENTS_STATE_KEY = "learning_activity_ingested.json"


def _metadata(df: pd.DataFrame, source: str, run_id: str) -> pd.DataFrame:
    """Append audit columns every Bronze row carries."""
    out = df.copy()
    out["_source"] = source
    out["_ingested_at"] = datetime.now(timezone.utc).isoformat()
    out["_run_id"] = run_id
    return out


def ingest_postgres(settings: Settings, lake: Lake, ctx: RunContext) -> None:
    """Snapshot every source table into today's Bronze partition."""
    partition = f"ingest_date={settings.run_date.isoformat()}/"
    with connect(settings) as conn:
        for table in settings.source_tables:
            with ctx.step("bronze", f"postgres/{table}") as rec:
                df = with_retries(
                    lambda t=table: read_table_raw(conn, settings.source_schema, t), f"read {table}"
                )
                source = f"postgresql://{settings.postgres_host}/{settings.postgres_name}/{settings.source_schema}.{table}"
                prefix = f"{settings.bronze_prefix}postgres/{table}/{partition}"
                rec.outputs = lake.replace_dataset(prefix, _metadata(df, source, ctx.run_id))
                rec.inputs = [source]
                rec.rows_read = rec.rows_written = len(df)
                if df.empty:
                    log.warning("bronze postgres/%s: source returned zero rows", table)


def _lines_frame(key: str, etag: str, body: bytes, run_id: str, lake: Lake) -> pd.DataFrame:
    """One Bronze row per raw line of a landing file (blank lines skipped)."""
    lines = body.decode("utf-8", errors="replace").splitlines()
    rows = [(i, line) for i, line in enumerate(lines, start=1) if line.strip()]
    df = pd.DataFrame(rows, columns=["_line_number", "raw_line"])
    df["_source_key"] = lake.uri(key)
    df["_source_etag"] = etag
    return _metadata(df, "lms", run_id)


def ingest_events(settings: Settings, lake: Lake, ctx: RunContext) -> None:
    """Incrementally copy new/changed landing JSONL files into Bronze."""
    with ctx.step("bronze", EVENTS_DATASET) as rec:
        state_key = f"{settings.state_prefix}{EVENTS_STATE_KEY}"
        state: dict[str, str] = lake.get_json(state_key, default={}) or {}

        landing = [o for o in lake.list_objects(settings.landing_prefix) if o["Key"].endswith(".jsonl")]
        todo = [o for o in landing if state.get(o["Key"]) != o["ETag"]]
        rec.details.update(landing_files=len(landing), new_or_changed_files=len(todo),
                           skipped_unchanged=len(landing) - len(todo))
        if not landing:
            log.warning("bronze %s: no landing files under %s", EVENTS_DATASET, lake.uri(settings.landing_prefix))

        bodies = lake.get_many_bytes(o["Key"] for o in todo)

        def write(obj: dict) -> tuple[str, int]:
            relative = obj["Key"][len(settings.landing_prefix):]  # year=YYYY/month=MM/day=DD/events-000.jsonl
            target = f"{settings.bronze_prefix}{EVENTS_DATASET}/{relative.removesuffix('.jsonl')}.parquet"
            frame = _lines_frame(obj["Key"], obj["ETag"], bodies[obj["Key"]], ctx.run_id, lake)
            lake.write_parquet(target, frame)
            return target, len(frame)

        with ThreadPoolExecutor(max_workers=16) as pool:
            written = list(pool.map(write, todo))

        for obj in todo:
            state[obj["Key"]] = obj["ETag"]
        lake.put_json(state_key, state)

        rec.inputs = [lake.uri(settings.landing_prefix)]
        rec.outputs = [lake.uri(f"{settings.bronze_prefix}{EVENTS_DATASET}/")]
        rec.rows_read = rec.rows_written = sum(n for _, n in written)
