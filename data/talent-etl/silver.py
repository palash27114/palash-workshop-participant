"""
Silver layer: parsed, typed, validated and conformed entities.

For every dataset Silver reads Bronze, applies the dataset's contract
(:mod:`contracts`), writes the valid rows to ``silver/<entity>/`` and the
rejected rows (with ``_dq_reason``) to ``quarantine/<entity>/run_date=<d>/``.

Invariant checked on every run (reconciliation)::

    bronze rows == silver rows + quarantined rows

Silver is a full, deterministic rebuild from Bronze, so re-running it never
duplicates data.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from config import Settings
from contracts import CONTRACTS, validate
from lineage import RunContext
from storage import Lake

log = logging.getLogger(__name__)

# Dependency order: referenced entities are validated before the entities that reference them.
ENTITY_ORDER = (
    "teams",
    "competencies",
    "employees",
    "projects",
    "performance_reviews",
    "competency_assessments",
    "talent_review",
)


class ReconciliationError(Exception):
    """Raised when rows go missing between Bronze and Silver."""


# --------------------------------------------------------------------------- pure transforms
def parse_event_lines(bronze: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse raw JSONL lines.

    Returns:
        (parsed, malformed): ``parsed`` has one column per JSON field plus the
        Bronze lineage columns; ``malformed`` keeps the raw line with
        ``_dq_reason='malformed_json'``.
    """
    meta_cols = [c for c in bronze.columns if c.startswith("_")]
    bronze = bronze.sort_values(["_source_key", "_line_number"]).reset_index(drop=True)
    records, good_idx, bad_idx = [], [], []
    for i, line in bronze["raw_line"].items():
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            obj = None
        if isinstance(obj, dict):
            records.append(obj)
            good_idx.append(i)
        else:
            bad_idx.append(i)

    parsed = pd.DataFrame.from_records(records, index=good_idx)
    # Every value arrives as text into the contract; it does the typing. JSON null and
    # keys absent from a record (NaN after from_records) must stay null, not become "nan".
    parsed = parsed.apply(lambda s: s.map(lambda v: None if v is None or v is pd.NA or v != v else str(v)))
    parsed = pd.concat([parsed, bronze.loc[good_idx, meta_cols]], axis=1).reset_index(drop=True)

    malformed = bronze.loc[bad_idx].copy()
    malformed["_dq_reason"] = "malformed_json"
    malformed = malformed.astype("string").reset_index(drop=True)
    return parsed, malformed


def latest_file_loads(bronze: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Keep only the most recent Bronze load of each landing file.

    Bronze is write-once, so a file that changed upstream (or was re-loaded
    with ``--force-bronze``) has several loads side by side. Each load has one
    ``_ingested_at`` per file; Silver must read exactly one load per file.

    Returns:
        (rows of the latest load per file, number of superseded rows excluded)
    """
    loads = bronze[["_source_key", "_ingested_at"]].drop_duplicates()
    latest = loads.sort_values("_ingested_at").drop_duplicates("_source_key", keep="last")
    current = bronze.merge(latest, on=["_source_key", "_ingested_at"], how="inner")
    return current, len(bronze) - len(current)


def conform_employees(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived attributes used downstream."""
    out = df.copy()
    out["full_name"] = out["first_name"] + " " + out["last_name"]
    out["email"] = out["email"].str.lower()
    return out


def conform_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Split ``YYYY-H1/H2`` into sortable year/half and a period end date."""
    out = df.copy()
    parts = out["review_period"].str.extract(r"^(\d{4})-H([12])$")
    out["review_year"] = pd.to_numeric(parts[0]).astype("Int64")
    out["review_half"] = pd.to_numeric(parts[1]).astype("Int64")
    out["period_end_date"] = pd.to_datetime(
        out["review_year"].astype("string") + out["review_half"].map({1: "-06-30", 2: "-12-31"}).astype("string"),
        errors="coerce",
    )
    return out


def conform_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add partition and calendar columns to learning events."""
    out = df.copy()
    out["event_day"] = out["event_date"].dt.normalize()
    out["event_year"] = out["event_date"].dt.year.astype("Int64")
    out["event_month"] = out["event_date"].dt.strftime("%Y-%m")
    return out


CONFORMERS = {
    "employees": conform_employees,
    "performance_reviews": conform_reviews,
    "learning_activity": conform_events,
}


# --------------------------------------------------------------------------- IO orchestration
def _latest_bronze_partition(lake: Lake, settings: Settings, table: str) -> str:
    """Most recent *committed* ``ingest_date=`` partition (has a manifest) on or before the run date."""
    base = f"{settings.bronze_prefix}postgres/{table}/"
    cutoff = f"{base}ingest_date={settings.run_date.isoformat()}/"
    partitions = [p for p in lake.list_partitions(base) if p <= cutoff and lake.exists(p + "_manifest.json")]
    if not partitions:
        raise FileNotFoundError(f"No committed Bronze snapshot for {table} at {lake.uri(base)}")
    return partitions[-1]


def _publish(
    lake: Lake,
    settings: Settings,
    ctx: RunContext,
    name: str,
    bronze_df: pd.DataFrame,
    valid: pd.DataFrame,
    quarantined: pd.DataFrame,
    stats: dict,
    rec,
    partition_col: str | None = None,
) -> None:
    """Reconcile counts, then write Silver + quarantine and fill the lineage record."""
    if len(bronze_df) != len(valid) + len(quarantined):
        raise ReconciliationError(
            f"{name}: bronze={len(bronze_df)} != silver={len(valid)} + quarantine={len(quarantined)}"
        )
    valid = valid.copy()
    valid["_silver_run_id"] = ctx.run_id
    rec.outputs = lake.replace_dataset(f"{settings.silver_prefix}{name}/", valid, partition_col)
    q_prefix = f"{settings.quarantine_prefix}{name}/run_date={settings.run_date.isoformat()}/"
    lake.delete_prefix(q_prefix)
    if not quarantined.empty:
        rec.outputs.append(lake.write_parquet(f"{q_prefix}part-000.parquet", quarantined))
    rec.rows_read = len(bronze_df)
    rec.rows_written = len(valid)
    rec.rows_quarantined = len(quarantined)
    rec.details["dq"] = stats
    if valid.empty:
        log.warning("silver %s: zero valid rows written", name)


def build_silver(settings: Settings, lake: Lake, ctx: RunContext) -> None:
    """Validate every entity from Bronze into Silver."""
    reference_keys: dict[str, set] = {}

    for name in ENTITY_ORDER:
        with ctx.step("silver", name) as rec:
            partition = _latest_bronze_partition(lake, settings, name)
            bronze_df = lake.read_parquet(partition).rename(columns={"_run_id": "_bronze_run_id"})
            rec.inputs = [lake.uri(partition)]
            result = validate(bronze_df, CONTRACTS[name], reference_keys)
            valid = CONFORMERS.get(name, lambda d: d)(result.valid)
            reference_keys[name] = set(valid[CONTRACTS[name].primary_key[0]].dropna())
            _publish(lake, settings, ctx, name, bronze_df, valid, result.quarantined, result.stats, rec)

    with ctx.step("silver", "learning_activity") as rec:
        prefix = f"{settings.bronze_prefix}lms/learning_activity/"
        all_loads = lake.read_parquet(prefix).rename(columns={"_run_id": "_bronze_run_id"})
        bronze_df, superseded = latest_file_loads(all_loads)
        rec.inputs = [lake.uri(prefix)]
        rec.details["superseded_bronze_rows"] = superseded
        parsed, malformed = parse_event_lines(bronze_df)
        result = validate(parsed, CONTRACTS["learning_activity"], reference_keys)
        valid = conform_events(result.valid).drop(columns=["raw_line"], errors="ignore")
        quarantined = pd.concat([malformed, result.quarantined], ignore_index=True)
        stats = dict(result.stats)
        stats["rows_in"] = len(bronze_df)
        stats["rows_quarantined"] = len(quarantined)
        stats["quarantine_reasons"] = {**stats["quarantine_reasons"], "malformed_json": len(malformed)}
        _publish(lake, settings, ctx, "learning_activity", bronze_df, valid, quarantined, stats, rec,
                 partition_col="event_year")
