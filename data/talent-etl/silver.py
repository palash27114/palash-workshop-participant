"""
Silver layer: parsed, typed, validated and conformed entities.

For every dataset Silver:

1. reads Bronze (the latest committed PostgreSQL snapshot; the latest load of
   each learning-activity file),
2. applies the dataset's contract (:mod:`contracts`): known renamed columns
   are mapped back, values are typed, rules are checked, and failing rows go to
   ``quarantine/<entity>/run_date=<d>/`` with a ``_dq_reason``,
3. conforms the valid rows (derived columns, standard column order),
4. checks the result before publishing (primary key present and unique,
   required fields filled, types exactly as contracted) and reconciles counts::

       bronze rows == silver rows + quarantined rows

5. writes ``silver/<entity>/`` and a per-run data-quality report at
   ``quarantine/_reports/run_date=<d>/dq_report.json``.

Silver is a full, deterministic rebuild from Bronze, so re-running it never
duplicates data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from config import Settings
from contracts import CONTRACTS, Contract, matches_type, order_columns, validate
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
SAMPLES_PER_REASON = 3


class ReconciliationError(Exception):
    """Raised when rows go missing between Bronze and Silver."""


class SilverCheckError(Exception):
    """Raised when validated Silver output breaks a structural guarantee."""


# --------------------------------------------------------------------------- pure transforms
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
    parsed = pd.concat([parsed, bronze.loc[good_idx, meta_cols]], axis=1)
    # The original line travels with the row so a quarantined record shows exactly what arrived.
    parsed["_raw_line"] = bronze.loc[good_idx, "raw_line"]
    parsed = parsed.reset_index(drop=True)

    malformed = bronze.loc[bad_idx].rename(columns={"raw_line": "_raw_line"})
    malformed["_dq_reason"] = "malformed_json"
    malformed = malformed.astype("string").reset_index(drop=True)
    return parsed, malformed


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


def silver_checks(valid: pd.DataFrame, contract: Contract) -> dict[str, bool]:
    """Structural guarantees every published Silver dataset must meet."""
    pk = list(contract.primary_key)
    return {
        "primary_key_not_null": not valid[pk].isna().any().any(),
        "primary_key_unique": not valid.duplicated(pk).any(),
        "required_fields_not_null": not valid[list(contract.required)].isna().any().any(),
        "types_match_contract": all(matches_type(valid[c], kind) for c, kind in contract.columns.items()),
        "renamed_columns_folded_in": not any(c in valid.columns for c in contract.aliases),
    }


def quarantine_samples(quarantined: pd.DataFrame, limit: int = SAMPLES_PER_REASON) -> dict[str, list[dict]]:
    """A few example rows per primary rejection reason, for the human-readable DQ report."""
    if quarantined.empty:
        return {}
    first_reason = quarantined["_dq_reason"].str.split(";").str[0]
    keep = [c for c in quarantined.columns
            if c in ("_raw_line", "_source_key", "_line_number", "_dq_reason") or not c.startswith("_")]
    samples = {}
    for reason, rows in quarantined.assign(_rule=first_reason).groupby("_rule", sort=True):
        head = rows[keep].head(limit).astype(object)
        samples[reason] = head.where(head.notna(), None).to_dict(orient="records")
    return samples


def conform(name: str, valid: pd.DataFrame) -> pd.DataFrame:
    """Apply the dataset's conformer and the standard column order."""
    # The raw line is kept for quarantined rows only; valid Silver rows don't need it.
    out = CONFORMERS.get(name, lambda d: d)(valid).drop(columns=["_raw_line"], errors="ignore")
    return order_columns(out, CONTRACTS[name])


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
    bronze_rows: int,
    valid: pd.DataFrame,
    quarantined: pd.DataFrame,
    stats: dict,
    rec,
    partition_col: str | None = None,
) -> dict:
    """Reconcile and check, then write Silver + quarantine; returns this dataset's DQ report entry."""
    if bronze_rows != len(valid) + len(quarantined):
        raise ReconciliationError(
            f"{name}: bronze={bronze_rows} != silver={len(valid)} + quarantine={len(quarantined)}")
    checks = silver_checks(valid, CONTRACTS[name])
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SilverCheckError(f"{name}: failed Silver checks {failed}")

    valid = valid.assign(_silver_run_id=ctx.run_id)
    rec.outputs = lake.replace_dataset(f"{settings.silver_prefix}{name}/", valid, partition_col)
    q_prefix = f"{settings.quarantine_prefix}{name}/run_date={settings.run_date.isoformat()}/"
    lake.delete_prefix(q_prefix)
    if not quarantined.empty:
        rec.outputs.append(lake.write_parquet(f"{q_prefix}part-000.parquet", quarantined))
    rec.rows_read = bronze_rows
    rec.rows_written = len(valid)
    rec.rows_quarantined = len(quarantined)
    rec.details.update(dq=stats, checks=checks)
    if valid.empty:
        log.warning("silver %s: zero valid rows written", name)
    return {**stats, "checks": checks, "quarantine_samples": quarantine_samples(quarantined)}


def build_silver(settings: Settings, lake: Lake, ctx: RunContext) -> None:
    """Validate every entity from Bronze into Silver and publish the run's DQ report."""
    reference_keys: dict[str, set] = {}
    report: dict[str, dict] = {}

    for name in ENTITY_ORDER:
        with ctx.step("silver", name) as rec:
            partition = _latest_bronze_partition(lake, settings, name)
            bronze_df = lake.read_parquet(partition).rename(columns={"_run_id": "_bronze_run_id"})
            rec.inputs = [lake.uri(partition)]
            result = validate(bronze_df, CONTRACTS[name], reference_keys)
            valid = conform(name, result.valid)
            reference_keys[name] = set(valid[CONTRACTS[name].primary_key[0]].dropna())
            report[name] = _publish(lake, settings, ctx, name, len(bronze_df), valid,
                                    result.quarantined, result.stats, rec)

    with ctx.step("silver", "learning_activity") as rec:
        prefix = f"{settings.bronze_prefix}lms/learning_activity/"
        all_loads = lake.read_parquet(prefix).rename(columns={"_run_id": "_bronze_run_id"})
        bronze_df, superseded = latest_file_loads(all_loads)
        rec.inputs = [lake.uri(prefix)]
        rec.details["superseded_bronze_rows"] = superseded
        parsed, malformed = parse_event_lines(bronze_df)
        result = validate(parsed, CONTRACTS["learning_activity"], reference_keys)
        valid = conform("learning_activity", result.valid)
        quarantined = pd.concat([malformed, result.quarantined], ignore_index=True)
        stats = dict(result.stats)
        stats.update(rows_in=len(bronze_df), rows_quarantined=len(quarantined), superseded_bronze_rows=superseded,
                     quarantine_reasons={**stats["quarantine_reasons"], "malformed_json": len(malformed)})
        report["learning_activity"] = _publish(lake, settings, ctx, "learning_activity", len(bronze_df), valid,
                                               quarantined, stats, rec, partition_col="event_year")

    report_key = f"{settings.quarantine_prefix}_reports/run_date={settings.run_date.isoformat()}/dq_report.json"
    lake.put_json(report_key, {
        "run_id": ctx.run_id,
        "run_date": settings.run_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {n: {k: r[k] for k in ("rows_in", "rows_valid", "rows_quarantined")} for n, r in report.items()},
        "datasets": report,
    })
    log.info("silver dq_report=%s", lake.uri(report_key))
