"""
Data access for the Streamlit dashboard: reads the Gold layer only.

Uses the pipeline's own :class:`storage.Lake` and :func:`config.load_settings`,
so the dashboard follows the same environment variables as the job
(LocalStack locally, Amazon S3 in the cloud). Everything here is plain pandas
so it can be unit-tested without Streamlit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

JOB_DIR = Path(__file__).resolve().parents[1]
if str(JOB_DIR) not in sys.path:
    sys.path.insert(0, str(JOB_DIR))

from config import load_settings  # noqa: E402
from storage import Lake  # noqa: E402

GOLD_TABLES = (
    "review_history",
    "employee_performance_summary",
    "skill_gaps",
    "promotion_pipeline",
    "training_summary",
    "attrition_risk",
    "team_performance_outcomes",
    "skill_distribution",
)
RISK_ORDER = ["High", "Medium", "Low"]
TREND_ORDER = ["Improving", "Stable", "Declining"]


@dataclass
class GoldData:
    """Everything the dashboard shows, loaded once per refresh."""

    tables: dict[str, pd.DataFrame]
    catalog: dict
    summary: dict
    manifest: dict | None
    dq_report: dict | None
    lake_uri: str


def load_gold() -> GoldData:
    """Read every Gold table plus the catalog, summary, last run manifest and DQ report."""
    settings = load_settings()
    lake = Lake(settings)
    tables = {name: lake.read_parquet(f"{settings.gold_prefix}{name}/") for name in GOLD_TABLES}
    catalog = lake.get_json(f"{settings.gold_prefix}_catalog.json", default={}) or {}
    summary = lake.get_json(f"{settings.gold_prefix}_summary.json", default={}) or {}
    manifest = lake.get_json(f"{settings.lineage_prefix}latest.json", default=None)
    as_of = catalog.get("as_of", settings.run_date.isoformat())
    dq_report = lake.get_json(f"{settings.quarantine_prefix}_reports/run_date={as_of}/dq_report.json", default=None)
    return GoldData(tables, catalog, summary, manifest, dq_report, lake.uri(""))


# --------------------------------------------------------------------------- filters
def filter_people(df: pd.DataFrame, departments: list[str], job_levels: list[str]) -> pd.DataFrame:
    """Keep rows in the selected departments / job levels; an empty selection means 'all'."""
    out = df
    if departments and "department" in out.columns:
        out = out[out["department"].isin(departments)]
    if job_levels and "job_level" in out.columns:
        out = out[out["job_level"].isin(job_levels)]
    return out


def filtered(gold: GoldData, departments: list[str], job_levels: list[str]) -> dict[str, pd.DataFrame]:
    """Apply the sidebar filters to every table they make sense for."""
    out = {}
    for name, df in gold.tables.items():
        if name == "skill_distribution":
            # Department rows only; the organisation-wide "All" rows are kept separately by the caller.
            out[name] = df[df["department"].isin(departments)] if departments else df[df["department"] != "All"]
        elif name == "team_performance_outcomes":
            out[name] = filter_people(df, departments, [])
        else:
            out[name] = filter_people(df, departments, job_levels)
    return out


# --------------------------------------------------------------------------- KPIs
def kpis(t: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Headline numbers for the current filter selection."""
    perf, gaps = t["employee_performance_summary"], t["skill_gaps"]
    promo, risk, train = t["promotion_pipeline"], t["attrition_risk"], t["training_summary"]
    n = len(perf)
    return {
        "employees": n,
        "avg_rating": float(perf["latest_rating"].mean()) if n else float("nan"),
        "declining": int((perf["rating_trend"] == "Declining").sum()),
        "with_critical_gap": int(gaps.loc[gaps["is_critical_gap"], "employee_id"].nunique()),
        "ready_now": int((promo["promotion_tier"] == "1 - Ready now").sum()),
        "high_risk": int((risk["computed_risk_tier"] == "High").sum()),
        "no_learning": int((train["enrollments"] == 0).sum()),
        "retention_flags": int(promo["retention_flag"].sum()),
    }


# --------------------------------------------------------------------------- employee 360
def employee_options(perf: pd.DataFrame) -> dict[str, int]:
    """'Name · Team (id)' label -> employee_id, sorted by name."""
    rows = perf.sort_values(["full_name", "employee_id"])
    return {f"{r.full_name} · {r.team_name} ({r.employee_id})": int(r.employee_id) for r in rows.itertuples()}


def employee_360(tables: dict[str, pd.DataFrame], employee_id: int) -> dict[str, pd.DataFrame | pd.Series | None]:
    """Every Gold fact about one employee."""
    def one(name: str) -> pd.Series | None:
        df = tables[name]
        match = df[df["employee_id"] == employee_id]
        return match.iloc[0] if len(match) else None

    history = tables["review_history"]
    gaps = tables["skill_gaps"]
    return {
        "profile": one("employee_performance_summary"),
        "promotion": one("promotion_pipeline"),
        "training": one("training_summary"),
        "risk": one("attrition_risk"),
        "reviews": history[history["employee_id"] == employee_id].sort_values(["review_year", "review_half"]),
        "gaps": gaps[gaps["employee_id"] == employee_id].sort_values("gap", ascending=False),
    }


def delivery_groups(teams: pd.DataFrame) -> pd.DataFrame:
    """Teams with projects grouped by delivery result (projects per team are few, so % on time is 0/50/100)."""
    wp = teams[teams["has_projects"]].copy()
    wp["delivery"] = "Mixed"
    wp.loc[wp["pct_projects_on_time"] == 100, "delivery"] = "All on time"
    wp.loc[wp["pct_projects_on_time"] == 0, "delivery"] = "All delayed"
    grouped = wp.groupby("delivery").agg(
        teams=("team_id", "count"),
        pct_high_performers=("pct_high_performers", "mean"),
        avg_latest_rating=("avg_latest_rating", "mean"),
    )
    return grouped.reindex([g for g in ["All on time", "Mixed", "All delayed"] if g in grouped.index]).reset_index()
