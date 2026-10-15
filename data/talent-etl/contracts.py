"""
Schema contracts and data-quality validation for the Silver layer.

Each dataset declares a :class:`Contract` (columns, types, required fields,
domain rules, keys). :func:`validate` applies it generically and splits the
input into valid rows and quarantined rows, every quarantined row carrying a
``_dq_reason`` that names the rule(s) it broke. Nothing is silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)

# Contract dtypes -> pandas nullable dtypes used in Silver.
_DTYPES = {
    "int": "Int64",
    "float": "Float64",
    "str": "string",
    "date": "datetime64[ns]",
    "datetime": "datetime64[ns]",
}


class SchemaError(Exception):
    """Raised when a source no longer carries a column the contract requires."""


@dataclass(frozen=True)
class Contract:
    """Declarative schema + quality rules for one dataset."""

    name: str
    columns: dict[str, str]
    primary_key: tuple[str, ...]
    required: tuple[str, ...] = ()
    allowed: dict[str, tuple] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    foreign_keys: dict[str, str] = field(default_factory=dict)  # column -> referenced entity
    date_formats: dict[str, str] = field(default_factory=dict)


CONTRACTS: dict[str, Contract] = {
    c.name: c
    for c in [
        Contract(
            name="teams",
            columns={"team_id": "int", "name": "str", "department": "str"},
            primary_key=("team_id",),
            required=("team_id", "name", "department"),
        ),
        Contract(
            name="competencies",
            columns={"competency_id": "int", "name": "str", "category": "str"},
            primary_key=("competency_id",),
            required=("competency_id", "name", "category"),
            allowed={"category": ("Technical", "Leadership", "Domain")},
        ),
        Contract(
            name="employees",
            columns={
                "employee_id": "int",
                "first_name": "str",
                "last_name": "str",
                "email": "str",
                "team_id": "int",
                "job_level": "str",
                "hire_date": "date",
            },
            primary_key=("employee_id",),
            required=("employee_id", "first_name", "last_name", "email", "team_id", "job_level", "hire_date"),
            foreign_keys={"team_id": "teams"},
            date_formats={"hire_date": "%Y-%m-%d"},
        ),
        Contract(
            name="projects",
            columns={
                "project_id": "int",
                "name": "str",
                "team_id": "int",
                "outcome_status": "str",
                "outcome_score": "float",
            },
            primary_key=("project_id",),
            required=("project_id", "name", "team_id", "outcome_status", "outcome_score"),
            allowed={"outcome_status": ("on-time", "delayed", "at-risk", "cancelled")},
            ranges={"outcome_score": (0, 100)},
            foreign_keys={"team_id": "teams"},
        ),
        Contract(
            name="performance_reviews",
            columns={
                "review_id": "int",
                "employee_id": "int",
                "review_period": "str",
                "rating": "int",
                "manager_comment": "str",
                "submitted_date": "date",
            },
            primary_key=("review_id",),
            required=("review_id", "employee_id", "review_period", "rating", "submitted_date"),
            ranges={"rating": (1, 5)},
            foreign_keys={"employee_id": "employees"},
            date_formats={"submitted_date": "%Y-%m-%d"},
        ),
        Contract(
            name="competency_assessments",
            columns={
                "assessment_id": "int",
                "employee_id": "int",
                "competency_id": "int",
                "proficiency_level": "int",
                "required_level": "int",
                "assessed_date": "date",
            },
            primary_key=("assessment_id",),
            required=(
                "assessment_id",
                "employee_id",
                "competency_id",
                "proficiency_level",
                "required_level",
                "assessed_date",
            ),
            ranges={"proficiency_level": (1, 5), "required_level": (1, 5)},
            foreign_keys={"employee_id": "employees", "competency_id": "competencies"},
            date_formats={"assessed_date": "%Y-%m-%d"},
        ),
        Contract(
            name="talent_review",
            columns={
                "employee_id": "int",
                "review_cycle": "str",
                "potential_rating": "str",
                "promotion_readiness": "str",
                "attrition_risk": "str",
            },
            primary_key=("employee_id",),
            required=("employee_id", "review_cycle", "potential_rating", "promotion_readiness", "attrition_risk"),
            allowed={
                "potential_rating": ("Low", "Medium", "High"),
                "promotion_readiness": ("Ready Now", "1-2 Years", "Not Ready"),
                "attrition_risk": ("Low", "Medium", "High"),
            },
            foreign_keys={"employee_id": "employees"},
        ),
        Contract(
            name="learning_activity",
            columns={
                "event_id": "str",
                "employee_id": "int",
                "event_date": "datetime",
                "event_type": "str",
                "course_name": "str",
                "text": "str",
                "score": "float",
            },
            primary_key=("event_id",),
            required=("event_id", "employee_id", "event_date", "event_type", "course_name"),
            allowed={
                "event_type": (
                    "enrollment",
                    "module_started",
                    "module_completed",
                    "quiz_attempted",
                    "quiz_passed",
                    "course_completed",
                    "certification_earned",
                )
            },
            ranges={"score": (0, 100)},
            foreign_keys={"employee_id": "employees"},
            date_formats={"event_date": "ISO8601"},
        ),
    ]
}


@dataclass
class ValidationResult:
    """Outcome of applying a contract to one batch."""

    valid: pd.DataFrame
    quarantined: pd.DataFrame
    stats: dict[str, object]


def _cast(series: pd.Series, kind: str, fmt: str | None) -> pd.Series:
    """Cast a raw column to its contract type; unparseable values become null."""
    if kind in ("date", "datetime"):
        return pd.to_datetime(series, format=fmt or "mixed", errors="coerce")
    if kind in ("int", "float"):
        numeric = pd.to_numeric(series, errors="coerce")
        if kind == "int":
            # Reject 3.7 -> 3 style truncation: non-integral numbers are invalid ints.
            numeric = numeric.where(numeric.isna() | (numeric == numeric.round()))
        return numeric.astype(_DTYPES[kind])
    cleaned = series.astype("string").str.strip()
    return cleaned.mask(cleaned == "")


def validate(
    df: pd.DataFrame,
    contract: Contract,
    reference_keys: dict[str, set] | None = None,
) -> ValidationResult:
    """
    Apply ``contract`` to ``df``.

    Args:
        df: Raw batch (any dtypes; typically strings straight from Bronze).
        contract: Rules to enforce.
        reference_keys: Valid primary keys of referenced entities, e.g.
            ``{"employees": {1, 2, ...}}``, used for foreign-key checks.

    Returns:
        ValidationResult with typed valid rows, quarantined rows (raw values
        plus ``_dq_reason``) and a stats dict for the data-quality report.

    Raises:
        SchemaError: when a contract column is missing from the input entirely.
    """
    reference_keys = reference_keys or {}
    missing_cols = [c for c in contract.columns if c not in df.columns]
    if missing_cols:
        raise SchemaError(f"{contract.name}: missing required column(s) {missing_cols}")

    extra_cols = sorted(c for c in df.columns if c not in contract.columns and not c.startswith("_"))
    if extra_cols:
        log.warning("%s: schema drift - dropping unexpected column(s) %s", contract.name, extra_cols)

    raw = df.reset_index(drop=True)
    typed = pd.DataFrame(index=raw.index)
    reasons = pd.Series([[] for _ in range(len(raw))], index=raw.index, dtype=object)

    def flag(mask: pd.Series, reason: str) -> None:
        for i in mask[mask].index:
            reasons.at[i].append(reason)

    for col, kind in contract.columns.items():
        typed[col] = _cast(raw[col], kind, contract.date_formats.get(col))
        present = raw[col].notna() & (raw[col].astype("string").str.strip() != "")
        flag(present & typed[col].isna(), f"invalid_{kind}:{col}")

    for col in contract.required:
        raw_present = raw[col].notna() & (raw[col].astype("string").str.strip() != "")
        flag(~raw_present, f"missing:{col}")

    for col, values in contract.allowed.items():
        flag(typed[col].notna() & ~typed[col].isin(values), f"out_of_domain:{col}")

    for col, (low, high) in contract.ranges.items():
        flag(typed[col].notna() & ~typed[col].between(low, high), f"out_of_range:{col}")

    for col, entity in contract.foreign_keys.items():
        if entity in reference_keys:
            flag(typed[col].notna() & ~typed[col].isin(reference_keys[entity]), f"orphan_fk:{col}->{entity}")

    is_bad = reasons.map(bool)
    # Duplicate primary keys among otherwise-valid rows: keep the first occurrence.
    dup = ~is_bad & typed.duplicated(subset=list(contract.primary_key), keep="first")
    flag(dup, "duplicate_pk")
    is_bad = reasons.map(bool)

    keep_meta = [c for c in raw.columns if c.startswith("_")]
    valid = pd.concat([typed[~is_bad], raw.loc[~is_bad, keep_meta]], axis=1).reset_index(drop=True)
    quarantined = raw[is_bad].copy()
    quarantined["_dq_reason"] = reasons[is_bad].map(";".join)
    quarantined = quarantined.astype({c: "string" for c in quarantined.columns}).reset_index(drop=True)

    by_reason: dict[str, int] = {}
    for rs in reasons[is_bad]:
        for r in rs:
            rule = r.split(":")[0]
            by_reason[rule] = by_reason.get(rule, 0) + 1

    stats = {
        "dataset": contract.name,
        "rows_in": int(len(raw)),
        "rows_valid": int(len(valid)),
        "rows_quarantined": int(len(quarantined)),
        "quarantine_reasons": dict(sorted(by_reason.items())),
        "dropped_columns": extra_cols,
    }
    log.info(
        "dq dataset=%s rows_in=%d valid=%d quarantined=%d reasons=%s",
        contract.name, stats["rows_in"], stats["rows_valid"], stats["rows_quarantined"], by_reason,
    )
    return ValidationResult(valid=valid, quarantined=quarantined, stats=stats)
