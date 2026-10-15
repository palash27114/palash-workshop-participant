"""
Gold layer: curated, business-ready tables, one per business question.

| Gold table                     | Grain                       | Business question                                   |
|--------------------------------|-----------------------------|-----------------------------------------------------|
| review_history                 | employee x review period    | Q1 ratings & review history                         |
| employee_performance_summary   | employee                    | Q1 ratings & review history (latest/trend)          |
| skill_gaps                     | employee x competency (gap) | Q2 critical skill gaps                              |
| promotion_pipeline             | employee                    | Q3 high-potential employees ready for promotion     |
| training_summary               | employee                    | Q4 training & development completed                 |
| attrition_risk                 | employee                    | Q5 attrition risk from performance trends           |
| team_performance_outcomes      | team                        | Q6 performance vs team & project outcomes           |
| skill_distribution             | competency x department     | Q7 distribution of skills across the organization   |

Every builder is a pure function of Silver DataFrames so it can be unit
tested, and :func:`build_gold` enforces each table's declared grain before
publishing. Business rules (thresholds, scoring) are named constants below so
they are reviewable in one place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd

from config import Settings
from lineage import RunContext
from storage import Lake

log = logging.getLogger(__name__)

# ----------------------------------------------------------------- business rules
HIGH_PERFORMER_MIN_RATING = 4
LOW_PERFORMER_MAX_RATING = 2
CRITICAL_GAP_MIN = 2                 # required - proficiency >= 2 levels ...
KEY_COMPETENCY_MIN_REQUIRED = 4      # ... or any shortfall where the role requires level >= 4
INACTIVE_LEARNING_DAYS = 180
NEGATIVE_COMMENT_PATTERN = r"trended down|missed|struggled|needs support|follow-up plan"
ATTRITION_POINTS = {
    "rating_declined": 2,
    "low_latest_rating": 2,
    "negative_manager_comment": 1,
    "no_recent_learning": 1,
    "manager_flag_high": 2,
    "manager_flag_medium": 1,
}
ATTRITION_TIERS = ((5, "High"), (3, "Medium"), (0, "Low"))  # score >= threshold -> tier


RULES = {
    "high_performer": f"latest rating >= {HIGH_PERFORMER_MIN_RATING}",
    "low_performer": f"latest rating <= {LOW_PERFORMER_MAX_RATING}",
    "critical_skill_gap": (f"latest assessment short of required level by >= {CRITICAL_GAP_MIN}, or short by any "
                           f"amount where the role requires level >= {KEY_COMPETENCY_MIN_REQUIRED}"),
    "high_potential": f"talent review potential = High and latest rating >= {HIGH_PERFORMER_MIN_RATING}",
    "promotion_tiers": {
        "1 - Ready now": "high potential, readiness 'Ready Now', no critical skill gaps",
        "2 - Ready now, close skill gaps": "high potential, readiness 'Ready Now', >= 1 critical gap",
        "3 - Emerging (1-2 years)": "high potential, readiness '1-2 Years'",
        "4 - Not in pipeline": "everyone else",
    },
    "learning_active": f"any learning event in the last {INACTIVE_LEARNING_DAYS} days before the analysis date",
    "quiz_pass_rate": "quiz_passed / (quiz_passed + quiz_attempted); quiz_attempted is a failed attempt",
    "attrition_points": ATTRITION_POINTS,
    "attrition_tiers": {tier: f"score >= {t}" for t, tier in ATTRITION_TIERS},
    "negative_comment_pattern": NEGATIVE_COMMENT_PATTERN,
}


class GrainError(Exception):
    """Raised when a Gold table violates its declared grain (duplicate keys)."""


class GoldCheckError(Exception):
    """Raised when a Gold table does not reconcile with Silver or breaks a business-rule invariant."""


# ----------------------------------------------------------------- helpers
def _employee_dim(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Employees enriched with team attributes - the spine for employee-grain tables."""
    teams = silver["teams"][["team_id", "name", "department"]].rename(columns={"name": "team_name"})
    emp = silver["employees"][["employee_id", "full_name", "email", "team_id", "job_level", "hire_date"]]
    return emp.merge(teams, on="team_id", how="left")


def _latest_assessments(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Most recent assessment per employee x competency, with gap metrics."""
    a = silver["competency_assessments"].sort_values(["assessed_date", "assessment_id"])
    a = a.drop_duplicates(["employee_id", "competency_id"], keep="last")
    comp = silver["competencies"][["competency_id", "name", "category"]].rename(
        columns={"name": "competency", "category": "competency_category"}
    )
    a = a.merge(comp, on="competency_id", how="left")
    a["gap"] = (a["required_level"] - a["proficiency_level"]).astype("Int64")
    a["is_critical_gap"] = (a["gap"] >= CRITICAL_GAP_MIN) | (
        (a["gap"] >= 1) & (a["required_level"] >= KEY_COMPETENCY_MIN_REQUIRED)
    )
    a["gap_severity"] = np.select(
        [a["is_critical_gap"], a["gap"] >= 1, a["gap"] == 0],
        ["Critical", "Moderate", "Meets"],
        default="Exceeds",
    )
    return a


def _as_of(silver: dict[str, pd.DataFrame], run_date: date) -> pd.Timestamp:
    """Analysis date: the run date, capped at the freshest learning event we hold."""
    latest_event = silver["learning_activity"]["event_date"].max()
    run_ts = pd.Timestamp(run_date)
    return min(run_ts, latest_event.normalize()) if pd.notna(latest_event) else run_ts


# ----------------------------------------------------------------- Q1
def build_review_history(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per employee per review period, with change vs the previous period."""
    r = silver["performance_reviews"].sort_values(["employee_id", "review_year", "review_half"])
    r = r.assign(previous_rating=r.groupby("employee_id")["rating"].shift(1))
    r["rating_change"] = r["rating"] - r["previous_rating"]
    cols = ["employee_id", "review_id", "review_period", "review_year", "review_half", "period_end_date",
            "rating", "previous_rating", "rating_change", "manager_comment", "submitted_date"]
    return _employee_dim(silver)[["employee_id", "full_name", "team_name", "department", "job_level"]].merge(
        r[cols], on="employee_id", how="inner"
    )


def build_performance_summary(silver: dict[str, pd.DataFrame], run_date: date) -> pd.DataFrame:
    """One row per employee: latest rating, trend and the full history as text."""
    hist = build_review_history(silver)
    hist["history_item"] = hist["review_period"] + ":" + hist["rating"].astype("string")
    g = hist.groupby("employee_id", sort=True)
    summary = pd.DataFrame({
        "reviews_count": g.size(),
        "first_period": g["review_period"].first(),
        "latest_period": g["review_period"].last(),
        "previous_rating": g["previous_rating"].last(),
        "latest_rating": g["rating"].last(),
        "avg_rating": g["rating"].mean().round(2),
        "latest_comment": g["manager_comment"].last(),
        "rating_history": g["history_item"].agg(" -> ".join),
    }).reset_index()
    summary["rating_change"] = summary["latest_rating"] - summary["previous_rating"]
    summary["rating_trend"] = np.select(
        [summary["rating_change"] > 0, summary["rating_change"] < 0, summary["rating_change"] == 0],
        ["Improving", "Declining", "Stable"],
        default="Single review",
    )
    summary["performance_band"] = pd.cut(
        summary["latest_rating"].astype(float), bins=[0, 2, 3, 4, 5],
        labels=["Needs Improvement", "Solid", "Strong", "Exceptional"],
    ).astype("string")
    out = _employee_dim(silver).merge(summary, on="employee_id", how="left")
    out["tenure_years"] = ((pd.Timestamp(run_date) - out["hire_date"]).dt.days / 365.25).round(1)
    return out


# ----------------------------------------------------------------- Q2
def build_skill_gaps(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every employee x competency where proficiency is below the required level."""
    a = _latest_assessments(silver)
    gaps = a[a["gap"] > 0]
    cols = ["employee_id", "competency_id", "competency", "competency_category", "proficiency_level",
            "required_level", "gap", "gap_severity", "is_critical_gap", "assessed_date"]
    dim = _employee_dim(silver)[["employee_id", "full_name", "team_name", "department", "job_level"]]
    return dim.merge(gaps[cols], on="employee_id", how="inner").sort_values(
        ["gap", "employee_id"], ascending=[False, True], ignore_index=True
    )


# ----------------------------------------------------------------- Q4
def build_training_summary(silver: dict[str, pd.DataFrame], run_date: date) -> pd.DataFrame:
    """Learning activity rolled up per employee (zeros for employees with no activity)."""
    ev = silver["learning_activity"]
    counts = pd.crosstab(ev["employee_id"], ev["event_type"]).add_prefix("n_")
    for t in ("enrollment", "module_completed", "course_completed", "quiz_attempted", "quiz_passed",
              "certification_earned"):
        counts[f"n_{t}"] = counts.get(f"n_{t}", 0)
    quiz = ev[ev["event_type"].isin(["quiz_attempted", "quiz_passed"])]
    completed = ev[ev["event_type"] == "course_completed"]
    certs = ev[ev["event_type"] == "certification_earned"]
    agg = pd.DataFrame({
        "first_activity": ev.groupby("employee_id")["event_date"].min(),
        "last_activity": ev.groupby("employee_id")["event_date"].max(),
        "avg_quiz_score": quiz.groupby("employee_id")["score"].mean().round(1),
        "distinct_courses_completed": completed.groupby("employee_id")["course_name"].nunique(),
        "courses_completed_list": completed.groupby("employee_id")["course_name"].agg(
            lambda s: "; ".join(sorted(set(s)))),
        "certifications_list": certs.groupby("employee_id")["course_name"].agg(
            lambda s: "; ".join(sorted(set(s)))),
    })
    t = counts.join(agg, how="outer").reset_index().rename(columns={
        "n_enrollment": "enrollments", "n_module_completed": "modules_completed",
        "n_course_completed": "courses_completed", "n_quiz_attempted": "quizzes_failed",
        "n_quiz_passed": "quizzes_passed", "n_certification_earned": "certifications_earned",
    })
    out = _employee_dim(silver)[["employee_id", "full_name", "team_name", "department", "job_level"]].merge(
        t, on="employee_id", how="left"
    )
    count_cols = ["enrollments", "modules_completed", "courses_completed", "quizzes_failed",
                  "quizzes_passed", "certifications_earned", "distinct_courses_completed"]
    out[count_cols] = out[count_cols].fillna(0).astype("Int64")
    out = out.drop(columns=[c for c in out.columns if c.startswith("n_")])
    attempts = out["quizzes_passed"] + out["quizzes_failed"]
    out["quiz_pass_rate"] = (out["quizzes_passed"] / attempts.where(attempts > 0)).astype("Float64").round(2)
    as_of = _as_of(silver, run_date)
    out["days_since_last_activity"] = (as_of - out["last_activity"].dt.normalize()).dt.days.astype("Int64")
    out["learning_active"] = out["days_since_last_activity"].fillna(10**6) <= INACTIVE_LEARNING_DAYS
    return out


# ----------------------------------------------------------------- Q3
def build_promotion_pipeline(silver: dict[str, pd.DataFrame], run_date: date) -> pd.DataFrame:
    """Combine talent review, performance, skill gaps and development into a promotion tier."""
    perf = build_performance_summary(silver, run_date)
    gaps = build_skill_gaps(silver)
    crit = gaps[gaps["is_critical_gap"]].groupby("employee_id").size().rename("critical_gap_count")
    train = build_training_summary(silver, run_date)[
        ["employee_id", "courses_completed", "certifications_earned"]]
    tr = silver["talent_review"][["employee_id", "review_cycle", "potential_rating", "promotion_readiness",
                                  "attrition_risk"]]
    df = (perf[["employee_id", "full_name", "team_name", "department", "job_level", "tenure_years",
                "latest_rating", "avg_rating", "rating_trend"]]
          .merge(tr, on="employee_id", how="inner")
          .merge(crit, on="employee_id", how="left")
          .merge(train, on="employee_id", how="left"))
    df["critical_gap_count"] = df["critical_gap_count"].fillna(0).astype("Int64")
    df["is_high_potential"] = (df["potential_rating"] == "High") & (df["latest_rating"] >= HIGH_PERFORMER_MIN_RATING)
    ready_now = df["promotion_readiness"] == "Ready Now"
    df["promotion_tier"] = np.select(
        [df["is_high_potential"] & ready_now & (df["critical_gap_count"] == 0),
         df["is_high_potential"] & ready_now,
         df["is_high_potential"] & (df["promotion_readiness"] == "1-2 Years")],
        ["1 - Ready now", "2 - Ready now, close skill gaps", "3 - Emerging (1-2 years)"],
        default="4 - Not in pipeline",
    )
    df["retention_flag"] = df["is_high_potential"] & (df["attrition_risk"] == "High")
    return df.sort_values(["promotion_tier", "latest_rating", "avg_rating"], ascending=[True, False, False],
                          ignore_index=True)


# ----------------------------------------------------------------- Q5
def build_attrition_risk(silver: dict[str, pd.DataFrame], run_date: date) -> pd.DataFrame:
    """Rule-based attrition score from performance trend, sentiment, engagement and manager view."""
    perf = build_performance_summary(silver, run_date)
    train = build_training_summary(silver, run_date)[["employee_id", "learning_active", "days_since_last_activity"]]
    tr = silver["talent_review"][["employee_id", "attrition_risk", "potential_rating"]].rename(
        columns={"attrition_risk": "manager_attrition_risk"})
    df = perf.merge(train, on="employee_id", how="left").merge(tr, on="employee_id", how="left")

    signals = pd.DataFrame({
        "rating_declined": (df["rating_change"] < 0).fillna(False),
        "low_latest_rating": (df["latest_rating"] <= LOW_PERFORMER_MAX_RATING).fillna(False),
        "negative_manager_comment": df["latest_comment"].fillna("").str.contains(
            NEGATIVE_COMMENT_PATTERN, case=False, regex=True),
        "no_recent_learning": ~df["learning_active"].fillna(False).astype(bool),
        "manager_flag_high": df["manager_attrition_risk"] == "High",
        "manager_flag_medium": df["manager_attrition_risk"] == "Medium",
    }).astype(bool)
    df["risk_score"] = sum(signals[k].astype(int) * pts for k, pts in ATTRITION_POINTS.items())
    df["risk_drivers"] = signals.apply(lambda row: ", ".join(k for k, v in row.items() if v), axis=1)
    df["computed_risk_tier"] = "Low"
    for threshold, tier in sorted(ATTRITION_TIERS):
        df.loc[df["risk_score"] >= threshold, "computed_risk_tier"] = tier
    cols = ["employee_id", "full_name", "team_name", "department", "job_level", "tenure_years",
            "rating_history", "latest_rating", "rating_change", "rating_trend", "latest_comment",
            "days_since_last_activity", "potential_rating", "manager_attrition_risk", "risk_score",
            "computed_risk_tier", "risk_drivers"]
    return df[cols].sort_values(["risk_score", "employee_id"], ascending=[False, True], ignore_index=True)


# ----------------------------------------------------------------- Q6
def build_team_outcomes(silver: dict[str, pd.DataFrame], run_date: date) -> pd.DataFrame:
    """Team-level people metrics joined to project delivery outcomes."""
    perf = build_performance_summary(silver, run_date)
    risk = build_attrition_risk(silver, run_date)[["employee_id", "computed_risk_tier"]]
    train = build_training_summary(silver, run_date)[["employee_id", "courses_completed"]]
    crit = build_skill_gaps(silver).query("is_critical_gap").groupby("employee_id").size().rename("crit")
    people = (perf.merge(risk, on="employee_id").merge(train, on="employee_id")
              .merge(crit, on="employee_id", how="left").fillna({"crit": 0}))
    team_people = people.groupby("team_id").agg(
        headcount=("employee_id", "count"),
        avg_latest_rating=("latest_rating", "mean"),
        avg_rating_change=("rating_change", "mean"),
        pct_high_performers=("latest_rating", lambda s: (s >= HIGH_PERFORMER_MIN_RATING).mean() * 100),
        pct_high_attrition_risk=("computed_risk_tier", lambda s: (s == "High").mean() * 100),
        critical_gaps_per_head=("crit", "mean"),
        courses_completed_per_head=("courses_completed", "mean"),
    )
    p = silver["projects"]
    team_projects = p.groupby("team_id").agg(
        projects_count=("project_id", "count"),
        avg_outcome_score=("outcome_score", "mean"),
        pct_projects_on_time=("outcome_status", lambda s: (s == "on-time").mean() * 100),
        projects=("name", lambda s: "; ".join(sorted(s))),
    )
    teams = silver["teams"][["team_id", "name", "department"]].rename(columns={"name": "team_name"})
    out = teams.merge(team_people, on="team_id", how="left").merge(team_projects, on="team_id", how="left")
    out["projects_count"] = out["projects_count"].fillna(0).astype("Int64")
    out["has_projects"] = out["projects_count"] > 0
    num = out.select_dtypes("number").columns.difference(["team_id", "headcount", "projects_count"])
    out[num] = out[num].astype(float).round(2)
    return out


# ----------------------------------------------------------------- Q7
def build_skill_distribution(silver: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Proficiency distribution per competency, by department and organisation-wide ('All')."""
    a = _latest_assessments(silver).merge(
        _employee_dim(silver)[["employee_id", "department"]], on="employee_id", how="left")
    frames = [a, a.assign(department="All")]
    a = pd.concat(frames, ignore_index=True)
    keys = ["competency_id", "competency", "competency_category", "department"]
    g = a.groupby(keys)
    out = pd.DataFrame({
        "employees_assessed": g["employee_id"].nunique(),
        "avg_proficiency": g["proficiency_level"].mean().round(2),
        "avg_required": g["required_level"].mean().round(2),
        "pct_meeting_required": g["gap"].apply(lambda s: (s <= 0).mean() * 100).round(1),
        "critical_gaps": g["is_critical_gap"].sum().astype("Int64"),
    })
    levels = pd.crosstab([a[k] for k in keys], a["proficiency_level"]).reindex(columns=range(1, 6), fill_value=0)
    levels.columns = [f"level_{c}" for c in levels.columns]
    return out.join(levels).reset_index()


# ----------------------------------------------------------------- orchestration
@dataclass(frozen=True)
class GoldTable:
    """Registry entry: how to build a Gold table and what its grain is."""

    name: str
    grain: tuple[str, ...]
    question: str
    build: Callable[[dict[str, pd.DataFrame], date], pd.DataFrame]
    employee_grain: bool = False


GOLD_TABLES = (
    GoldTable("review_history", ("employee_id", "review_period"), "Q1 performance ratings & review history",
              lambda s, d: build_review_history(s)),
    GoldTable("employee_performance_summary", ("employee_id",), "Q1 performance ratings & review history",
              build_performance_summary, employee_grain=True),
    GoldTable("skill_gaps", ("employee_id", "competency_id"), "Q2 critical skill gaps",
              lambda s, d: build_skill_gaps(s)),
    GoldTable("promotion_pipeline", ("employee_id",), "Q3 high-potential employees ready for promotion",
              build_promotion_pipeline),
    GoldTable("training_summary", ("employee_id",), "Q4 training & development completed",
              build_training_summary, employee_grain=True),
    GoldTable("attrition_risk", ("employee_id",), "Q5 attrition risk from performance trends",
              build_attrition_risk, employee_grain=True),
    GoldTable("team_performance_outcomes", ("team_id",), "Q6 performance vs team & project outcomes",
              build_team_outcomes),
    GoldTable("skill_distribution", ("competency_id", "department"), "Q7 distribution of skills",
              lambda s, d: build_skill_distribution(s)),
)

SILVER_INPUTS = ("teams", "competencies", "employees", "projects", "performance_reviews",
                 "competency_assessments", "talent_review", "learning_activity")


def check_grain(df: pd.DataFrame, table: GoldTable) -> None:
    """Fail fast if a Gold table has duplicate rows at its declared grain."""
    dupes = int(df.duplicated(list(table.grain)).sum())
    if dupes:
        raise GrainError(f"{table.name}: {dupes} duplicate rows at grain {table.grain}")


# ----------------------------------------------------------------- reconciliation & invariants
_LEVELS = [f"level_{i}" for i in range(1, 6)]
_TRAINING_COUNTS = {
    "enrollments": "enrollment", "modules_completed": "module_completed", "courses_completed": "course_completed",
    "quizzes_failed": "quiz_attempted", "quizzes_passed": "quiz_passed", "certifications_earned": "certification_earned",
}


def reconcile(name: str, df: pd.DataFrame, silver: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Cross-check a Gold table against Silver and its own business rules.

    Every entry is ``{"expected", "actual", "ok"}``. Three kinds of check:
    cardinality (row counts vs the Silver population), conservation (sums in
    Gold equal counts in Silver, so nothing was dropped or double counted) and
    invariants (every row obeys the rule that produced it).
    """
    checks: dict[str, dict] = {}

    def eq(label: str, expected, actual) -> None:
        expected, actual = int(expected), int(actual)
        checks[label] = {"expected": expected, "actual": actual, "ok": expected == actual}

    n_emp = len(silver["employees"])
    if name == "review_history":
        eq("rows_equal_silver_reviews", len(silver["performance_reviews"]), len(df))
    elif name == "employee_performance_summary":
        eq("one_row_per_employee", n_emp, len(df))
        means = silver["performance_reviews"].groupby("employee_id")["rating"].mean().round(2)
        joined = df.set_index("employee_id")["avg_rating"].astype(float)
        eq("avg_rating_mismatches", 0, (joined.sub(means.astype(float), fill_value=0).abs() > 1e-9).sum())
    elif name == "skill_gaps":
        latest = _latest_assessments(silver)
        eq("rows_equal_assessments_below_required", (latest["gap"] > 0).sum(), len(df))
        eq("critical_gaps_equal_rule", latest["is_critical_gap"].sum(), df["is_critical_gap"].sum())
    elif name == "promotion_pipeline":
        eq("rows_equal_talent_reviews", len(silver["talent_review"]), len(df))
        tier1 = df[df["promotion_tier"] == "1 - Ready now"]
        violations = (~tier1["is_high_potential"] | (tier1["promotion_readiness"] != "Ready Now")
                      | (tier1["critical_gap_count"] > 0)).sum()
        eq("ready_now_rule_violations", 0, violations)
    elif name == "training_summary":
        eq("one_row_per_employee", n_emp, len(df))
        events = silver["learning_activity"]["event_type"].value_counts()
        for col, event_type in _TRAINING_COUNTS.items():
            eq(f"{col}_sum_equals_silver_events", events.get(event_type, 0), df[col].sum())
    elif name == "attrition_risk":
        eq("one_row_per_employee", n_emp, len(df))
        rescored = df["risk_drivers"].fillna("").map(
            lambda s: sum(ATTRITION_POINTS[d] for d in s.split(", ") if d))
        eq("risk_score_equals_sum_of_driver_points", 0, (rescored != df["risk_score"]).sum())
    elif name == "team_performance_outcomes":
        eq("one_row_per_team", len(silver["teams"]), len(df))
        eq("headcount_sums_to_employees", n_emp, df["headcount"].sum())
        eq("project_counts_sum_to_silver_projects", len(silver["projects"]), df["projects_count"].sum())
    elif name == "skill_distribution":
        org = df[df["department"] == "All"]
        eq("org_levels_sum_to_latest_assessments", len(_latest_assessments(silver)), org[_LEVELS].to_numpy().sum())
        eq("department_rows_sum_to_org_rows", org[_LEVELS].to_numpy().sum(),
           df[df["department"] != "All"][_LEVELS].to_numpy().sum())
    return checks


# ----------------------------------------------------------------- headline answers
def headline_summary(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    """One short, factual answer per business question, for the notebook and the demo."""
    perf = tables["employee_performance_summary"]
    gaps = tables["skill_gaps"]
    promo = tables["promotion_pipeline"]
    train = tables["training_summary"]
    risk = tables["attrition_risk"]
    teams = tables["team_performance_outcomes"]
    skills = tables["skill_distribution"]
    with_projects = teams[teams["has_projects"]]
    crit = gaps[gaps["is_critical_gap"]]
    org_skills = skills[skills["department"] == "All"]
    manager_high = risk[risk["manager_attrition_risk"] == "High"]

    def corr(a: str, b: str) -> float | None:
        pair = with_projects[[a, b]].astype(float).dropna()
        if len(pair) < 3 or pair.nunique().min() < 2:
            return None  # a correlation needs at least 3 points with some variation
        value = pair[a].corr(pair[b])
        return None if pd.isna(value) else round(float(value), 3)

    return {
        "Q1_performance": {
            "employees_reviewed": int(perf["reviews_count"].notna().sum()),
            "avg_latest_rating": round(float(perf["latest_rating"].mean()), 2),
            "rating_trend": perf["rating_trend"].value_counts().to_dict(),
            "performance_band": perf["performance_band"].value_counts().to_dict(),
        },
        "Q2_skill_gaps": {
            "critical_gaps": int(len(crit)),
            "employees_with_critical_gap": int(crit["employee_id"].nunique()),
            "top_competencies_by_critical_gaps": crit["competency"].value_counts().head(5).to_dict(),
        },
        "Q3_promotion": {
            "tiers": promo["promotion_tier"].value_counts().sort_index().to_dict(),
            "high_potential_at_high_attrition_risk": int(promo["retention_flag"].sum()),
        },
        "Q4_training": {
            "avg_courses_completed": round(float(train["courses_completed"].mean()), 2),
            "certifications_earned": int(train["certifications_earned"].sum()),
            "employees_with_no_learning_activity": int((train["enrollments"] == 0).sum()),
            "employees_learning_active": int(train["learning_active"].sum()),
        },
        "Q5_attrition": {
            "computed_tiers": risk["computed_risk_tier"].value_counts().to_dict(),
            "manager_high_also_computed_high": int((manager_high["computed_risk_tier"] == "High").sum()),
            "manager_high_total": int(len(manager_high)),
        },
        "Q6_team_outcomes": {
            "teams_with_projects": int(len(with_projects)),
            "corr_avg_rating_vs_outcome_score": corr("avg_latest_rating", "avg_outcome_score"),
            "corr_pct_high_performers_vs_on_time": corr("pct_high_performers", "pct_projects_on_time"),
            "corr_critical_gaps_vs_outcome_score": corr("critical_gaps_per_head", "avg_outcome_score"),
            "note": "team-level; only teams with projects; small sample, indicative only",
        },
        "Q7_skill_distribution": {
            "competencies": int(org_skills["competency_id"].nunique()),
            "lowest_pct_meeting_required": org_skills.nsmallest(5, "pct_meeting_required")
                .set_index("competency")["pct_meeting_required"].to_dict(),
            "avg_proficiency_by_category": org_skills.groupby("competency_category")["avg_proficiency"]
                .mean().round(2).to_dict(),
        },
    }


def build_gold(settings: Settings, lake: Lake, ctx: RunContext) -> None:
    """Read Silver once, build every Gold table, check grain + reconciliation, publish, summarise."""
    silver = {name: lake.read_parquet(f"{settings.silver_prefix}{name}/") for name in SILVER_INPUTS}
    silver_rows = sum(len(df) for df in silver.values())
    catalog, tables = {}, {}

    for table in GOLD_TABLES:
        with ctx.step("gold", table.name) as rec:
            df = table.build(silver, settings.run_date)
            check_grain(df, table)
            checks = reconcile(table.name, df, silver)
            rec.details.update(grain=list(table.grain), question=table.question, checks=checks)
            failed = {k: v for k, v in checks.items() if not v["ok"]}
            if failed:
                raise GoldCheckError(f"{table.name}: {failed}")
            tables[table.name] = df
            df = df.assign(_gold_run_id=ctx.run_id, _as_of=settings.run_date.isoformat())
            rec.outputs = lake.replace_dataset(f"{settings.gold_prefix}{table.name}/", df)
            rec.inputs = [lake.uri(f"{settings.silver_prefix}{n}/") for n in SILVER_INPUTS]
            rec.rows_read = silver_rows
            rec.rows_written = len(df)
            catalog[table.name] = {"grain": list(table.grain), "question": table.question, "rows": len(df),
                                   "checks_passed": sorted(checks), "columns": list(df.columns)}

    with ctx.step("gold", "_summary") as rec:
        summary = headline_summary(tables)
        lake.put_json(f"{settings.gold_prefix}_summary.json",
                      {"run_id": ctx.run_id, "as_of": settings.run_date.isoformat(), "answers": summary})
        rec.outputs = [lake.uri(f"{settings.gold_prefix}_summary.json")]
    lake.put_json(f"{settings.gold_prefix}_catalog.json",
                  {"run_id": ctx.run_id, "as_of": settings.run_date.isoformat(), "rules": RULES, "tables": catalog})
