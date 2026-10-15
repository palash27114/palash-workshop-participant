"""Shared fixtures: make the job modules importable and build a tiny Silver dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _dt(values):
    return pd.to_datetime(pd.Series(values))


@pytest.fixture
def silver() -> dict[str, pd.DataFrame]:
    """
    Three employees in two teams, hand-built so every Gold rule has a known answer:

    * emp 1 - rating 4 -> 5, High potential, Ready Now, no gaps, active learner  -> promote now
    * emp 2 - rating 4 -> 2, negative comment, no learning, manager risk High     -> high attrition
    * emp 3 - rating 3 -> 3, critical gap (prof 1 vs required 4)                   -> skill gap
    """
    teams = pd.DataFrame({"team_id": pd.array([10, 20], "Int64"), "name": ["Alpha", "Beta"],
                          "department": ["Engineering", "Sales"]})
    competencies = pd.DataFrame({"competency_id": pd.array([1, 2], "Int64"), "name": ["Python", "Negotiation"],
                                 "category": ["Technical", "Domain"]})
    employees = pd.DataFrame({
        "employee_id": pd.array([1, 2, 3], "Int64"),
        "first_name": ["Ann", "Bob", "Cy"], "last_name": ["A", "B", "C"],
        "email": ["a@x.io", "b@x.io", "c@x.io"], "team_id": pd.array([10, 10, 20], "Int64"),
        "job_level": ["IC2", "IC3", "Manager"], "hire_date": _dt(["2020-01-01", "2021-01-01", "2022-01-01"]),
    })
    employees["full_name"] = employees["first_name"] + " " + employees["last_name"]
    reviews = pd.DataFrame({
        "review_id": pd.array(range(1, 7), "Int64"),
        "employee_id": pd.array([1, 1, 2, 2, 3, 3], "Int64"),
        "review_period": ["2024-H2", "2025-H1"] * 3,
        "rating": pd.array([4, 5, 4, 2, 3, 3], "Int64"),
        "manager_comment": ["ok", "great", "ok", "Missed several key deliverables this cycle; needs support.", "ok", "ok"],
        "submitted_date": _dt(["2025-01-15", "2025-07-15"] * 3),
        "review_year": pd.array([2024, 2025] * 3, "Int64"),
        "review_half": pd.array([2, 1] * 3, "Int64"),
        "period_end_date": _dt(["2024-12-31", "2025-06-30"] * 3),
    })
    assessments = pd.DataFrame({
        "assessment_id": pd.array([1, 2, 3, 4], "Int64"),
        "employee_id": pd.array([1, 3, 3, 3], "Int64"),
        "competency_id": pd.array([1, 1, 2, 2], "Int64"),
        "proficiency_level": pd.array([4, 1, 2, 3], "Int64"),
        "required_level": pd.array([3, 4, 3, 3], "Int64"),
        # emp 3 / comp 2 was re-assessed later (2 -> 3): only the latest counts.
        "assessed_date": _dt(["2025-01-01", "2025-01-01", "2025-01-01", "2025-06-01"]),
    })
    talent = pd.DataFrame({
        "employee_id": pd.array([1, 2, 3], "Int64"), "review_cycle": ["2025-H1"] * 3,
        "potential_rating": ["High", "Medium", "Low"],
        "promotion_readiness": ["Ready Now", "Not Ready", "Not Ready"],
        "attrition_risk": ["Low", "High", "Low"],
    })
    projects = pd.DataFrame({"project_id": pd.array([1, 2], "Int64"), "name": ["P1", "P2"],
                             "team_id": pd.array([10, 10], "Int64"), "outcome_status": ["on-time", "delayed"],
                             "outcome_score": pd.array([80.0, 40.0], "Float64")})
    events = pd.DataFrame({
        "event_id": ["E1", "E2", "E3", "E4"],
        "employee_id": pd.array([1, 1, 1, 3], "Int64"),
        "event_date": _dt(["2026-08-01", "2026-08-10", "2026-08-20", "2025-01-01"]),
        "event_type": ["enrollment", "quiz_passed", "course_completed", "quiz_attempted"],
        "course_name": ["Py", "Py", "Py", "Neg"],
        "text": ["", "", "", ""],
        "score": pd.array([None, 90.0, None, 30.0], "Float64"),
    })
    events["event_year"] = events["event_date"].dt.year
    return {"teams": teams, "competencies": competencies, "employees": employees, "projects": projects,
            "performance_reviews": reviews, "competency_assessments": assessments, "talent_review": talent,
            "learning_activity": events}
