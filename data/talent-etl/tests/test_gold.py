"""Unit tests for Gold business rules against the hand-built Silver fixture."""

from datetime import date

import pandas as pd
import pytest

import gold

RUN = date(2026, 9, 1)


def _by_emp(df):
    return df.set_index("employee_id")


def test_performance_summary_trend_and_history(silver):
    s = _by_emp(gold.build_performance_summary(silver, RUN))
    assert s.loc[1, "rating_trend"] == "Improving"
    assert s.loc[2, "rating_trend"] == "Declining"
    assert s.loc[3, "rating_trend"] == "Stable"
    assert s.loc[2, "rating_history"] == "2024-H2:4 -> 2025-H1:2"
    assert s.loc[1, "performance_band"] == "Exceptional"


def test_skill_gaps_use_latest_assessment_and_severity(silver):
    gaps = gold.build_skill_gaps(silver)
    # emp 3 / Negotiation went 2 -> 3 (meets required 3): no longer a gap.
    assert gaps[["employee_id", "competency"]].values.tolist() == [[3, "Python"]]
    row = gaps.iloc[0]
    assert row["gap"] == 3 and row["gap_severity"] == "Critical" and row["is_critical_gap"]


def test_training_summary_zero_fills_inactive_employees(silver):
    t = _by_emp(gold.build_training_summary(silver, RUN))
    assert t.loc[1, "courses_completed"] == 1
    assert t.loc[1, "quiz_pass_rate"] == 1.0
    assert t.loc[2, "enrollments"] == 0 and not t.loc[2, "learning_active"]
    assert t.loc[3, "quiz_pass_rate"] == 0.0


def test_promotion_pipeline_tiers(silver):
    p = _by_emp(gold.build_promotion_pipeline(silver, RUN))
    assert p.loc[1, "promotion_tier"] == "1 - Ready now"
    assert p.loc[2, "promotion_tier"] == "4 - Not in pipeline"


def test_attrition_risk_scores_trend_and_signals(silver):
    a = _by_emp(gold.build_attrition_risk(silver, RUN))
    assert a.loc[2, "computed_risk_tier"] == "High"
    assert {"rating_declined", "low_latest_rating", "negative_manager_comment"} <= set(
        a.loc[2, "risk_drivers"].split(", "))
    assert a.loc[1, "computed_risk_tier"] == "Low"


def test_team_outcomes_join_people_and_projects(silver):
    t = gold.build_team_outcomes(silver, RUN).set_index("team_id")
    assert t.loc[10, "headcount"] == 2
    assert t.loc[10, "pct_projects_on_time"] == 50.0
    assert not t.loc[20, "has_projects"]


def test_skill_distribution_includes_org_wide_rows(silver):
    d = gold.build_skill_distribution(silver)
    org = d[(d["department"] == "All") & (d["competency"] == "Python")].iloc[0]
    assert org["employees_assessed"] == 2 and org["level_1"] == 1 and org["level_4"] == 1


def test_every_gold_table_respects_its_grain(silver):
    for table in gold.GOLD_TABLES:
        gold.check_grain(table.build(silver, RUN), table)


def test_grain_violation_raises():
    table = gold.GOLD_TABLES[1]
    with pytest.raises(gold.GrainError):
        gold.check_grain(pd.DataFrame({"employee_id": [1, 1]}), table)


def _built(silver):
    return {t.name: t.build(silver, RUN) for t in gold.GOLD_TABLES}


def test_every_gold_table_reconciles_with_silver(silver):
    for name, df in _built(silver).items():
        checks = gold.reconcile(name, df, silver)
        assert checks, f"{name} has no reconciliation checks"
        assert all(c["ok"] for c in checks.values()), (name, checks)


@pytest.mark.parametrize("name,tamper,check", [
    ("review_history", lambda d: d.iloc[1:], "rows_equal_silver_reviews"),
    ("training_summary", lambda d: d.assign(courses_completed=d["courses_completed"] + 1),
     "courses_completed_sum_equals_silver_events"),
    ("attrition_risk", lambda d: d.assign(risk_score=d["risk_score"] + 1), "risk_score_equals_sum_of_driver_points"),
    ("team_performance_outcomes", lambda d: d.assign(headcount=d["headcount"] * 2), "headcount_sums_to_employees"),
    ("promotion_pipeline", lambda d: d.assign(promotion_tier="1 - Ready now"), "ready_now_rule_violations"),
    ("skill_distribution", lambda d: d.assign(level_1=d["level_1"] + 1), "department_rows_sum_to_org_rows"),
])
def test_reconcile_catches_dropped_or_wrong_numbers(silver, name, tamper, check):
    df = tamper(_built(silver)[name])
    assert not gold.reconcile(name, df, silver)[check]["ok"]


def test_headline_summary_answers_every_question(silver):
    summary = gold.headline_summary(_built(silver))
    assert [k.split("_")[0] for k in summary] == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]
    assert summary["Q2_skill_gaps"]["employees_with_critical_gap"] == 1
    assert summary["Q3_promotion"]["tiers"]["1 - Ready now"] == 1
    assert summary["Q5_attrition"]["manager_high_also_computed_high"] == 1
