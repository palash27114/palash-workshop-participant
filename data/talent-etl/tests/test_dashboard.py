"""Dashboard tests: pure data helpers, plus a headless run of the whole app against the local lake."""

import sys
from pathlib import Path

import pandas as pd
import pytest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD))

import gold_data  # noqa: E402


def _people():
    return pd.DataFrame({"employee_id": [1, 2, 3], "full_name": ["Bo", "Al", "Cy"], "team_name": ["T", "T", "U"],
                         "department": ["Sales", "HR", "Sales"], "job_level": ["IC1", "IC2", "IC2"]})


def test_filter_people_empty_selection_means_all():
    assert len(gold_data.filter_people(_people(), [], [])) == 3
    assert gold_data.filter_people(_people(), ["Sales"], ["IC2"])["employee_id"].tolist() == [3]


def test_employee_options_sorted_by_name():
    assert list(gold_data.employee_options(_people()).values()) == [2, 1, 3]


def test_delivery_groups_buckets_by_on_time_share():
    teams = pd.DataFrame({"team_id": [1, 2, 3, 4], "has_projects": [True, True, True, False],
                          "pct_projects_on_time": [100.0, 50.0, 0.0, None],
                          "pct_high_performers": [60.0, 40.0, 20.0, None], "avg_latest_rating": [4, 3, 2, None]})
    g = gold_data.delivery_groups(teams)
    assert g["delivery"].tolist() == ["All on time", "Mixed", "All delayed"]
    assert g["teams"].tolist() == [1, 1, 1]


@pytest.mark.integration
def test_app_renders_every_tab_without_errors():
    from streamlit.testing.v1 import AppTest
    try:
        gold_data.load_gold()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gold layer not available: {exc}")
    at = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 7
    assert at.metric[0].value == "2,000"
    # Pick an employee and filter to one department: still no errors.
    at.selectbox(key="employee_pick").select_index(0).run()
    assert not at.exception, at.exception
    at.multiselect(key="f_departments").select("Sales").run()
    assert not at.exception, at.exception
    assert at.metric[0].value != "2,000"


def test_password_gate_hides_data_until_correct_password(monkeypatch):
    """Public deployments set DASHBOARD_PASSWORD; nothing but the password box renders before it is entered."""
    from streamlit.testing.v1 import AppTest
    try:
        gold_data.load_gold()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Gold layer not available: {exc}")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "s3cret-test")
    at = AppTest.from_file(str(DASHBOARD / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 0 and len(at.metric) == 0
    at.text_input(key="dashboard_password").input("wrong").run()
    assert at.error and len(at.tabs) == 0
    at.text_input(key="dashboard_password").input("s3cret-test").run()
    assert not at.exception
    assert len(at.tabs) == 7
