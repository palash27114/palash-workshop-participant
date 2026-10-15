"""
ACME Talent Lake - interactive dashboard over the Gold layer.

Run from data/talent-etl:

    .venv/bin/streamlit run dashboard/app.py

then open http://localhost:8501

Reads Gold only (same lake and environment variables as job.py). Use the
sidebar to filter by department and job level; every tab follows the filter.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from gold_data import (
    RISK_ORDER,
    TREND_ORDER,
    delivery_groups,
    employee_360,
    employee_options,
    filtered,
    kpis,
    load_gold,
)

# Colorblind-validated palette: one blue for single series, ordinal blues for tiers,
# status colors only for risk (always paired with a text label).
BLUE = "#2a78d6"
STATUS = {"High": "#d03b3b", "Medium": "#ec835a", "Low": "#0ca30c"}
TIER_COLORS = ["#0d366b", "#256abf", "#5598e7", "#9ec5f4"]
SEQ = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

st.set_page_config(page_title="ACME Talent Lake", page_icon=":material/insights:", layout="wide")


@st.cache_data(ttl=600, show_spinner="Reading the Gold layer…")
def get_gold():
    return load_gold()


def chart(c: alt.Chart) -> None:
    st.altair_chart(c, width="stretch")


def hbar(df: pd.DataFrame, y: str, x: str, x_title: str, fmt: str = ",.0f", color=BLUE,
         sort="-x", tooltip=None, color_scale: alt.Scale | None = None) -> alt.Chart:
    """Horizontal bars with the value at the bar end and a hover tooltip."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{x}:Q", title=x_title),
        tooltip=tooltip or [alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)],
    )
    if color_scale is not None:
        bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
            color=alt.Color(f"{y}:N", scale=color_scale, legend=None))
    else:
        bars = base.mark_bar(cornerRadiusEnd=4, size=18, color=color)
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{x}:Q", format=fmt))
    return (bars + text).properties(height=max(120, 30 * len(df)))


def vbar(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> alt.Chart:
    """Vertical bars for small ordered categories (ratings, levels)."""
    base = alt.Chart(df).encode(x=alt.X(f"{x}:O", title=x_title, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y(f"{y}:Q", title=y_title),
                                tooltip=[alt.Tooltip(f"{x}:O", title=x_title), alt.Tooltip(f"{y}:Q", format=",")])
    return (base.mark_bar(cornerRadiusEnd=4, size=36, color=BLUE)
            + base.mark_text(dy=-8).encode(text=alt.Text(f"{y}:Q", format=","))).properties(height=260)


# --------------------------------------------------------------------------- load + sidebar
try:
    gold = get_gold()
except Exception as exc:  # noqa: BLE001 - show the operator what to do instead of a stack trace
    st.error(f"Could not read the Gold layer: {exc}")
    st.info("Run the pipeline first: `cd data/talent-etl && .venv/bin/python seed_sources.py && "
            ".venv/bin/python job.py`. Locally LocalStack must be running; set PARTICIPANT_ID or LAKE_BUCKET.")
    st.stop()

perf_all = gold.tables["employee_performance_summary"]
with st.sidebar:
    st.header("Filters")
    departments = st.multiselect("Department", sorted(perf_all["department"].dropna().unique()),
                                 placeholder="All departments", key="f_departments")
    levels = st.multiselect("Job level", sorted(perf_all["job_level"].dropna().unique()),
                            placeholder="All job levels", key="f_levels")
    st.divider()
    st.caption(f"Data as of **{gold.catalog.get('as_of', '?')}**")
    st.caption(f"Gold run `{gold.catalog.get('run_id', '?')}`")
    st.caption(f"Lake `{gold.lake_uri}`")
    if st.button("Refresh data", key="refresh"):
        get_gold.clear()
        st.rerun()

t = filtered(gold, departments, levels)
k = kpis(t)
scope = ", ".join(departments) if departments else "all departments"
if levels:
    scope += f" · {', '.join(levels)}"

st.title("ACME Talent Lake")
st.caption(f"Performance, skills, development and attrition from the Gold layer · showing **{scope}** "
           f"({k['employees']:,} employees)")

if k["employees"] == 0:
    st.warning("No employees match these filters. Clear a filter in the sidebar.")
    st.stop()

tabs = st.tabs(["Overview", "Employee 360", "Performance & attrition", "Skills", "Promotion",
                "Teams & projects", "Data quality"])

# --------------------------------------------------------------------------- Overview
with tabs[0]:
    cols = st.columns(4)
    cols[0].metric("Employees", f"{k['employees']:,}")
    cols[1].metric("Average latest rating", f"{k['avg_rating']:.2f} / 5")
    cols[2].metric("Ready for promotion now", f"{k['ready_now']:,}")
    cols[3].metric("High attrition risk", f"{k['high_risk']:,}")
    cols = st.columns(4)
    cols[0].metric("Rating declined", f"{k['declining']:,}")
    cols[1].metric("With a critical skill gap", f"{k['with_critical_gap']:,}")
    cols[2].metric("No learning activity", f"{k['no_learning']:,}")
    cols[3].metric("High potential, high risk", f"{k['retention_flags']:,}",
                   help="High-potential employees whose manager rates them high attrition risk")

    left, right = st.columns(2)
    with left:
        st.subheader("Latest rating")
        dist = (t["employee_performance_summary"]["latest_rating"].value_counts()
                .reindex(range(1, 6), fill_value=0).rename_axis("rating").reset_index(name="employees"))
        chart(vbar(dist, "rating", "employees", "rating (1-5)", "employees"))
    with right:
        st.subheader("Computed attrition risk")
        risk_counts = (t["attrition_risk"]["computed_risk_tier"].value_counts().reindex(RISK_ORDER, fill_value=0)
                       .rename_axis("tier").reset_index(name="employees"))
        risk_counts["tier"] = risk_counts["tier"] + " risk"
        chart(hbar(risk_counts, "tier", "employees", "employees", sort=None,
                   color_scale=alt.Scale(domain=[f"{r} risk" for r in RISK_ORDER], range=list(STATUS.values()))))

# --------------------------------------------------------------------------- Employee 360
with tabs[1]:
    options = employee_options(t["employee_performance_summary"])
    choice = st.selectbox("Find an employee", list(options), index=None, placeholder="Type a name…",
                          key="employee_pick")
    if choice is None:
        st.info("Type a name to see one employee's reviews, skill gaps, training, promotion tier and attrition risk.")
    else:
        e = employee_360(t, options[choice])
        p, pr, tr, rk = e["profile"], e["promotion"], e["training"], e["risk"]
        st.subheader(p["full_name"])
        st.caption(f"{p['job_level']} · {p['team_name']} · {p['department']} · {p['tenure_years']} years · {p['email']}")
        c = st.columns(4)
        c[0].metric("Latest rating", f"{p['latest_rating']} / 5",
                    delta=None if pd.isna(p["rating_change"]) else f"{p['rating_change']:+.0f} vs previous")
        c[1].metric("Promotion tier", pr["promotion_tier"].split(" - ")[1] if pr is not None else "n/a")
        c[2].metric("Attrition risk", f"{rk['computed_risk_tier']} ({rk['risk_score']} pts)",
                    help=f"Signals: {rk['risk_drivers'] or 'none'}")
        c[3].metric("Courses completed", int(tr["courses_completed"]),
                    help=f"Certifications: {int(tr['certifications_earned'])}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Review history**")
            st.dataframe(e["reviews"][["review_period", "rating", "rating_change", "manager_comment"]],
                         hide_index=True, width="stretch")
            st.markdown("**Skill gaps**")
            if e["gaps"].empty:
                st.success("No skill gaps: meets or exceeds every assessed required level.")
            else:
                st.dataframe(e["gaps"][["competency", "proficiency_level", "required_level", "gap", "gap_severity"]],
                             hide_index=True, width="stretch")
        with right:
            st.markdown("**Attrition signals**")
            if rk["risk_drivers"]:
                for driver in rk["risk_drivers"].split(", "):
                    st.markdown(f"- {driver.replace('_', ' ')}")
            else:
                st.markdown("No risk signals.")
            st.caption(f"Manager's own assessment: {rk['manager_attrition_risk']} · potential {rk['potential_rating']}")
            st.markdown("**Learning**")
            pass_rate = "n/a" if pd.isna(tr["quiz_pass_rate"]) else f"{tr['quiz_pass_rate']:.0%}"
            st.markdown(
                f"Enrolments {int(tr['enrollments'])} · courses {int(tr['courses_completed'])} · "
                f"certifications {int(tr['certifications_earned'])} · quiz pass rate {pass_rate}")
            if isinstance(tr["courses_completed_list"], str) and tr["courses_completed_list"]:
                st.caption(f"Completed: {tr['courses_completed_list']}")
            if isinstance(tr["certifications_list"], str) and tr["certifications_list"]:
                st.caption(f"Certified: {tr['certifications_list']}")

# --------------------------------------------------------------------------- Performance & attrition
with tabs[2]:
    perf, risk = t["employee_performance_summary"], t["attrition_risk"]
    left, right = st.columns(2)
    with left:
        st.subheader("Change since previous review")
        trend = (perf["rating_trend"].value_counts().reindex(TREND_ORDER, fill_value=0)
                 .rename_axis("trend").reset_index(name="employees"))
        chart(hbar(trend, "trend", "employees", "employees", sort=None))
    with right:
        st.subheader("Signals behind high-risk employees")
        drivers = (risk.loc[risk["computed_risk_tier"] == "High", "risk_drivers"].str.split(", ").explode()
                   .dropna().str.replace("_", " ").value_counts().rename_axis("signal").reset_index(name="employees"))
        if drivers.empty:
            st.info("No high-risk employees in this selection.")
        else:
            chart(hbar(drivers, "signal", "employees", "high-risk employees with this signal"))

    st.subheader("Manager flag vs computed risk")
    cross = (pd.crosstab(risk["manager_attrition_risk"], risk["computed_risk_tier"])
             .reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0)
             .rename_axis(index="manager flag", columns="computed tier"))
    st.dataframe(cross, width="content")

    st.subheader("Employees at risk")
    tier_pick = st.segmented_control("Tier", RISK_ORDER, default="High", key="risk_tier")
    at_risk = risk[risk["computed_risk_tier"] == (tier_pick or "High")]
    st.dataframe(at_risk[["full_name", "department", "job_level", "rating_history", "latest_comment",
                          "manager_attrition_risk", "risk_score", "risk_drivers"]],
                 hide_index=True, width="stretch")
    st.download_button("Download as CSV", at_risk.to_csv(index=False), "attrition_risk.csv", "text/csv",
                       key="dl_risk")

# --------------------------------------------------------------------------- Skills
with tabs[3]:
    gaps = t["skill_gaps"]
    crit = gaps[gaps["is_critical_gap"]]
    st.caption(f"Critical gap rule: {gold.catalog.get('rules', {}).get('critical_skill_gap', '')}")
    left, right = st.columns(2)
    with left:
        st.subheader("Competencies with the most critical gaps")
        top = crit["competency"].value_counts().head(12).rename_axis("competency").reset_index(name="critical gaps")
        if top.empty:
            st.info("No critical gaps in this selection.")
        else:
            chart(hbar(top, "competency", "critical gaps", "critical gaps"))
    with right:
        st.subheader("Weakest coverage of required levels")
        dist = t["skill_distribution"]
        weighted = dist.assign(meeting=dist["pct_meeting_required"] * dist["employees_assessed"] / 100)
        sums = weighted.groupby("competency")[["meeting", "employees_assessed"]].sum()
        cover = ((sums["meeting"] / sums["employees_assessed"] * 100).nsmallest(12)
                 .rename_axis("competency").reset_index(name="% meeting required"))
        chart(hbar(cover, "competency", "% meeting required", "% of assessed employees", fmt=".0f", sort="x"))

    st.subheader("Coverage by competency and department")
    heat = t["skill_distribution"]
    order = (heat.groupby("competency")["pct_meeting_required"].mean().sort_values().index.tolist())
    chart(alt.Chart(heat).mark_rect().encode(
        x=alt.X("department:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("competency:N", sort=order, title=None),
        color=alt.Color("pct_meeting_required:Q", title="% meeting required",
                        scale=alt.Scale(domain=[0, 100], range=SEQ)),
        tooltip=["competency", "department", alt.Tooltip("pct_meeting_required:Q", title="% meeting", format=".0f"),
                 "employees_assessed", "critical_gaps", alt.Tooltip("avg_proficiency:Q", format=".2f")],
    ).properties(height=900))
    st.caption("Each cell rests on roughly 10 assessments; read rows and columns rather than single squares.")

    st.subheader("All skill gaps")
    severity = st.segmented_control("Severity", ["Critical", "Moderate"], default="Critical", key="gap_sev")
    shown = gaps[gaps["gap_severity"] == (severity or "Critical")]
    st.dataframe(shown[["full_name", "department", "job_level", "competency", "competency_category",
                        "proficiency_level", "required_level", "gap"]], hide_index=True, width="stretch")
    st.download_button("Download as CSV", shown.to_csv(index=False), "skill_gaps.csv", "text/csv", key="dl_gaps")

# --------------------------------------------------------------------------- Promotion
with tabs[4]:
    promo = t["promotion_pipeline"]
    rules = gold.catalog.get("rules", {}).get("promotion_tiers", {})
    tiers = promo["promotion_tier"].value_counts().reindex(list(rules) or None, fill_value=0) \
        .rename_axis("tier").reset_index(name="employees")
    tiers["label"] = tiers["tier"].str.split(" - ").str[1]
    st.subheader("Promotion pipeline")
    chart(hbar(tiers, "label", "employees", "employees", sort=None,
               color_scale=alt.Scale(domain=tiers["label"].tolist(), range=TIER_COLORS[: len(tiers)])))
    with st.expander("How tiers are decided"):
        for tier, rule in rules.items():
            st.markdown(f"- **{tier}**: {rule}")

    tier_pick = st.selectbox("Show tier", list(rules) or sorted(promo["promotion_tier"].unique()), index=0,
                             key="promo_tier")
    st.dataframe(promo[promo["promotion_tier"] == tier_pick][
        ["full_name", "department", "job_level", "tenure_years", "latest_rating", "avg_rating", "rating_trend",
         "critical_gap_count", "certifications_earned", "attrition_risk"]],
        hide_index=True, width="stretch")
    flags = promo[promo["retention_flag"]]
    st.subheader(f"Retention watch: {len(flags)} high-potential employees at high attrition risk")
    st.dataframe(flags[["full_name", "department", "job_level", "promotion_tier", "latest_rating",
                        "attrition_risk"]], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- Teams & projects
with tabs[5]:
    teams = t["team_performance_outcomes"]
    wp = teams[teams["has_projects"]]
    st.caption("Projects belong to teams, so outcomes are compared at team level. Only teams that own a project "
               f"appear here ({len(wp)} in this selection); treat relationships as indicative.")
    if len(wp) < 3:
        st.info("Fewer than three teams with projects in this selection; widen the department filter.")
    else:
        r = wp["avg_latest_rating"].corr(wp["avg_outcome_score"])
        left, right = st.columns(2)
        with left:
            st.subheader(f"Team rating vs project outcome (r = {r:+.2f})")
            points = alt.Chart(wp).mark_circle(size=90, color=BLUE, opacity=0.9, stroke="white", strokeWidth=1.5).encode(
                x=alt.X("avg_latest_rating:Q", title="team average rating", scale=alt.Scale(zero=False)),
                y=alt.Y("avg_outcome_score:Q", title="project outcome score", scale=alt.Scale(zero=False)),
                tooltip=["team_name", "department", "headcount", alt.Tooltip("avg_latest_rating:Q", format=".2f"),
                         "projects", alt.Tooltip("avg_outcome_score:Q", format=".1f"), "pct_projects_on_time"])
            chart(points + points.transform_regression("avg_latest_rating", "avg_outcome_score")
                  .mark_line(color="gray", strokeDash=[5, 4]))
        with right:
            st.subheader("High performers by delivery result")
            groups = delivery_groups(teams)
            groups["label"] = groups["delivery"] + " (" + groups["teams"].astype(str) + " teams)"
            chart(hbar(groups, "label", "pct_high_performers", "% high performers in team", fmt=".0f", sort=None))
    st.dataframe(wp[["team_name", "department", "headcount", "avg_latest_rating", "pct_high_performers",
                     "critical_gaps_per_head", "courses_completed_per_head", "projects", "avg_outcome_score",
                     "pct_projects_on_time"]].sort_values("avg_outcome_score", ascending=False),
                 hide_index=True, width="stretch")

# --------------------------------------------------------------------------- Data quality
with tabs[6]:
    st.caption("Pipeline health from the last run. Unaffected by the sidebar filters.")
    if gold.manifest:
        m = gold.manifest
        c = st.columns(3)
        c[0].metric("Last run", m["status"])
        c[1].metric("Duration", f"{m['duration_s']:.1f} s")
        c[2].metric("Run id", m["run_id"][-13:])
        steps = pd.DataFrame([{"layer": s["layer"], "dataset": s["dataset"], "rows read": s["rows_read"],
                               "rows written": s["rows_written"], "quarantined": s["rows_quarantined"],
                               "status": s["status"], "seconds": s["duration_s"]} for s in m["steps"]])
        st.dataframe(steps, hide_index=True, width="stretch")
    if gold.dq_report:
        la = gold.dq_report["datasets"]["learning_activity"]
        st.subheader(f"Learning events: {la['rows_quarantined']:,} of {la['rows_in']:,} quarantined")
        reasons = (pd.Series(la["quarantine_reasons"]).rename_axis("reason").reset_index(name="rows"))
        reasons["reason"] = reasons["reason"].str.replace("_", " ")
        chart(hbar(reasons, "reason", "rows", "quarantined rows"))
        st.caption("Duplicates are extra copies whose original is kept; the other reasons are events that cannot be "
                   "used until the source system corrects them.")
        for reason, rows in la["quarantine_samples"].items():
            with st.expander(f"Example: {reason}"):
                st.code(rows[0].get("_raw_line") or str(rows[0]), language="json")
