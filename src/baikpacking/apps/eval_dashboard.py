"""Local Streamlit dashboard for manual eval scenario runs."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_RUNS_PATH = Path(__file__).resolve().parents[3] / "data/eval/scenario_runs.jsonl"


def _load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame, skipping blank lines."""
    if not path.exists():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Ensure a DataFrame has a fixed set of columns."""
    if df.empty:
        return pd.DataFrame(columns=columns)
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    """Convert truthy values to booleans while preserving missing values."""
    if series.empty:
        return series
    return series.map(lambda value: bool(value) if pd.notna(value) else pd.NA)


def _list_from_cell(value: Any) -> list[Any]:
    """Normalize a cell into a list for issue aggregation and details."""
    if value is None or value is pd.NA:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return [value]


def _issue_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """Count issue strings in a list-valued column."""
    if df.empty or column not in df.columns:
        return pd.Series(dtype="int64")
    exploded = df[column].apply(_list_from_cell).explode()
    exploded = exploded.dropna()
    exploded = exploded[exploded.astype(str).str.strip() != ""]
    if exploded.empty:
        return pd.Series(dtype="int64")
    return exploded.astype(str).value_counts().sort_values(ascending=False)


def _pass_fail_by_category(df: pd.DataFrame, category_col: str, pass_col: str) -> pd.DataFrame:
    """Return pass/fail counts grouped by a category column."""
    if df.empty or category_col not in df.columns or pass_col not in df.columns:
        return pd.DataFrame()

    subset = df[[category_col, pass_col]].copy()
    subset[category_col] = subset[category_col].fillna("unknown").astype(str)
    subset[pass_col] = _normalize_bool_series(subset[pass_col])
    subset = subset.dropna(subset=[pass_col])
    if subset.empty:
        return pd.DataFrame()

    pivot = (
        subset.assign(result=subset[pass_col].map({True: "pass", False: "fail"}))
        .groupby([category_col, "result"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    for col in ["pass", "fail"]:
        if col not in pivot.columns:
            pivot[col] = 0
    return pivot[["pass", "fail"]]


def _format_rate(numerator: int, denominator: int) -> str:
    """Format a percentage rate for display."""
    if denominator <= 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def _safe_text(value: Any) -> str:
    """Convert a value to readable text for detail rendering."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _horizontal_bar_chart(
    data: pd.DataFrame | pd.Series,
    *,
    category_name: str,
    value_name: str = "count",
    title: str | None = None,
    stacked: bool = False,
) -> None:
    """Render a horizontal bar chart with categories on the y-axis."""
    if isinstance(data, pd.Series):
        if data.empty:
            st.info("No data")
            return
        plot_df = data.rename_axis(category_name).reset_index(name=value_name)
    else:
        if data.empty:
            st.info("No data")
            return
        plot_df = data.reset_index()
        if category_name not in plot_df.columns:
            plot_df = plot_df.rename(columns={plot_df.columns[0]: category_name})

    if stacked:
        long_df = plot_df.melt(
            id_vars=[category_name],
            var_name="result",
            value_name=value_name,
        )
        fig = px.bar(
            long_df,
            x=value_name,
            y=category_name,
            color="result",
            orientation="h",
            title=title,
            barmode="stack",
        )
    else:
        fig = px.bar(
            plot_df,
            x=value_name,
            y=category_name,
            orientation="h",
            title=title,
        )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Render the local eval dashboard."""
    st.set_page_config(page_title="bAIpacking Eval Dashboard", layout="wide")
    st.title("bAIpacking Eval Dashboard")
    st.caption("Local-only viewer for `data/eval/scenario_runs.jsonl`.")

    runs_path = Path(st.sidebar.text_input("Scenario runs JSONL", value=str(DEFAULT_RUNS_PATH)))
    refresh = st.sidebar.button("Reload data")

    df = _load_jsonl(runs_path)
    if df.empty:
        st.warning(f"No runs found at {runs_path}")
        st.stop()

    base_columns = [
        "scenario_id",
        "status",
        "failure_kind",
        "expected_event",
        "resolved_event_name",
        "expected_component",
        "policy_mode",
        "content_assertion_issue_count",
        "event_alignment_issue_count",
        "content_assertion_issues",
        "event_alignment_issues",
        "schema_error_paths",
        "error",
        "summary",
        "reasoning",
        "recommended_setup",
        "raw_trace_steps",
        "event_match_type",
        "expected_event_match_type",
    ]
    df = _ensure_columns(df, base_columns)

    with st.sidebar:
        st.subheader("Filters")
        status_options = ["all"] + sorted(df["status"].fillna("unknown").astype(str).unique().tolist())
        selected_status = st.selectbox("Status", status_options, index=0)

        failure_options = ["all"] + sorted(df["failure_kind"].fillna("none").astype(str).unique().tolist())
        selected_failure_kind = st.selectbox("Failure kind", failure_options, index=0)

        match_options = ["all"] + sorted(df["expected_event_match_type"].fillna("unknown").astype(str).unique().tolist())
        selected_match_type = st.selectbox("Expected event match type", match_options, index=0)

        component_options = ["all"] + sorted(df["expected_component"].fillna("unknown").astype(str).unique().tolist())
        selected_component = st.selectbox("Expected component", component_options, index=0)

        scenario_options = ["all"] + sorted(df["scenario_id"].fillna("unknown").astype(str).unique().tolist())
        selected_scenario = st.selectbox("Scenario id", scenario_options, index=0)

    filtered = df.copy()
    if selected_status != "all":
        filtered = filtered[filtered["status"].fillna("unknown").astype(str) == selected_status]
    if selected_failure_kind != "all":
        filtered = filtered[filtered["failure_kind"].fillna("none").astype(str) == selected_failure_kind]
    if selected_match_type != "all":
        filtered = filtered[filtered["expected_event_match_type"].fillna("unknown").astype(str) == selected_match_type]
    if selected_component != "all":
        filtered = filtered[filtered["expected_component"].fillna("unknown").astype(str) == selected_component]
    if selected_scenario != "all":
        filtered = filtered[filtered["scenario_id"].fillna("unknown").astype(str) == selected_scenario]

    if filtered.empty:
        st.warning("No rows match the current filters.")
        st.stop()

    total_runs = len(filtered)
    success_runs = int((filtered["status"].fillna("") == "success").sum())
    output_schema_failures = int((filtered["failure_kind"].fillna("") == "output_schema_failure").sum())
    content_pass_col = filtered["content_assertions_passed"] if "content_assertions_passed" in filtered.columns else pd.Series(dtype="object")
    event_pass_col = filtered["event_alignment_assertions_passed"] if "event_alignment_assertions_passed" in filtered.columns else pd.Series(dtype="object")
    content_pass_rate = _format_rate(int((content_pass_col == True).sum()), int(content_pass_col.notna().sum())) if not content_pass_col.empty else "n/a"
    event_pass_rate = _format_rate(int((event_pass_col == True).sum()), int(event_pass_col.notna().sum())) if not event_pass_col.empty else "n/a"
    success_rate = _format_rate(success_runs, total_runs)

    card_cols = st.columns(5)
    card_cols[0].metric("Total runs", total_runs)
    card_cols[1].metric("Success rate", success_rate)
    card_cols[2].metric("Output schema failures", output_schema_failures)
    card_cols[3].metric("Content assertion pass rate", content_pass_rate)
    card_cols[4].metric("Event alignment pass rate", event_pass_rate)

    st.subheader("Breakdowns")
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Failures by failure_kind")
        failures_by_kind = (
            filtered[filtered["status"].fillna("") == "failure"]["failure_kind"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        _horizontal_bar_chart(failures_by_kind, category_name="failure_kind")

        st.caption("Failures by scenario_id")
        failures_by_scenario = (
            filtered[filtered["status"].fillna("") == "failure"]["scenario_id"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .sort_values(ascending=False)
        )
        _horizontal_bar_chart(failures_by_scenario, category_name="scenario_id")

    with chart_cols[1]:
        st.caption("Issue counts by issue type")
        issue_counts = pd.concat(
            [
                _issue_counts(filtered, "content_assertion_issues"),
                _issue_counts(filtered, "event_alignment_issues"),
            ]
        ).groupby(level=0).sum().sort_values(ascending=False)
        _horizontal_bar_chart(issue_counts, category_name="issue_type")

        st.caption("Pass/fail by expected_event_match_type")
        event_match_breakdown = _pass_fail_by_category(
            filtered,
            "expected_event_match_type",
            "event_alignment_assertions_passed",
        )
        _horizontal_bar_chart(
            event_match_breakdown,
            category_name="expected_event_match_type",
            stacked=True,
        )

    st.caption("Pass/fail by expected_component")
    component_breakdown = _pass_fail_by_category(
        filtered,
        "expected_component",
        "content_assertions_passed",
    )
    _horizontal_bar_chart(
        component_breakdown,
        category_name="expected_component",
        stacked=True,
    )

    st.subheader("Drill-down table")
    table_columns = [
        "scenario_id",
        "status",
        "failure_kind",
        "expected_event",
        "resolved_event_name",
        "expected_component",
        "policy_mode",
        "content_assertion_issue_count",
        "event_alignment_issue_count",
    ]
    drilldown = filtered[table_columns].copy()
    st.dataframe(drilldown, use_container_width=True, hide_index=True)

    selected_index = st.selectbox(
        "Row details",
        options=list(filtered.index),
        format_func=lambda idx: f"{filtered.loc[idx, 'scenario_id']} | {filtered.loc[idx, 'status']} | {filtered.loc[idx, 'failure_kind']}",
    )
    row = filtered.loc[selected_index]

    st.subheader("Row details")
    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("**Error**")
        st.code(_safe_text(row.get("error")) or "(none)")
        st.markdown("**Schema error paths**")
        st.code(_safe_text(row.get("schema_error_paths")) or "(none)")
        st.markdown("**Summary**")
        st.write(_safe_text(row.get("summary")) or "(none)")
        st.markdown("**Reasoning**")
        st.write(_safe_text(row.get("reasoning")) or "(none)")

    with detail_cols[1]:
        st.markdown("**Recommended setup**")
        st.code(_safe_text(row.get("recommended_setup")) or "(none)")
        st.markdown("**Raw trace snippet**")
        raw_trace = _list_from_cell(row.get("raw_trace_steps"))
        if raw_trace:
            st.code(json.dumps(raw_trace[:3], ensure_ascii=False, indent=2))
        else:
            st.code("(none)")

    if refresh:
        st.rerun()


if __name__ == "__main__":
    main()