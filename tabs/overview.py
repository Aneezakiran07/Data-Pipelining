import pandas as pd
import streamlit as st

from ai_insights import render_summary
from cache import get_quality_score


def _score_color(score):
    if score >= 80:
        return "#16a34a"
    if score >= 55:
        return "#d97706"
    return "#dc2626"


def _grade_color(grade):
    return {"good": "#16a34a", "fair": "#d97706", "poor": "#dc2626"}.get(grade, "#888888")


def _render_quality_score(cdf):
    result = get_quality_score(cdf)
    total = result["total"]
    breakdown = result["breakdown"]
    color = _score_color(total)
    label = "excellent" if total >= 80 else "needs work" if total >= 55 else "poor"

    gauge_html = f"""
    <div style="display:flex; align-items:center; gap:32px; margin-bottom:8px;">
        <div style="position:relative; width:130px; height:130px;">
            <svg viewBox="0 0 36 36" width="130" height="130">
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke="#e5e7eb" stroke-width="3.5" stroke-dasharray="100, 100"/>
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke="{color}" stroke-width="3.5"
                    stroke-dasharray="{total}, 100" stroke-linecap="round"/>
                <text x="18" y="17" text-anchor="middle" font-size="8" font-weight="bold" fill="{color}">{total}</text>
                <text x="18" y="23" text-anchor="middle" font-size="3.2" fill="#888">/100</text>
            </svg>
        </div>
        <div>
            <div style="font-size:2rem; font-weight:800; color:{color};">{total}/100</div>
            <div style="font-size:1rem; color:#888; margin-top:2px;">quality score: {label}</div>
            <div style="font-size:0.8rem; color:#aaa; margin-top:4px;">updates live as you clean</div>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (dim, data) in enumerate(breakdown.items()):
        gc = _grade_color(data["grade"])
        with cols[i]:
            st.markdown(
                f"""<div style="background:#1a1a2e; border-radius:8px; padding:12px 10px;
                            border-left:4px solid {gc}; min-height:110px;">
                    <div style="font-size:0.75rem; color:#aaa; margin-bottom:4px;">{dim}</div>
                    <div style="font-size:1.5rem; font-weight:700; color:{gc};">
                        {data['score']}<span style="font-size:0.8rem; color:#888;">/{data['max']}</span>
                    </div>
                    <div style="font-size:0.7rem; color:#888; margin-top:6px; line-height:1.4;">{data['detail']}</div>
                </div>""",
                unsafe_allow_html=True,
            )


def render(tab, cdf, stats, orig_stats, file_id=None):
    with tab:
        if file_id:
            render_summary(cdf, file_id)
            st.divider()

        st.subheader("Data Quality Score")
        _render_quality_score(cdf)

        st.divider()
        st.subheader("Data Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", stats["rows"], delta=stats["rows"] - orig_stats["rows"], delta_color="inverse")
            st.metric("Columns", stats["columns"], delta=stats["columns"] - orig_stats["columns"], delta_color="inverse")
        with c2:
            st.metric("Missing Cells", stats["missing_cells"], delta=stats["missing_cells"] - orig_stats["missing_cells"], delta_color="inverse")
            st.metric("Duplicate Rows", stats["duplicate_rows"], delta=stats["duplicate_rows"] - orig_stats["duplicate_rows"], delta_color="inverse")
        with c3:
            st.metric("Numeric Columns", stats["numeric_cols"])
            st.metric("Categorical Columns", stats["categorical_cols"])

        st.divider()
        st.subheader("Data Preview")
        total_rows = len(cdf)
        max_preview = min(50, total_rows)
        default_preview = min(10, total_rows)
        n_prev = st.slider("Rows to display", min(5, total_rows), max_preview, default_preview, key="prev_slider")
        if total_rows < 50:
            st.caption(f"File has {total_rows} rows.")
        st.dataframe(cdf.head(n_prev), use_container_width=True)

        st.divider()
        with st.expander("Column Types and Info", expanded=False):
            st.dataframe(
                pd.DataFrame({
                    "Column": cdf.columns,
                    "Type": cdf.dtypes.values,
                    "Non-Null": cdf.count().values,
                    "Null": cdf.isna().sum().values,
                    "Unique": cdf.nunique().values,
                }),
                use_container_width=True,
            )
        st.caption("Download and reset options are in the History and Export tab.")

        