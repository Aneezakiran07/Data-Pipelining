import numpy as np
import streamlit as st

from cache import get_analysis_and_recommendations
from cleaning import (
    _convert_duration_val,
    clean_string_edges,
    drop_duplicate_columns,
    drop_duplicate_rows,
    missing_value_handler,
    smart_column_cleaner,
    stripping_whitespace,
)
from pipeline import commit_history, snapshot
from state import show_msg


def _rec_col_popover(action_key, af_cols):
    n_sel = len(st.session_state.selected_columns.get(action_key, []))

    def _rall(ak, cols):
        def h():
            checked = st.session_state.get(f"_ra_{ak}", False)
            st.session_state.selected_columns[ak] = cols.copy() if checked else []
        return h

    def _rcol(ak, col):
        def h():
            sel = st.session_state.selected_columns.get(ak, [])
            if st.session_state.get(f"_rc_{ak}_{col}"):
                if col not in sel:
                    st.session_state.selected_columns[ak] = sel + [col]
            else:
                st.session_state.selected_columns[ak] = [c for c in sel if c != col]
        return h

    with st.popover(f"{n_sel} selected" if n_sel else "Select columns", use_container_width=True):
        st.checkbox("All", key=f"_ra_{action_key}", on_change=_rall(action_key, af_cols))
        for c in af_cols:
            st.checkbox(c, key=f"_rc_{action_key}_{c}", on_change=_rcol(action_key, c))
    return n_sel


def _apply_fix(action_key, sel_cols, cdf, missing_threshold, numeric_strategy):
    tmp = cdf.copy()

    if action_key == "strip_whitespace":
        for c in sel_cols:
            if tmp[c].dtype == "object":
                tmp[c] = tmp[c].str.strip()

    elif action_key in ["convert_currency", "convert_percentage", "convert_units", "convert_duration"]:
        for c in sel_cols:
            if c not in tmp.columns:
                continue
            ne = tmp[c].astype(str).str.strip().replace("", np.nan).dropna()
            if ne.empty:
                continue
            if action_key == "convert_currency":
                cl = (
                    ne.str.replace(r"[^\d.,\-()]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                    .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                    .str.replace(",", "", regex=False)
                )
                converted = __import__("pandas").to_numeric(cl, errors="coerce")
                tmp[c] = converted.reindex(tmp.index)
            elif action_key == "convert_percentage":
                converted = (
                    __import__("pandas")
                    .to_numeric(
                        ne.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True),
                        errors="coerce",
                    )
                    / 100
                )
                tmp[c] = converted.reindex(tmp.index)
            elif action_key == "convert_units":
                converted = __import__("pandas").to_numeric(
                    ne.str.extract(r"([-]?\d+\.?\d*)", expand=False), errors="coerce"
                )
                tmp[c] = converted.reindex(tmp.index)
            elif action_key == "convert_duration":
                tmp[c] = ne.apply(_convert_duration_val).reindex(tmp.index)

    elif action_key == "clean_edges":
        for c in sel_cols:
            if tmp[c].dtype == "object":
                tmp[c] = (
                    tmp[c].astype(str)
                    .str.replace(r"^\W+", "", regex=True)
                    .str.replace(r"\W+$", "", regex=True)
                )

    elif action_key == "handle_missing":
        vc = [c for c in sel_cols if c in tmp.columns]
        if vc:
            sub = missing_value_handler(
                tmp[vc].copy(), threshold=missing_threshold, numeric_strategy=numeric_strategy
            )
            for c in vc:
                if c in sub.columns:
                    tmp[c] = sub[c]

    return tmp


def _render_recommendations(cdf, conversion_threshold, missing_threshold, numeric_strategy, df_key):
    # fire any pending toast from the previous rerun
    if "_toast" in st.session_state:
        msg, icon = st.session_state.pop("_toast")
        st.toast(msg, icon=icon)

    show_msg()
    st.subheader("Smart Recommendations")

    with st.spinner("Scanning your data for issues..."):
        issues, recs = get_analysis_and_recommendations(df_key, cdf, conversion_threshold)

    if not recs:
        st.success("Your data looks clean. No issues detected.")
        return

    if issues.get("is_sampled"):
        st.caption(
            f"Analysis ran on a {issues['sample_size']:,} row sample for performance. "
            "Recommendations still cover all columns."
        )

    st.warning(f"Found {len(recs)} potential issue(s)")
    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = {}

    for icon, title, desc, action_key in recs:
        af_cols = {
            "strip_whitespace": issues["whitespace_cols"],
            "convert_currency": issues["currency_cols"],
            "convert_percentage": issues["percentage_cols"],
            "convert_units": issues["unit_cols"],
            "convert_duration": issues["duration_cols"],
            "clean_edges": issues["edge_char_cols"],
            "handle_missing": [c for c, _ in issues["missing_cols"]],
        }.get(action_key, [])

        if af_cols:
            r1, r2, r3 = st.columns([5, 1.4, 1])
            with r1:
                st.write(f"**{title}**")
                st.caption(desc)
            with r2:
                n_sel = _rec_col_popover(action_key, af_cols)
            with r3:
                if st.button(
                    "Fix",
                    key=f"fix_{action_key}",
                    disabled=n_sel == 0,
                    type="primary" if n_sel else "secondary",
                    use_container_width=True,
                ):
                    sel_cols = st.session_state.selected_columns.get(action_key, [])
                    try:
                        _snap = snapshot()
                        with st.spinner(f"Fixing {action_key.replace('_', ' ')} in {len(sel_cols)} column(s)..."):
                            tmp = _apply_fix(action_key, sel_cols, cdf, missing_threshold, numeric_strategy)
                        st.session_state.current_df = tmp
                        commit_history(f"Fix: {action_key}", _snap)
                        st.session_state.selected_columns.pop(action_key, None)
                        st.session_state["_toast"] = (
                            f"Fixed {action_key.replace('_', ' ')} on {len(sel_cols)} column(s).", "✅"
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            r1, r2 = st.columns([5, 1])
            with r1:
                st.write(f"**{title}**")
                st.caption(desc)
            with r2:
                if st.button("Fix", key=f"fix_{action_key}", type="primary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        if action_key == "drop_duplicates":
                            with st.spinner("Removing duplicate rows..."):
                                tmp = drop_duplicate_rows(cdf)
                            st.session_state.current_df = tmp
                            commit_history(f"Fix: {action_key}", _snap)
                            st.session_state["_toast"] = ("Duplicate rows removed.", "✅")
                        elif action_key == "drop_dup_cols":
                            with st.spinner("Removing duplicate columns..."):
                                tmp = drop_duplicate_columns(cdf)
                            st.session_state.current_df = tmp
                            commit_history(f"Fix: {action_key}", _snap)
                            st.session_state["_toast"] = ("Duplicate columns removed.", "✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
        st.write("")

    st.divider()
    if st.button("Auto-Fix All Issues", key="auto_fix_all", type="primary", use_container_width=True):
        try:
            _snap = snapshot()
            with st.spinner("Running full auto-fix pipeline..."):
                tmp = stripping_whitespace(cdf)
                tmp = drop_duplicate_rows(tmp)
                tmp = drop_duplicate_columns(tmp)
                tmp = clean_string_edges(tmp, threshold=0.7)
                tmp = smart_column_cleaner(tmp, conversion_threshold=conversion_threshold)
                tmp = missing_value_handler(tmp, threshold=missing_threshold, numeric_strategy=numeric_strategy)
            st.session_state.current_df = tmp
            commit_history("Auto-Fix All", _snap)
            st.session_state.selected_columns = {}
            st.session_state["_toast"] = ("All issues fixed automatically.", "✅")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")


def render(tab, cdf, conversion_threshold, missing_threshold, numeric_strategy, df_key=""):
    with tab:
        # stable key — not tied to df_key so it survives after fixes update current_df
        scan_key = "rec_scanned"

        if not st.session_state.get(scan_key):
            st.subheader("Smart Recommendations")
            st.write("Scans your data for common issues like dirty columns, duplicates, and missing values.")
            st.write("This scan can take a few seconds on large files.")
            if st.button("Scan for Issues", key="rec_scan_btn", type="primary", use_container_width=True):
                st.session_state[scan_key] = True
                st.rerun()
        else:
            _render_recommendations(cdf, conversion_threshold, missing_threshold, numeric_strategy, df_key)
            st.write("")
            if st.button("Re-scan", key="rec_rescan_btn", use_container_width=True):
                # stay in scanned state — just bust the analysis cache so
                # _render_recommendations re-runs the scan immediately on next render
                from cache import get_analysis_and_recommendations
                get_analysis_and_recommendations.clear()
                st.rerun()