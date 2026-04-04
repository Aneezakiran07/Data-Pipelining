import io

import pandas as pd
import streamlit as st


_TEXT_OPS = ["contains", "equals", "starts with", "ends with", "is empty", "is not empty"]
_NUM_OPS = ["equals", "greater than", "less than", "between", "is empty", "is not empty"]
_DATE_OPS = ["equals", "after", "before", "between", "is empty", "is not empty"]


def _dtype_ops(series):
    if pd.api.types.is_numeric_dtype(series):
        return _NUM_OPS, "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return _DATE_OPS, "datetime"
    return _TEXT_OPS, "text"


def _apply_single_filter(df, col, op, val1, val2, case_sensitive):
    s = df[col]
    kind = (
        "numeric" if pd.api.types.is_numeric_dtype(s)
        else "datetime" if pd.api.types.is_datetime64_any_dtype(s)
        else "text"
    )

    if op == "is empty":
        mask = s.isna() | (s.astype(str).str.strip() == "")
    elif op == "is not empty":
        mask = ~(s.isna() | (s.astype(str).str.strip() == ""))

    elif kind == "text":
        sv = s.astype(str)
        v = val1 if case_sensitive else val1.lower()
        col_v = sv if case_sensitive else sv.str.lower()
        if op == "contains":
            mask = col_v.str.contains(v, regex=False, na=False)
        elif op == "equals":
            mask = col_v == v
        elif op == "starts with":
            mask = col_v.str.startswith(v, na=False)
        elif op == "ends with":
            mask = col_v.str.endswith(v, na=False)
        else:
            mask = pd.Series(True, index=df.index)

    elif kind == "numeric":
        try:
            n1 = float(val1) if val1 else None
            n2 = float(val2) if val2 else None
        except ValueError:
            return df.head(0), "Value must be a number."
        if op == "equals":
            mask = s == n1
        elif op == "greater than":
            mask = s > n1
        elif op == "less than":
            mask = s < n1
        elif op == "between":
            if n1 is None or n2 is None:
                return df.head(0), "Enter both Min and Max for between."
            mask = s.between(n1, n2, inclusive="both")
        else:
            mask = pd.Series(True, index=df.index)

    elif kind == "datetime":
        try:
            d1 = pd.to_datetime(val1) if val1 else None
            d2 = pd.to_datetime(val2) if val2 else None
        except Exception:
            return df.head(0), "Could not parse date. Try YYYY-MM-DD."
        if op == "equals":
            mask = s == d1
        elif op == "after":
            mask = s > d1
        elif op == "before":
            mask = s < d1
        elif op == "between":
            if d1 is None or d2 is None:
                return df.head(0), "Enter both dates for between."
            mask = s.between(d1, d2, inclusive="both")
        else:
            mask = pd.Series(True, index=df.index)

    else:
        mask = pd.Series(True, index=df.index)

    return df[mask], None


def _apply_all_filters(cdf, filters, match_mode):
    if not filters:
        return cdf.copy(), None

    if match_mode == "ALL":
        result = cdf.copy()
        for col, op, val1, val2, case_sensitive in filters:
            result, err = _apply_single_filter(result, col, op, val1, val2, case_sensitive)
            if err:
                return cdf.head(0), err
        return result, None

    else:
        combined_mask = pd.Series(False, index=cdf.index)
        for col, op, val1, val2, case_sensitive in filters:
            filtered, err = _apply_single_filter(cdf, col, op, val1, val2, case_sensitive)
            if err:
                return cdf.head(0), err
            row_mask = pd.Series(False, index=cdf.index)
            row_mask.loc[filtered.index] = True
            combined_mask = combined_mask | row_mask
        return cdf[combined_mask], None


def _add_filter_row(idx, all_cols, cdf):
    fk = f"_fp_{idx}"
    c1, c2, c3, c4 = st.columns([2, 1.8, 2, 0.5])

    with c1:
        col = st.selectbox("Column", all_cols, key=f"{fk}_col")
    ops, kind = _dtype_ops(cdf[col])
    with c2:
        op = st.selectbox("Condition", ops, key=f"{fk}_op")

    val1, val2 = "", ""
    with c3:
        if op not in ("is empty", "is not empty"):
            if op == "between":
                vi1, vi2 = st.columns(2)
                with vi1:
                    val1 = st.text_input(
                        "From", key=f"{fk}_v1",
                        label_visibility="visible",
                        placeholder="Min" if kind != "text" else "From",
                    )
                with vi2:
                    val2 = st.text_input(
                        "To", key=f"{fk}_v2",
                        label_visibility="visible",
                        placeholder="Max" if kind != "text" else "To",
                    )
            else:
                val1 = st.text_input(
                    "Value", key=f"{fk}_v1",
                    label_visibility="visible",
                    placeholder="Value...",
                )
        else:
            st.write(" ")

    with c4:
        st.write(" ")
        removed = st.button("x", key=f"{fk}_remove", use_container_width=True)

    case_sensitive = False
    if kind == "text" and op not in ("is empty", "is not empty"):
        case_sensitive = st.checkbox("Case sensitive", key=f"{fk}_case", value=False)

    if removed:
        return None
    return col, op, val1, val2, case_sensitive


def _is_filter_active(col, op, val1, val2):
    if op in ("is empty", "is not empty"):
        return True
    if op == "between":
        return bool(val1.strip() or val2.strip())
    return bool(val1.strip())


def render_filter_preview(cdf, all_cols):
    st.subheader("Filter and Inspect")
    st.caption(
        "Read-only view. Filters never modify your data. "
        "Use this to spot dirty values before deciding which operation to run."
    )

    if "fp_filters" not in st.session_state:
        st.session_state.fp_filters = [0]
        st.session_state.fp_next_id = 1

    match_mode = st.radio(
        "Filter mode",
        ["ALL (AND)", "ANY (OR)"],
        key="fp_match_mode",
        horizontal=True,
        help=(
            "ALL (AND): a row must match every filter to be shown. "
            "Use this to narrow down e.g. show rows where email is empty AND salary is above 50000.\n\n"
            "ANY (OR): a row only needs to match one filter to be shown. "
            "Use this to find all problem rows at once e.g. show rows where email is empty OR phone is empty."
        ),
    )
    match_mode = "ALL" if "ALL" in match_mode else "ANY"

    st.divider()

    kept = []
    filters = []
    for idx in st.session_state.fp_filters:
        result = _add_filter_row(idx, all_cols, cdf)
        if result is not None:
            kept.append(idx)
            col, op, val1, val2, case_sensitive = result
            if _is_filter_active(col, op, val1, val2):
                filters.append(result)

    st.session_state.fp_filters = kept

    if len(st.session_state.fp_filters) > 1:
        connector = "AND every filter above must match" if match_mode == "ALL" else "OR any filter above can match"
        st.caption(f"Mode: {connector}")

    a1, a2, _ = st.columns([1, 1, 5])
    with a1:
        if st.button(
            "+ Add filter", key="fp_add", use_container_width=True,
            help="Add another filter row. Use Filter Mode above to control whether rows must match all filters or just one.",
        ):
            st.session_state.fp_filters.append(st.session_state.fp_next_id)
            st.session_state.fp_next_id += 1
            st.rerun()
    with a2:
        if st.button(
            "Clear all", key="fp_clear", use_container_width=True,
            help="Remove all filters and reset the view to the full dataset.",
        ):
            st.session_state.fp_filters = [0]
            st.session_state.fp_next_id = 1
            stale = [k for k in st.session_state if k.startswith("_fp_")]
            for k in stale:
                del st.session_state[k]
            st.rerun()

    st.divider()

    filtered, error_msg = _apply_all_filters(cdf, filters, match_mode)

    if error_msg:
        st.warning(error_msg)

    total = len(cdf)
    shown = len(filtered)
    pct = (shown / total * 100) if total > 0 else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            "Matching rows", f"{shown:,}",
            help="Number of rows in your dataset that match the active filters.",
        )
    with m2:
        st.metric(
            "Total rows", f"{total:,}",
            help="Total rows in the current dataset before any filtering.",
        )
    with m3:
        st.metric(
            "Match rate", f"{pct:.1f}%",
            help="Percentage of total rows that match the active filters.",
        )

    display_df = filtered.head(500)
    st.dataframe(display_df, use_container_width=True, hide_index=False)

    if shown > 500:
        st.caption(f"Showing first 500 of {shown:,} matching rows. Export below downloads all {shown:,}.")

    if shown > 0:
        st.write("**Export filtered results**")
        st.caption(
            "Downloads the full filtered result: all matching rows, not just the 500 shown above.",
            help="Use this to save suspicious or interesting rows for further investigation outside the app.",
        )

        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "Download as CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="filtered_results.csv",
                mime="text/csv",
                key="fp_dl_csv",
                use_container_width=True,
                help="Download all matching rows as a CSV file. Opens in Excel, Google Sheets, or any text editor.",
            )
        with ex2:
            excel_key = f"fp_excel_{shown}_{match_mode}_{len(filters)}"
            if st.session_state.get("fp_excel_key") != excel_key:
                st.session_state.pop("fp_excel_bytes", None)

            if st.session_state.get("fp_excel_bytes") is None:
                if st.button(
                    "Prepare Excel Download",
                    key="fp_prep_excel",
                    use_container_width=True,
                    help="Builds an Excel file from the filtered rows. Click once to prepare, then download.",
                ):
                    with st.spinner(f"Building Excel file for {shown:,} rows..."):
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="openpyxl") as w:
                            filtered.to_excel(w, index=False, sheet_name="Filtered Results")
                        st.session_state["fp_excel_bytes"] = buf.getvalue()
                        st.session_state["fp_excel_key"] = excel_key
                    st.rerun()
            else:
                st.download_button(
                    "Download as Excel",
                    data=st.session_state["fp_excel_bytes"],
                    file_name="filtered_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="fp_dl_excel",
                    use_container_width=True,
                    help="Download all matching rows as an Excel file.",
                )
    else:
        st.caption("No matching rows to export.")