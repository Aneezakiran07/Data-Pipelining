import pandas as pd
import streamlit as st

# operators available per column dtype
_TEXT_OPS = ["contains", "equals", "starts with", "ends with", "is empty", "is not empty"]
_NUM_OPS = ["equals", "greater than", "less than", "between", "is empty", "is not empty"]
_DATE_OPS = ["equals", "after", "before", "between", "is empty", "is not empty"]


def _dtype_ops(series):
    if pd.api.types.is_numeric_dtype(series):
        return _NUM_OPS, "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return _DATE_OPS, "datetime"
    return _TEXT_OPS, "text"


def _apply_filter(df, col, op, val1, val2, case_sensitive):
    s = df[col]
    kind = "numeric" if pd.api.types.is_numeric_dtype(s) else (
        "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "text"
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


def _add_filter_row(idx, all_cols, cdf):
    # renders one filter row and returns (col, op, val1, val2, case_sensitive) or None if removed
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
                    val1 = st.text_input("Min" if kind == "numeric" else "From", key=f"{fk}_v1", label_visibility="collapsed", placeholder="Min" if kind != "text" else "From")
                with vi2:
                    val2 = st.text_input("Max" if kind == "numeric" else "To", key=f"{fk}_v2", label_visibility="collapsed", placeholder="Max" if kind != "text" else "To")
            else:
                val1 = st.text_input("Value", key=f"{fk}_v1", label_visibility="collapsed", placeholder="Value...")

    with c4:
        removed = st.button("x", key=f"{fk}_remove", use_container_width=True)

    case_sensitive = False
    if kind == "text" and op not in ("is empty", "is not empty"):
        case_sensitive = st.checkbox("Case sensitive", key=f"{fk}_case", value=False)

    if removed:
        return None
    return col, op, val1, val2, case_sensitive


def render_filter_preview(cdf, all_cols):
    st.subheader("Filter and Inspect")
    st.caption(
        "Read-only view — filters never modify your data. "
        "Use this to spot dirty values before deciding which operation to run."
    )

    # initialise filter list in session state
    if "fp_filters" not in st.session_state:
        st.session_state.fp_filters = [0]
        st.session_state.fp_next_id = 1

    # render each filter row and collect results
    kept = []
    filters = []
    for idx in st.session_state.fp_filters:
        result = _add_filter_row(idx, all_cols, cdf)
        if result is not None:
            kept.append(idx)
            filters.append(result)
        # if result is None the user hit x so we drop this row

    st.session_state.fp_filters = kept

    a1, a2, a3 = st.columns([1, 1, 5])
    with a1:
        if st.button("+ Add filter", key="fp_add", use_container_width=True):
            st.session_state.fp_filters.append(st.session_state.fp_next_id)
            st.session_state.fp_next_id += 1
            st.rerun()
    with a2:
        if st.button("Clear all", key="fp_clear", use_container_width=True):
            st.session_state.fp_filters = [0]
            st.session_state.fp_next_id = 1
            # wipe stale filter widget keys so widgets reset cleanly
            stale = [k for k in st.session_state if k.startswith("_fp_")]
            for k in stale:
                del st.session_state[k]
            st.rerun()

    # apply all active filters in sequence
    filtered = cdf.copy()
    error_msg = None
    for col, op, val1, val2, case_sensitive in filters:
        # skip rows where the user hasnt typed a value yet for ops that need one
        if op not in ("is empty", "is not empty") and op != "between" and not val1.strip():
            continue
        if op == "between" and not val1.strip() and not val2.strip():
            continue
        filtered, err = _apply_filter(filtered, col, op, val1, val2, case_sensitive)
        if err:
            error_msg = err
            break

    if error_msg:
        st.warning(error_msg)

    total = len(cdf)
    shown = len(filtered)
    pct = (shown / total * 100) if total > 0 else 0

    # metrics row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Matching rows", f"{shown:,}")
    with m2:
        st.metric("Total rows", f"{total:,}")
    with m3:
        st.metric("Match rate", f"{pct:.1f}%")

    # cap display at 500 rows for performance — never touches cdf
    display_df = filtered.head(500)
    st.dataframe(display_df, use_container_width=True, hide_index=False)

    if shown > 500:
        st.caption(f"Showing first 500 of {shown:,} matching rows.")