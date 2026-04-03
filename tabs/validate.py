import streamlit as st

from cleaning import cap_outliers, validate_date_col, validate_email_col, validate_phone_col, validate_range
from pipeline import commit_history, snapshot
from state import col_popover


def render(tab, cdf, text_cols, num_cols, df_key=""):
    with tab:
        st.subheader("Validation and Quality")

        # fire pending toast from previous rerun
        if "_toast" in st.session_state:
            msg, icon = st.session_state.pop("_toast")
            st.toast(msg, icon=icon)

        if text_cols:
            st.write("**Validate Email**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                st.caption("Flag adds a boolean column. Remove drops invalid rows.")
                ea = st.radio(
                    "", ["Flag invalid", "Remove invalid rows"], key="email_radio",
                    horizontal=True, label_visibility="collapsed",
                )
                custom_email_pattern = st.text_input(
                    "Custom regex pattern (optional)",
                    key="email_custom_pattern",
                    placeholder=r"e.g. ^[\w]+@company\.com$",
                    help=(
                        "Leave blank to use the built-in standard email check "
                        "(anything@anything.tld).\n\n"
                        "Fill this in only if you want stricter rules. A regex is a "
                        "pattern that describes what valid text looks like. Here are "
                        "real examples you can copy:\n\n"
                        "Only allow your company domain:\n"
                        r"^[\w\.\-]+@acme\.com$"
                        "\n\nReplace acme with your actual domain name. "
                        "The ^ means start of value, the $ means end of value, "
                        r"[\w\.\-]+ means one or more letters/numbers/dots/dashes "
                        "before the @ sign.\n\n"
                        "Only allow gmail or outlook:\n"
                        r"^[\w\.\-]+@(gmail|outlook)\.com$"
                        "\n\nThe (gmail|outlook) part means either one is accepted."
                    ),
                )
            with v2:
                n_em = col_popover("email", text_cols)
            with v3:
                if st.button("Run", key="run_email", disabled=n_em == 0,
                             type="primary" if n_em else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        cols = st.session_state.val_selected.get("email", [])
                        action = "flag" if "Flag" in ea else "remove"
                        with st.spinner(f"Scanning {len(cdf):,} rows — validating emails in {len(cols)} column(s)..."):
                            tmp = cdf.copy()
                            for c in cols:
                                tmp = validate_email_col(tmp, c, action, custom_pattern=custom_email_pattern)
                        st.session_state.current_df = tmp
                        commit_history("Validate Email", _snap)
                        st.session_state.val_selected.pop("email", None)
                        st.session_state["_toast"] = (f"Email validation done on {n_em} column(s).", "✔")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            st.divider()
            st.write("**Standardize Phone Numbers**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                st.caption("Strips non-digit characters and formats to +[country code][number].")
                phone_cc = st.text_input(
                    "Default country code (optional)",
                    key="phone_country_code",
                    value="1",
                    placeholder="e.g. 44",
                    help=(
                        "This is the number that gets added in front of any phone "
                        "number that has no country code already.\n\n"
                        "A country code is the digits you dial before the local number "
                        "when calling internationally. Do NOT include the + sign here, "
                        "just the digits.\n\n"
                        "Common codes:\n"
                        "1 = USA and Canada\n"
                        "44 = United Kingdom\n"
                        "92 = Pakistan\n"
                        "91 = India\n"
                        "61 = Australia\n"
                        "49 = Germany\n\n"
                        "How it works: if your data has 10-digit numbers like "
                        "3001234567 and you enter 92, the result will be +923001234567. "
                        "Numbers that already start with a country code are left alone."
                    ),
                )
            with v2:
                n_ph = col_popover("phone", text_cols)
            with v3:
                if st.button("Run", key="run_phone", disabled=n_ph == 0,
                             type="primary" if n_ph else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        cols = st.session_state.val_selected.get("phone", [])
                        with st.spinner(f"Scanning {len(cdf):,} rows — standardizing phones in {len(cols)} column(s)..."):
                            tmp = cdf.copy()
                            for c in cols:
                                tmp = validate_phone_col(tmp, c, default_country_code=phone_cc)
                        st.session_state.current_df = tmp
                        commit_history("Standardize Phone", _snap)
                        st.session_state.val_selected.pop("phone", None)
                        st.session_state["_toast"] = (f"Phone standardized in {n_ph} column(s).", "✔")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            st.divider()
            st.write("**Standardize Dates**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                date_fmt = st.selectbox(
                    "Output format",
                    ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"],
                    key="date_fmt",
                )
                custom_date_fmt = st.text_input(
                    "Custom input format (optional)",
                    key="date_custom_fmt",
                    placeholder="e.g. %Y.%m.%d or %d %b %Y %H:%M",
                    help=(
                        "Leave blank and the tool will automatically detect the date "
                        "format in your column — it tries 14 common patterns on its own.\n\n"
                        "Only fill this in if your dates look unusual and the tool is "
                        "failing to parse them correctly.\n\n"
                        "The format uses special codes where:\n"
                        "%Y = 4-digit year (e.g. 2024)\n"
                        "%y = 2-digit year (e.g. 24)\n"
                        "%m = month as a number (e.g. 03)\n"
                        "%d = day as a number (e.g. 07)\n"
                        "%H = hour in 24h format (e.g. 14)\n"
                        "%M = minutes (e.g. 30)\n"
                        "%b = short month name (e.g. Jan)\n"
                        "%B = full month name (e.g. January)\n\n"
                        "Real examples:\n"
                        "2024.03.07  →  %Y.%m.%d\n"
                        "07 Mar 2024 14:30  →  %d %b %Y %H:%M\n"
                        "March 7, 2024  →  %B %d, %Y"
                    ),
                )
            with v2:
                n_dt = col_popover("date", text_cols)
            with v3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_date", disabled=n_dt == 0,
                             type="primary" if n_dt else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        cols = st.session_state.val_selected.get("date", [])
                        with st.spinner(f"Scanning {len(cdf):,} rows — formatting dates in {len(cols)} column(s)..."):
                            tmp = cdf.copy()
                            for c in cols:
                                tmp = validate_date_col(
                                    tmp, c,
                                    output_format=date_fmt,
                                    custom_input_format=custom_date_fmt,
                                )
                        st.session_state.current_df = tmp
                        commit_history("Standardize Dates", _snap)
                        st.session_state.val_selected.pop("date", None)
                        st.session_state["_toast"] = (f"Dates standardized in {n_dt} column(s).", "✔")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if num_cols:
            st.divider()
            st.write("**Cap and Remove Outliers**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                o1, o2, o3 = st.columns(3)
                with o1:
                    o_method = st.selectbox("Method", ["iqr", "zscore"], key="o_method")
                with o2:
                    o_action = st.selectbox("Action", ["cap", "remove"], key="o_action")
                with o3:
                    o_thresh = st.number_input(
                        "Threshold", min_value=0.5, max_value=10.0, value=1.5, step=0.5, key="o_thresh"
                    )
            with v2:
                n_out = col_popover("outlier", num_cols)
            with v3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_outlier", disabled=n_out == 0,
                             type="primary" if n_out else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        cols = st.session_state.val_selected.get("outlier", [])
                        action_label = "Capping" if o_action == "cap" else "Removing"
                        with st.spinner(f"Scanning {len(cdf):,} rows — {action_label.lower()} outliers in {len(cols)} column(s) using {o_method.upper()}..."):
                            before = len(cdf)
                            tmp = cdf.copy()
                            for c in cols:
                                tmp = cap_outliers(tmp, c, method=o_method, action=o_action, threshold=o_thresh)
                        after = len(tmp)
                        st.session_state.current_df = tmp
                        commit_history("Cap Outliers", _snap)
                        st.session_state.val_selected.pop("outlier", None)
                        msg = (
                            f"Outliers capped in {n_out} column(s)."
                            if o_action == "cap"
                            else f"Removed {before - after:,} outlier rows."
                        )
                        st.session_state["_toast"] = (msg, "✔")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

            st.divider()
            st.write("**Validate Value Range**")
            v1, v2, v3 = st.columns([5, 1.4, 1])
            with v1:
                r1, r2, r3 = st.columns(3)
                with r1:
                    rng_min = st.number_input("Min", value=0.0, key="rng_min")
                with r2:
                    rng_max = st.number_input("Max", value=100.0, key="rng_max")
                with r3:
                    rng_act = st.selectbox("Action", ["flag", "remove"], key="rng_act")
            with v2:
                n_rng = col_popover("range", num_cols)
            with v3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_range", disabled=n_rng == 0,
                             type="primary" if n_rng else "secondary", use_container_width=True):
                    try:
                        _snap = snapshot()
                        cols = st.session_state.val_selected.get("range", [])
                        with st.spinner(f"Scanning {len(cdf):,} rows — checking range [{rng_min}, {rng_max}] in {len(cols)} column(s)..."):
                            before = len(cdf)
                            tmp = cdf.copy()
                            for c in cols:
                                tmp = validate_range(tmp, c, rng_min, rng_max, rng_act)
                        after = len(tmp)
                        st.session_state.current_df = tmp
                        commit_history("Validate Range", _snap)
                        st.session_state.val_selected.pop("range", None)
                        msg = (
                            f"Range flagged across {n_rng} column(s)."
                            if rng_act == "flag"
                            else f"Removed {before - after:,} out of range rows."
                        )
                        st.session_state["_toast"] = (msg, "✔")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

                   
