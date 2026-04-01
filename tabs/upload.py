import pandas as pd
import streamlit as st


def render(tab):
    with tab:
        st.subheader("Advanced Data Cleaning Pipeline")
        st.caption(
            "Upload a CSV or Excel file to get started. "
            "[Test CSV on GitHub](https://github.com/Aneezakiran07/Data-Pipelining)"
        )
        st.write("")
        st.file_uploader(
            "",
            type=["csv", "xlsx", "xls"],
            key="uploader",
            label_visibility="collapsed",
        )

        uploaded = st.session_state.get("uploader")
        if uploaded is None:
            st.write("")
            st.subheader("Sample Data Format")
            st.dataframe(
                pd.DataFrame({
                    "name": [" Alice ", "Bob", "Charlie", "Alice"],
                    "price": ["$100", "$200.50", "300EUR", "$100"],
                    "percentage": ["75%", "80.5%", "99%", "75%"],
                    "weight": ["100kg", "150.5 lbs", "?", "100kg"],
                    "duration": ["1h30m", "90min", "NA", "1h30m"],
                }),
                use_container_width=True,
            )
            st.caption("The pipeline handles currency, percentages, units, missing values, and more.")
        else:
            # loaded_file_id is only set in session state after app.py fully
            # finishes init_state, so we can use it to distinguish between
            # the first render pass (file picked, not yet loaded) and done
            already_loaded = st.session_state.get("loaded_file_id") == uploaded.file_id

            if already_loaded:
                st.success(f"{uploaded.name} is loaded. Navigate to any tab to start cleaning.")
            else:
                # this shows on the very first rerun after the user picks the file
                # before app.py has had a chance to read and parse it
                # gives immediate visual confirmation that the upload was accepted
                st.info(f"Reading {uploaded.name}, please wait...")
