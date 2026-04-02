from dotenv import load_dotenv
from state import init_state, maybe_reset_on_new_upload, render_sidebar, resolve_upload, handle_resume_loading_screen
load_dotenv()

import numpy as np
import streamlit as st

from config import apply_page_config, inject_css

apply_page_config()
inject_css()

from cache import get_dataframe_stats, load_file, make_df_key
from guide import render as render_guide
from state import init_state, maybe_reset_on_new_upload, render_sidebar, resolve_upload
from tabs import (
    clean,
    history_export,
    overview,
    profile,
    recommendations,
    upload,
    validate,
)

missing_threshold, numeric_strategy, conversion_threshold, mode = render_sidebar()

(
    tab_upload,
    tab_overview,
    tab_recommend,
    tab_clean,
    tab_validate,
    tab_profile,
    tab_history,
    tab_guide,
) = st.tabs([
    "  Upload  ",
    "  Overview  ",
    "  Recommendations  ",
    "  Clean  ",
    "  Validate  ",
    "  Profile  ",
    "  History and Export  ",
    "  Guide  ",
])

upload.render(tab_upload)

uploaded = st.session_state.get("uploader")
if handle_resume_loading_screen():
    st.stop()

if uploaded is None:
    render_guide(tab_guide)
    for tab in (tab_overview, tab_recommend, tab_clean, tab_validate, tab_profile, tab_history):
        with tab:
            st.info("Upload a file in the Upload tab to get started.")
else:
    try:
        file_id = uploaded.file_id
        maybe_reset_on_new_upload(file_id)

        uploaded.seek(0)

        # spinner wraps only the parse and init work
        # st.rerun is intentionally outside this block so the spinner
        # has a chance to render before the script reruns
        with st.spinner(f"Loading {uploaded.name}..."):
            file_bytes, selected_sheet = resolve_upload(uploaded)
            load_key = f"{file_id}_{selected_sheet}"
            df = load_file(file_bytes, uploaded.name, load_key, sheet_name=selected_sheet)
            init_state(df, load_key, file_bytes=file_bytes, filename=uploaded.name)

        # init_state calls st.stop() while the resume dialog is open
        # so current_df will not exist yet in that render pass
        # everything below is skipped until the user makes a choice and st.rerun fires
        if "current_df" not in st.session_state:
            st.stop()

        cdf = st.session_state.current_df
        df_key = st.session_state.get("current_df_key", make_df_key(cdf))
        stats = get_dataframe_stats(df_key, cdf)
        orig_stats = st.session_state.original_stats
        all_cols = list(cdf.columns)
        text_cols = list(cdf.select_dtypes(include="object").columns)
        num_cols = list(cdf.select_dtypes(include=np.number).columns)

        st.info(
            f"{uploaded.name}  |  {stats['rows']:,} rows x {stats['columns']} cols  |  "
            f"{stats['memory_usage']:.2f} MB"
        )

        render_guide(tab_guide, cdf=cdf, file_id=load_key)

        overview.render(tab_overview, cdf, stats, orig_stats, file_id=load_key, df_key=df_key)

        recommendations.render(
            tab_recommend,
            cdf,
            df_key=df_key,
            conversion_threshold=conversion_threshold,
            missing_threshold=missing_threshold,
            numeric_strategy=numeric_strategy,
        )

        clean.render(
            tab_clean,
            cdf,
            all_cols,
            df_key=df_key,
            missing_threshold=missing_threshold,
            numeric_strategy=numeric_strategy,
            conversion_threshold=conversion_threshold,
        )

        validate.render(tab_validate, cdf, text_cols, num_cols)

        profile.render(tab_profile, cdf, mode)

        pipeline_settings = {
            "missing_threshold": missing_threshold,
            "numeric_strategy": numeric_strategy,
            "conversion_threshold": conversion_threshold,
            "filename": uploaded.name,
        }
        history_export.render(tab_history, cdf, pipeline_settings)

        # ai insights are fetched on demand inside guide.py when the user opens the guide tab
        # nothing needs to run here at page load time

    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.info("Make sure your file is a valid CSV or Excel format.")

