import streamlit as st

from ai_insights import get_ai_insights, render_summary


GUIDE_ITEMS = [
    {
        "tab": "Upload",
        "title": "Upload your file",
        "desc": "Drop a CSV or Excel file in the Upload tab. If your Excel has multiple sheets you will be asked to pick one. The app resets automatically when you load a new file.",
        "tips": [
            "CSV and XLSX files both work.",
            "Files only live in your browser session, nothing is stored.",
            "Upload a new file any time and the app resets cleanly.",
        ],
    },
    {
        "tab": "Overview",
        "title": "Check your quality score",
        "desc": "The Overview tab shows your overall quality score out of 100, broken into five dimensions: Completeness, Uniqueness, Type Consistency, Outlier Cleanliness, and Validity. Read this first before touching anything.",
        "tips": [
            "Score updates live every time you clean something.",
            "Red scores mean the biggest problems to fix first.",
            "Data Statistics below the score shows row counts, missing cells, and duplicate rows.",
        ],
    },
    {
        "tab": "Recommendations",
        "title": "Let the app find and fix issues",
        "desc": "The Recommendations tab scans your data automatically and lists every issue it finds. Each issue has a Select columns popover and a Fix button. You can fix them one by one or press Auto-Fix All at the bottom to resolve everything in one click.",
        "tips": [
            "Always start here before doing anything manually.",
            "Click Select columns first, then click Fix.",
            "Auto-Fix All runs all safe fixes in the correct order.",
            "Every fix is saved to History so you can undo it.",
        ],
    },
    {
        "tab": "Clean",
        "title": "Clean manually or with AI",
        "desc": "The Clean tab has Basic Cleaning buttons (Strip Whitespace, Drop Duplicate Rows, Drop Duplicate Cols, Clean String Edges), Advanced Cleaning (Smart Column Cleaner, Handle Missing Values), and tools for Find and Replace, Column Type Override, Split Column, Merge Columns, Rename Columns, and Data Type Guesser. At the very top is the AI Cleaner where you can type what you want in plain English.",
        "tips": [
            "AI Cleaner at the top handles complex requests in plain English.",
            "Data Type Guesser finds columns storing numbers or dates as text.",
            "Split Column is useful for things like a full name stored in one column.",
            "Every action is logged to History automatically.",
        ],
    },
    {
        "tab": "Validate",
        "title": "Validate formats and outliers",
        "desc": "The Validate tab checks correctness rather than cleanliness. It has Validate Email, Standardize Phone Numbers, Standardize Dates, Cap and Remove Outliers, and Validate Value Range. Run this after cleaning, not before.",
        "tips": [
            "Email validation flags bad formats and can remove those rows.",
            "Standardize Dates converts mixed date formats to one consistent format.",
            "Cap and Remove Outliers uses IQR or Z-score method.",
            "Validate Value Range flags or removes rows outside a min and max.",
        ],
    },
    {
        "tab": "Profile",
        "title": "Explore your data visually",
        "desc": "The Profile tab shows a Column Profiler with per-column stats, Distribution Charts for any column you pick, a Missing Value Heatmap, a Correlation Heatmap in Advanced mode, and a Before and After Comparison for every column.",
        "tips": [
            "Switch to Advanced mode in the sidebar to unlock the Correlation Heatmap.",
            "The heatmap shows exactly which rows and columns have missing values.",
            "Before and After Comparison shows what changed since you loaded the file.",
        ],
    },
    {
        "tab": "History & Export",
        "title": "Export and save your work",
        "desc": "The History and Export tab lets you download your cleaned data as CSV or Excel, export a PDF cleaning report, save your cleaning steps as a JSON pipeline to reuse on future files, reload a saved pipeline, export as a Python script, and reset the data back to the original.",
        "tips": [
            "Run at least one cleaning operation before exporting a report.",
            "Save Pipeline as JSON to replay the same steps on a new file later.",
            "Reset Data restores the original uploaded file if you want to start over.",
        ],
    },
]


def _init_guide():
    if "guide_checked" not in st.session_state:
        st.session_state.guide_checked = set()


def _render_ai_section(cdf, file_id):
    st.markdown("## AI Analysis")
    st.caption("AI has read your specific data and tells you exactly what to fix and how.")

    # never trigger the API call here — app.py handles that after all tabs render
    cache_key = f"ai_insights_{file_id}"
    insights = st.session_state.get(cache_key)

    if not insights:
        st.caption("AI analysis loading...")
        return

    summary = insights.get("summary", "")
    if summary:
        st.info(f"**What your data looks like:** {summary}")

    fixes = insights.get("fixes", [])
    if not fixes:
        return

    st.markdown("### What to fix and how")
    st.caption("Each card below tells you the problem, which tab to go to, what to click, and a shortcut using the AI Cleaner.")

    for fix in fixes:
        priority = fix.get("priority", "")
        issue    = fix.get("issue", "")
        column   = fix.get("column", "")
        reason   = fix.get("reason", "")
        tab      = fix.get("tab", "")
        action   = fix.get("action", "")
        shortcut = fix.get("shortcut", "")

        col_label = f"Column: **{column}**" if column and column != "ALL" else "Affects the **whole dataset**"

        st.markdown(
            f"""
<div style="background:#1a1a2e;border-left:4px solid #1f77b4;
            border-radius:10px;padding:16px 18px;margin-bottom:14px;">

  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="color:#fff;font-weight:800;font-size:1rem;">#{priority} &nbsp; {issue}</span>
    <span style="background:#1f77b4;color:#fff;font-size:0.7rem;font-weight:700;
                 padding:2px 10px;border-radius:99px;">{tab} tab</span>
  </div>

  <div style="color:#aaa;font-size:0.78rem;margin-bottom:10px;">{col_label}</div>

  <div style="color:#ccc;font-size:0.82rem;margin-bottom:10px;line-height:1.6;">
    <b style="color:#fff;">Why it matters:</b> {reason}
  </div>

  <div style="background:#0f0f23;border-radius:8px;padding:10px 14px;margin-bottom:8px;">
    <div style="color:#1f77b4;font-size:0.72rem;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
      How to fix it
    </div>
    <div style="color:#ddd;font-size:0.82rem;line-height:1.6;">
      Go to the <b style="color:#1f77b4;">{tab}</b> tab. {action}
    </div>
  </div>

  <div style="background:#0a1a0a;border-radius:8px;padding:10px 14px;">
    <div style="color:#22c55e;font-size:0.72rem;font-weight:700;
                text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
      AI Cleaner shortcut
    </div>
    <div style="color:#86efac;font-size:0.82rem;line-height:1.6;">
      {shortcut}
    </div>
  </div>

</div>
""",
            unsafe_allow_html=True,
        )


def _render_general_guide():
    _init_guide()
    checked = st.session_state.guide_checked

    st.markdown("## General Guide")
    st.caption("Work through each step at your own pace. Tick items off as you go.")

    done  = len(checked)
    total = len(GUIDE_ITEMS)
    pct   = int(done / total * 100)

    st.markdown(f"""
<div style="margin:16px 0 24px 0;">
  <div style="display:flex;justify-content:space-between;
              font-size:0.75rem;color:#888;margin-bottom:6px;">
    <span>{done} of {total} steps done</span>
    <span>{pct}%</span>
  </div>
  <div style="background:#1a1a2e;border-radius:99px;height:6px;overflow:hidden;">
    <div style="background:#1f77b4;width:{pct}%;height:100%;
                border-radius:99px;transition:width .3s;"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    for i, item in enumerate(GUIDE_ITEMS):
        is_checked = i in checked
        col_check, col_content = st.columns([0.06, 0.94])

        with col_check:
            ticked = st.checkbox(
                label="", value=is_checked,
                key=f"guide_check_{i}",
                label_visibility="collapsed",
            )
            if ticked and i not in checked:
                checked.add(i)
                st.session_state.guide_checked = checked
                st.rerun()
            elif not ticked and i in checked:
                checked.discard(i)
                st.session_state.guide_checked = checked
                st.rerun()

        with col_content:
            title_style = "text-decoration:line-through;color:#555;" if is_checked else "color:#fff;"
            tab_badge = (
                f'<span style="font-size:0.65rem;font-weight:700;color:#1f77b4;'
                f'text-transform:uppercase;letter-spacing:1px;margin-left:8px;">'
                f'{item["tab"]} tab</span>'
            )
            st.markdown(
                f'<p style="margin:0 0 4px 0;font-size:0.95rem;font-weight:700;{title_style}">'
                f'{item["title"]}{tab_badge}</p>',
                unsafe_allow_html=True,
            )
            st.caption(item["desc"])
            with st.expander("Tips"):
                for tip in item["tips"]:
                    st.markdown(f"- {tip}")

        st.divider()

    if checked:
        if st.button("Reset all checkboxes", key="guide_reset"):
            st.session_state.guide_checked = set()
            st.rerun()


def render(tab, cdf=None, file_id=None):
    with tab:
        st.markdown("# Guide")

        if cdf is not None and file_id is not None:
            _render_ai_section(cdf, file_id)
            st.divider()

        _render_general_guide()