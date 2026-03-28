#not used anywhere cuz its not as efficient as the guide tab
import streamlit as st


STEPS_NO_FILE = [
    {
        "step": 1,
        "total": 1,
        "tab": "Upload",
        "title": "upload a file to get started",
        "body": "go to the Upload tab and drop in a CSV or Excel file. once your file is loaded the tour will walk you through the full cleaning workflow.",
    },
]

STEPS = [
    {
        "step": 1,
        "total": 6,
        "tab": "Overview",
        "title": "start by looking at your data",
        "body": "open the Overview tab. check your quality score, row and column counts, missing cells, and the data preview. get familiar with what you are working with before cleaning anything.",
    },
    {
        "step": 2,
        "total": 6,
        "tab": "Recommendations",
        "title": "let the app find issues for you",
        "body": "go to the Recommendations tab. we already scanned your data and flagged every issue. press Auto-Fix All to resolve everything in one click, or fix issues one by one.",
    },
    {
        "step": 3,
        "total": 6,
        "tab": "Clean",
        "title": "handle anything recommendations missed",
        "body": "the Clean tab gives you manual control. use the Data Type Guesser to fix columns storing numbers as text. use Split Column, Find and Replace, or Rename Columns for anything else.",
    },
    {
        "step": 4,
        "total": 6,
        "tab": "Validate",
        "title": "check formats and outliers",
        "body": "the Validate tab checks correctness. flag invalid emails, standardise phone numbers, normalise date formats, and cap extreme outliers. run this after cleaning, not before.",
    },
    {
        "step": 5,
        "total": 6,
        "tab": "Profile",
        "title": "explore your data visually",
        "body": "the Profile tab shows distribution charts, a missing value heatmap, and a before and after comparison. use it to spot anything that still looks off before exporting.",
    },
    {
        "step": 6,
        "total": 6,
        "tab": "History & Export",
        "title": "export when your score looks good",
        "body": "go to History and Export. download CSV or Excel, grab a PDF report to share with others, or save your pipeline as JSON to replay the same steps on a new dataset later.",
    },
]


def _init():
    if "tour_active" not in st.session_state:
        st.session_state.tour_active = False
    if "tour_step" not in st.session_state:
        st.session_state.tour_step = 0
    if "tour_seen" not in st.session_state:
        st.session_state.tour_seen = False


def render_guide_me_button():
    _init()
    _, btn_col = st.columns([6, 1])
    with btn_col:
        if st.button("Guide Me", key="guide_me_btn", use_container_width=True):
            st.session_state.tour_active = True
            st.session_state.tour_step = 0
            st.rerun()


def render(file_just_loaded=False, file_loaded=False):
    _init()

    if file_just_loaded and not st.session_state.tour_seen:
        st.session_state.tour_active = True
        st.session_state.tour_step = 0

    if not st.session_state.tour_active:
        return

    if not file_loaded:
        steps = STEPS_NO_FILE
        step_idx = 0
        has_back = False
        is_last = True
    else:
        steps = STEPS
        step_idx = min(st.session_state.tour_step, len(STEPS) - 1)
        st.session_state.tour_step = step_idx
        has_back = step_idx > 0
        is_last = step_idx == len(steps) - 1

    step = steps[step_idx]
    step_n = step["step"]
    total = step["total"]
    tab_name = step["tab"]
    title = step["title"]
    body = step["body"]
    next_label = "Finish" if is_last else "Next ›"

    with st.container():
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #0d1b2a 0%, #0f2236 100%);
                border-left: 4px solid #1f77b4;
                border-radius: 8px;
                padding: 14px 18px 12px 18px;
                margin-bottom: 8px;
            ">
                <div style="font-size:0.65rem;color:#1f77b4;font-weight:700;
                            letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">
                    guide &bull; step {step_n} of {total} &bull; go to {tab_name} tab
                </div>
                <div style="font-size:0.95rem;font-weight:700;color:#ffffff;margin-bottom:5px;">
                    {title}
                </div>
                <div style="font-size:0.82rem;color:#9db4c8;line-height:1.55;">
                    {body}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prog_col, skip_col, back_col, next_col = st.columns([4, 1, 1, 1])

        with prog_col:
            dots = ""
            for i in range(total):
                if i == step_n - 1:
                    dots += "● "
                else:
                    dots += "○ "
            st.markdown(
                f"<div style='padding-top:8px;font-size:0.75rem;color:#3a6a9a;'>{dots.strip()}</div>",
                unsafe_allow_html=True,
            )

        with skip_col:
            if st.button("Skip", key="tour_skip", use_container_width=True):
                st.session_state.tour_active = False
                st.session_state.tour_seen = True
                st.rerun()

        with back_col:
            if st.button("‹ Back", key="tour_back",
                         disabled=not has_back, use_container_width=True):
                st.session_state.tour_step -= 1
                st.rerun()

        with next_col:
            if st.button(next_label, key="tour_next",
                         type="primary", use_container_width=True):
                if is_last:
                    st.session_state.tour_active = False
                    st.session_state.tour_seen = True
                else:
                    st.session_state.tour_step += 1
                st.rerun()

        st.write("")