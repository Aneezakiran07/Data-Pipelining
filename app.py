import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from typing import Optional, Dict, List, Tuple
import io

st.set_page_config(
    page_title="Advanced Data Cleaning Pipeline",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* main header */
.main-header {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: #1f77b4 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.5px;
}

/* buttons */
.stButton > button {
    width: 100%;
    border-radius: 5px;
    height: 3em;
    font-weight: 500;
}

/* navbar-style tab bar */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px !important;
    background-color: #0e1117;
    padding: 0 8px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 18px 32px !important;
    color: #aaaaaa !important;
    border-radius: 0 !important;
    border: none !important;
    background: transparent !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background: rgba(255,255,255,0.05) !important;
}

.stTabs [aria-selected="true"] {
    color: #1f77b4 !important;
    border-bottom: 3px solid #1f77b4 !important;
    background: transparent !important;
    font-weight: 700 !important;
}

/* override Streamlit's default red active indicator line */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1f77b4 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
}

/* make the tab container flush with page edge */
.stTabs {
    margin-top: -1rem;
}

button[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    padding: 18px 32px !important;
}
</style>
""", unsafe_allow_html=True)


# cleaning helpers 

def checking_valid_input(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")

def drop_duplicate_rows(df):
    checking_valid_input(df)
    return df.drop_duplicates()

def drop_duplicate_columns(df):
    checking_valid_input(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.T.drop_duplicates().T

def stripping_whitespace(df):
    checking_valid_input(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def clean_string_edges(df, threshold=0.7, inplace=False, verbose=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    try:
        df_clean = df if inplace else df.copy()
        cleaned_cols = []
        for col in df_clean.select_dtypes(include=["object"]).columns:
            col_series = df_clean[col].astype(str)
            leading = col_series.str.extract(r"^([^\w\s])")[0].dropna()
            trailing = col_series.str.extract(r"([^\w\s])$")[0].dropna()
            keep_leading = (leading.value_counts(normalize=True).iloc[0] > threshold if len(leading) > 0 else False)
            keep_trailing = (trailing.value_counts(normalize=True).iloc[0] > threshold if len(trailing) > 0 else False)
            if not keep_leading:
                df_clean[col] = col_series.str.replace(r"^\W+", "", regex=True)
                cleaned_cols.append(col)
            if not keep_trailing:
                df_clean[col] = col_series.str.replace(r"\W+$", "", regex=True)
                if col not in cleaned_cols:
                    cleaned_cols.append(col)
        return None if inplace else df_clean
    except Exception:
        raise

def smart_column_cleaner(df, conversion_threshold=0.6, inplace=False, verbose=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    try:
        df_clean = df if inplace else df.copy()
        currency_pattern = r'[$€£¥₹₽₺₩฿]|(USD|EUR|GBP|JPY|CNY|INR|PKR|AUD|CAD)'
        for col in df_clean.select_dtypes(include="object").columns:
            series = df_clean[col].astype(str).str.strip()
            non_empty = series.replace("", np.nan).dropna()
            if non_empty.empty:
                continue
            currency_like = (non_empty.str.contains(r"\d", regex=True) &
                             non_empty.str.contains(currency_pattern, case=False, regex=True))
            if currency_like.mean() > conversion_threshold:
                cleaned = (non_empty.str.replace(r"[^\d.,\-()]", " ", regex=True)
                           .str.replace(r"\s+", " ", regex=True)
                           .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                           .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                           .str.replace(",", "", regex=False))
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    continue
            if non_empty.str.contains("%").mean() > conversion_threshold:
                cleaned = non_empty.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
                converted = pd.to_numeric(cleaned, errors="coerce") / 100
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    continue
            unit_pattern = r"\d+\s?(kg|g|mg|cm|mm|km|ml|l|lb|oz)"
            if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
                cleaned = non_empty.str.extract(r"([-]?\d+\.?\d*)", expand=False)
                converted = pd.to_numeric(cleaned, errors="coerce")
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    continue
            cleaned = non_empty.str.replace(r"[^\d.\-]", "", regex=True)
            converted = pd.to_numeric(cleaned, errors="coerce")
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted
        return None if inplace else df_clean
    except Exception:
        raise

def missing_value_handler(df, threshold=0.3, inplace=False, numeric_strategy="auto", verbose=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    try:
        df_clean = df.copy() if not inplace else df
        if numeric_strategy == "auto" and (df_clean.shape[1] > 50 or len(df_clean) > 5000):
            numeric_strategy = "mice"
        df_clean.replace(["?", "NA", "unknown", "n/a", "NaN", "null", -999, 999, 9999, ""], np.nan, inplace=True)
        cols_to_drop = df_clean.columns[df_clean.isna().mean() > threshold]
        if len(cols_to_drop):
            df_clean.drop(columns=cols_to_drop, inplace=True)
        num_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns
        if not num_cols.empty and df_clean[num_cols].isna().any().any():
            if numeric_strategy == "knn" or (numeric_strategy == "auto" and len(df_clean) <= 5000):
                imputer = KNNImputer(n_neighbors=min(5, max(3, len(df_clean) // 1000)))
            else:
                imputer = IterativeImputer(max_iter=10, random_state=42)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
        for col in cat_cols:
            if df_clean[col].isna().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Missing"
                df_clean[col] = df_clean[col].fillna(mode_val)
        return None if inplace else df_clean
    except Exception:
        raise

def validate_email_col(df, col, action="flag"):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    is_valid = df_clean[col].astype(str).str.strip().str.match(pattern)
    if action == "flag":
        df_clean[f"{col}_valid_email"] = is_valid
    elif action == "remove":
        df_clean = df_clean[is_valid]
    return df_clean

def validate_phone_col(df, col):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    def standardize(val):
        digits = re.sub(r"\D", "", str(val))
        if len(digits) == 10: return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"): return f"+{digits}"
        elif len(digits) >= 7: return f"+{digits}"
        return np.nan
    df_clean[col] = df_clean[col].apply(standardize)
    return df_clean

def validate_date_col(df, col, output_format="%Y-%m-%d"):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
            "%m-%d-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
            "%d/%m/%y", "%m/%d/%y", "%Y.%m.%d", "%d.%m.%Y"]
    def parse(val):
        if pd.isna(val) or str(val).strip() == "": return pd.NaT
        s = str(val).strip()
        for fmt in fmts:
            try: return pd.to_datetime(s, format=fmt)
            except: continue
        try: return pd.to_datetime(s)
        except: return pd.NaT
    parsed = df_clean[col].apply(parse)
    df_clean[col] = parsed.dt.strftime(output_format).where(parsed.notna(), other=np.nan)
    return df_clean

def cap_outliers(df, col, method="iqr", action="cap", threshold=1.5):
    checking_valid_input(df)
    if col not in df.columns: raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]): raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    s = df_clean[col].dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lower, upper = q1 - threshold * (q3 - q1), q3 + threshold * (q3 - q1)
    else:
        lower, upper = s.mean() - threshold * s.std(), s.mean() + threshold * s.std()
    if action == "cap":
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    else:
        df_clean = df_clean[df_clean[col].isna() | df_clean[col].between(lower, upper)]
    return df_clean

def validate_range(df, col, min_val, max_val, action="flag"):
    checking_valid_input(df)
    if col not in df.columns: raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]): raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    in_range = df_clean[col].between(min_val, max_val, inclusive="both") | df_clean[col].isna()
    if action == "flag":
        df_clean[f"{col}_in_range"] = in_range
    else:
        df_clean = df_clean[in_range]
    return df_clean

def find_and_replace(df, col, find, replace, use_regex=False):
    checking_valid_input(df)
    if col not in df.columns: raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    df_clean[col] = df_clean[col].astype(str).str.replace(find, replace, regex=use_regex)
    return df_clean

def push_history(label):
    if "history" not in st.session_state:
        st.session_state.history = []
    if len(st.session_state.history) >= 20:
        st.session_state.history.pop(0)
    st.session_state.history.append({"label": label, "df": st.session_state.current_df.copy()})

def undo_last():
    if st.session_state.get("history"):
        last = st.session_state.history.pop()
        st.session_state.current_df = last["df"]
        return last["label"]
    return None

def build_pipeline_script(history):
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import re",
        "from sklearn.impute import KNNImputer",
        "",
        "# load your file here",
        "df = pd.read_csv('your_file.csv')",
        "",
    ]

    for step in history:
        label = step["label"]
        lines.append(f"# --- {label} ---")

        if label == "Strip Whitespace":
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")

        elif label == "Drop Duplicate Rows":
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")

        elif label == "Drop Duplicate Columns":
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df = df.T.drop_duplicates().T")

        elif label == "Clean String Edges":
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    df[col] = df[col].astype(str).str.replace(r'^\\W+', '', regex=True).str.replace(r'\\W+$', '', regex=True)")

        elif label == "Smart Column Cleaner":
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.replace(r'[^\\d.\\-]', '', regex=True)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6:")
            lines.append("        df[col] = converted")

        elif label == "Handle Missing Values":
            lines.append("df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan, inplace=True)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty and df[num_cols].isna().any().any():")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")
            lines.append("for col in df.select_dtypes(exclude=np.number).columns:")
            lines.append("    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Missing')")

        elif label == "Auto-Fix All":
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan, inplace=True)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty and df[num_cols].isna().any().any():")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")

        elif label.startswith("Fix: strip_whitespace"):
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")

        elif label.startswith("Fix: convert_currency"):
            lines.append("# currency conversion — update column names as needed")
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.replace(r'[^\\d.,\\-()]', ' ', regex=True).str.extract(r'([-]?\\d[\\d\\.,]*)', expand=False).str.replace(',', '', regex=False)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6: df[col] = converted")

        elif label.startswith("Fix: convert_percentage"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    if df[col].str.contains('%').mean() > 0.6:")
            lines.append("        df[col] = pd.to_numeric(df[col].str.replace('%', '', regex=False), errors='coerce') / 100")

        elif label.startswith("Fix: convert_units"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    cleaned = df[col].str.extract(r'([-]?\\d+\\.?\\d*)', expand=False)")
            lines.append("    converted = pd.to_numeric(cleaned, errors='coerce')")
            lines.append("    if converted.notna().mean() > 0.6: df[col] = converted")

        elif label.startswith("Fix: handle_missing"):
            lines.append("df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan, inplace=True)")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("if not num_cols.empty:")
            lines.append("    df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")

        elif label.startswith("Fix: clean_edges"):
            lines.append("for col in df.select_dtypes(include='object').columns:")
            lines.append("    df[col] = df[col].astype(str).str.replace(r'^\\W+', '', regex=True).str.replace(r'\\W+$', '', regex=True)")

        elif label.startswith("Fix: drop_duplicates"):
            lines.append("df = df.drop_duplicates().reset_index(drop=True)")

        elif label.startswith("Fix: drop_dup_cols"):
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")

        elif label.startswith("Find & Replace in"):
            col = label.replace("Find & Replace in ", "").strip()
            lines.append(f"df['{col}'] = df['{col}'].astype(str).str.replace('FIND', 'REPLACE', regex=False)  # update FIND and REPLACE")

        elif label.startswith("Type Override:"):
            # label format: "Type Override: col_name -> type"
            parts = label.replace("Type Override: ", "").split(" -> ")
            if len(parts) == 2:
                col, dtype = parts[0].strip(), parts[1].strip()
                if "int" in dtype:
                    lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').astype('Int64')")
                elif "float" in dtype:
                    lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
                elif "datetime" in dtype:
                    lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
                elif "bool" in dtype:
                    lines.append(f"df['{col}'] = df['{col}'].astype(str).str.lower().map({{'true':True,'1':True,'yes':True,'false':False,'0':False,'no':False}})")
                elif "category" in dtype:
                    lines.append(f"df['{col}'] = df['{col}'].astype('category')")
                else:
                    lines.append(f"df['{col}'] = df['{col}'].astype(str)")

        elif label == "Validate Email":
            lines.append("pattern = r'^[\\w\\.\\+\\-]+@[\\w\\-]+\\.[a-zA-Z]{2,}$'")
            lines.append("# to flag: df['email_valid'] = df['email_col'].astype(str).str.match(pattern)")
            lines.append("# to remove: df = df[df['email_col'].astype(str).str.match(pattern)]")

        elif label == "Standardize Phone":
            lines.append("def standardize_phone(val):")
            lines.append("    digits = re.sub(r'\\D', '', str(val))")
            lines.append("    if len(digits) == 10: return f'+1{digits}'")
            lines.append("    elif len(digits) >= 7: return f'+{digits}'")
            lines.append("    return np.nan")
            lines.append("# df['phone_col'] = df['phone_col'].apply(standardize_phone)  # update col name")

        elif label == "Standardize Dates":
            lines.append("# df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce').dt.strftime('%Y-%m-%d')  # update col name and format")

        elif label == "Cap Outliers":
            lines.append("# IQR outlier capping — update col name")
            lines.append("# col = 'your_column'")
            lines.append("# q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)")
            lines.append("# df[col] = df[col].clip(lower=q1 - 1.5*(q3-q1), upper=q3 + 1.5*(q3-q1))")

        elif label == "Validate Range":
            lines.append("# range validation — update col name and bounds")
            lines.append("# df = df[df['your_column'].between(0, 100)]")

        else:
            lines.append(f"# (manual step — no code generated)")

        lines.append("")

    lines += ["print('Pipeline complete. Shape:', df.shape)"]
    return "\n".join(lines)


# cached functions 

@st.cache_data(show_spinner=False)
def load_file(file_bytes, filename, file_id, sheet_name=None):
    ext = filename.split(".")[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == "csv":
        return pd.read_csv(buf, quotechar='"', skipinitialspace=True)
    return pd.read_excel(buf, sheet_name=sheet_name)

@st.cache_data(show_spinner=False)
def get_dataframe_stats(df):
    return {
        "rows": df.shape[0], "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_cols": len(df.select_dtypes(include=np.number).columns),
        "categorical_cols": len(df.select_dtypes(exclude=np.number).columns),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024 ** 2,
    }

@st.cache_data(show_spinner=False)
def get_column_profile(df):
    rows = []
    for col in df.columns:
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        row = {
            "Column": col, "Type": str(s.dtype),
            "Non-Null": int(s.notna().sum()), "Null": int(s.isna().sum()),
            "Null %": f"{s.isna().mean()*100:.1f}%", "Unique": int(s.nunique()),
        }
        if is_num:
            row.update({
                "Min": round(float(s.min()), 4) if s.notna().any() else None,
                "Max": round(float(s.max()), 4) if s.notna().any() else None,
                "Mean": round(float(s.mean()), 4) if s.notna().any() else None,
                "Median": round(float(s.median()), 4) if s.notna().any() else None,
                "Std": round(float(s.std()), 4) if s.notna().any() else None,
                "Skew": round(float(s.skew()), 4) if s.notna().any() else None,
                "Sample Values": ", ".join(str(v) for v in s.dropna().head(3).tolist()),
            })
        else:
            row.update({"Min": "-", "Max": "-", "Mean": "-", "Median": "-", "Std": "-", "Skew": "-",
                        "Sample Values": ", ".join(f'"{v}"' for v in s.dropna().value_counts().head(3).index.tolist())})
        rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_analysis_and_recommendations(df, conversion_threshold):
    issues = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_cols": max(len(df.columns[df.columns.duplicated()].tolist()),
                              len(df.columns) - len(df.T.drop_duplicates().T.columns)),
        "whitespace_cols": [], "currency_cols": [], "percentage_cols": [],
        "unit_cols": [], "duration_cols": [],
        "missing_cells": int(df.isna().sum().sum()),
        "missing_cols": [], "edge_char_cols": [],
    }
    currency_pattern = r'[$€£¥₹₽₺₩฿]|(USD|EUR|GBP|JPY|CNY|INR|PKR)'
    for col in df.select_dtypes(include="object").columns:
        s_str = df[col].astype(str)
        if s_str.str.strip().ne(s_str).any():
            issues["whitespace_cols"].append(col)
        non_empty = s_str.str.strip().replace("", np.nan).dropna()
        if non_empty.empty:
            continue
        if (non_empty.str.contains(r"\d", regex=True) &
                non_empty.str.contains(currency_pattern, case=False, regex=True)).mean() > conversion_threshold:
            issues["currency_cols"].append(col)
        elif non_empty.str.contains("%").mean() > conversion_threshold:
            issues["percentage_cols"].append(col)
        elif non_empty.str.contains(r"\d+\s?(kg|g|cm|mm|km|ml|l|lb)", case=False, regex=True).mean() > conversion_threshold:
            issues["unit_cols"].append(col)
        elif non_empty.str.contains(r"(h|hr|hour|min|minute|sec|second)", case=False, regex=True).mean() > conversion_threshold:
            issues["duration_cols"].append(col)
        if (s_str.str.match(r"^\W") | s_str.str.match(r"\W$")).sum() > 0:
            issues["edge_char_cols"].append(col)
    for col in df.columns:
        p = df[col].isna().mean()
        if p > 0:
            issues["missing_cols"].append((col, p))
    recs = []
    if issues["duplicate_rows"] > 0:
        recs.append(("", f"Found {issues['duplicate_rows']} duplicate rows", "Removing duplicates prevents skewed analysis.", "drop_duplicates"))
    if issues["duplicate_cols"] > 0:
        recs.append(("", f"Found {issues['duplicate_cols']} duplicate columns", "These columns are wasting memory.", "drop_dup_cols"))
    if issues["whitespace_cols"]:
        recs.append(("", f"{len(issues['whitespace_cols'])} columns have extra whitespace", f"Columns: {', '.join(issues['whitespace_cols'][:3])}", "strip_whitespace"))
    if issues["currency_cols"]:
        recs.append(("", f"{len(issues['currency_cols'])} columns look like currency but aren't numeric", f"Columns: {', '.join(issues['currency_cols'][:3])}", "convert_currency"))
    if issues["percentage_cols"]:
        recs.append(("", f"{len(issues['percentage_cols'])} columns contain percentages as text", f"Columns: {', '.join(issues['percentage_cols'][:3])}", "convert_percentage"))
    if issues["unit_cols"]:
        recs.append(("", f"{len(issues['unit_cols'])} columns have measurement units mixed in", f"Columns: {', '.join(issues['unit_cols'][:3])}", "convert_units"))
    if issues["duration_cols"]:
        recs.append(("", f"{len(issues['duration_cols'])} columns contain time durations", f"Columns: {', '.join(issues['duration_cols'][:3])}", "convert_duration"))
    if issues["missing_cells"] > 0:
        top = sorted(issues["missing_cols"], key=lambda x: x[1], reverse=True)[:3]
        recs.append(("", f"{issues['missing_cells']} missing values found", f"Worst: {', '.join(f'{c} ({p*100:.0f}%)' for c,p in top)}", "handle_missing"))
    if issues["edge_char_cols"]:
        recs.append(("", f"{len(issues['edge_char_cols'])} columns have unwanted edge characters", f"Columns: {', '.join(issues['edge_char_cols'][:3])}", "clean_edges"))
    return issues, recs


# sidebar — once
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Simple", "Advanced"], horizontal=True, key="mode_radio",
        help="Simple uses sensible defaults so you can start cleaning right away. Advanced lets you tune the thresholds manually.")
    if mode == "Simple":
        st.caption("Using default settings. Switch to Advanced to customize.")
        missing_threshold = 0.30
        numeric_strategy = "auto"
        conversion_threshold = 0.60
    else:
        st.subheader("Missing Value Handler")
        missing_threshold = st.slider(
            "Drop columns with missing % >", 0, 100,
            value=st.session_state.get("missing_threshold_val", 30),
            help="Columns where more than this % of values are missing will be dropped entirely. Default 30% — raise it if you want to keep sparser columns, lower it to be stricter."
        ) / 100
        st.session_state["missing_threshold_val"] = int(missing_threshold * 100)
        numeric_strategy = st.selectbox(
            "Numeric Imputation Strategy", ["auto", "knn", "mice"],
            key="numeric_strategy_select",
            help="auto: picks KNN for small files and MICE for large ones. KNN: fills missing values using the k nearest similar rows — fast and good for most cases. MICE: iteratively models each column — more accurate but slower, best for large datasets with lots of missing data."
        )
        st.subheader("Smart Cleaner")
        conversion_threshold = st.slider(
            "Conversion Threshold %", 0, 100,
            value=st.session_state.get("conversion_threshold_val", 60),
            help="How many values in a column must match a pattern before the whole column gets converted. At 60%, if 60% of values look like currency then the whole column is converted. Lower it to convert more aggressively, raise it to be more conservative."
        ) / 100
        st.session_state["conversion_threshold_val"] = int(conversion_threshold * 100)


# tabs sit at the very top — navbar style
tab_upload, tab_overview, tab_recommend, tab_clean, tab_validate, tab_profile, tab_history = st.tabs([
    "  Upload  ", "  Overview  ", "  Recommendations  ",
    "  Clean  ", "  Validate  ", "  Profile  ", "  History & Export  ",
])

# upload tab — file uploader lives here, renders once inside this tab
with tab_upload:
    st.subheader("Advanced Data Cleaning Pipeline")
    st.caption("Upload a CSV or Excel file to get started. [Test CSV on GitHub](https://github.com/Aneezakiran07/Data-Pipelining)")
    st.write("")
    uploaded = st.file_uploader("", type=["csv", "xlsx", "xls"], key="uploader", label_visibility="collapsed")
    if uploaded is None:
        st.write("")
        st.subheader("Sample Data Format")
        st.dataframe(pd.DataFrame({
            "name":       [" Alice ", "Bob", "Charlie", "Alice"],
            "price":      ["$100", "$200.50", "300EUR", "$100"],
            "percentage": ["75%", "80.5%", "99%", "75%"],
            "weight":     ["100kg", "150.5 lbs", "?", "100kg"],
            "duration":   ["1h30m", "90min", "NA", "1h30m"],
        }), use_container_width=True)
        st.caption("The pipeline handles currency, percentages, units, missing values, and more.")
    else:
        st.success(f"{uploaded.name} is loaded. Navigate to any tab to start cleaning.")

# read upload from session state so other tabs can access it
uploaded = st.session_state.get("uploader")

# everything below only runs when a file is uploaded
# other tabs show a prompt if no file is loaded yet
if uploaded is None:
    with tab_overview:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_recommend:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_clean:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_validate:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_profile:
        st.info("Upload a file in the Upload tab to get started.")
    with tab_history:
        st.info("Upload a file in the Upload tab to get started.")
else:
    try:
        ext = uploaded.name.split(".")[-1].lower()
        selected_sheet = None
        uploaded.seek(0)
        # use Streamlit's internal file_id — changes every upload even for same file
        file_id = uploaded.file_id

        # detect new upload BEFORE loading — wipe state and rerun clean
        if st.session_state.get("loaded_file_id") != file_id:
            st.cache_data.clear()
            keys_to_clear = [k for k in st.session_state.keys()
                             if k not in ("uploader", "mode_radio", "sheet_selector")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.session_state["loaded_file_id"] = file_id
            st.session_state["missing_threshold_val"] = 30
            st.session_state["conversion_threshold_val"] = 60
            st.rerun()

        if ext in ["xlsx", "xls"]:
            xl_bytes = uploaded.read()
            xl = pd.ExcelFile(io.BytesIO(xl_bytes))
            sheets = xl.sheet_names
            if len(sheets) > 1:
                selected_sheet = st.selectbox("Select sheet:", sheets, key="sheet_selector")
            else:
                selected_sheet = sheets[0]
            file_bytes = xl_bytes
        else:
            file_bytes = uploaded.read()

        # include selected_sheet in the load key so switching sheets busts cache
        load_key = f"{file_id}_{selected_sheet}"
        df = load_file(file_bytes, uploaded.name, load_key, sheet_name=selected_sheet)

        # reset state when sheet changes or on first load
        state_key = f"state_{load_key}"
        if st.session_state.get("state_key_id") != state_key:
            st.session_state.update({
                "original_df": df.copy(),
                "current_df": df.copy(),
                "original_stats": get_dataframe_stats(df),
                "selected_columns": {},
                "val_selected": {},
                "last_success_msg": None,
                "history": [],
                "state_key_id": state_key,
            })

        if "val_selected" not in st.session_state:
            st.session_state.val_selected = {}

        cdf = st.session_state.current_df
        stats = get_dataframe_stats(cdf)
        orig_stats = st.session_state.original_stats
        all_cols  = list(cdf.columns)
        text_cols = list(cdf.select_dtypes(include="object").columns)
        num_cols  = list(cdf.select_dtypes(include=np.number).columns)

        st.info(f"{uploaded.name}  ·  {stats['rows']} rows × {stats['columns']} cols  ·  {stats['memory_usage']:.2f} MB")

        # col popover helper — defined here so val_selected is guaranteed to exist
        def _make_all_handler(section, cols):
            def h():
                st.session_state.val_selected[section] = cols.copy() if st.session_state.get(f"_va_{section}") else []
            return h

        def _make_col_handler(section, col):
            def h():
                sel = st.session_state.val_selected.get(section, [])
                if st.session_state.get(f"_vc_{section}_{col}"):
                    if col not in sel: st.session_state.val_selected[section] = sel + [col]
                else:
                    st.session_state.val_selected[section] = [c for c in sel if c != col]
            return h

        def col_popover(section, available_cols):
            n = len(st.session_state.val_selected.get(section, []))
            with st.popover(f"▼ {n} selected" if n else "▼ Select columns", use_container_width=True):
                st.checkbox("Apply to all", key=f"_va_{section}", on_change=_make_all_handler(section, available_cols))
                for c in available_cols:
                    st.checkbox(c, key=f"_vc_{section}_{c}", on_change=_make_col_handler(section, c))
            return n

        # show and immediately clear any pending success message
        def show_msg():
            if st.session_state.get("last_success_msg"):
                st.success(st.session_state.last_success_msg)
                st.session_state.last_success_msg = None

        # OVERVIEW TAB
        with tab_overview:
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
                st.caption(f"File has {total_rows} rows — slider capped at {total_rows}.")
            st.dataframe(cdf.head(n_prev), use_container_width=True)
            st.divider()
            with st.expander("Column Types & Info", expanded=False):
                st.dataframe(pd.DataFrame({
                    "Column": cdf.columns, "Type": cdf.dtypes.values,
                    "Non-Null": cdf.count().values, "Null": cdf.isna().sum().values,
                    "Unique": cdf.nunique().values,
                }), use_container_width=True)
            st.caption("Download and reset options are in the **History & Export** tab.")

        # RECOMMENDATIONS TAB
        with tab_recommend:
            show_msg()
            st.subheader("Smart Recommendations")
            issues, recs = get_analysis_and_recommendations(cdf, conversion_threshold)
            if not recs:
                st.success("Your data looks clean! No issues detected.")
            else:
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

                    n_sel = len(st.session_state.selected_columns.get(action_key, []))

                    if af_cols:
                        r1, r2, r3 = st.columns([5, 1.4, 1])
                        with r1:
                            st.write(f"{icon} **{title}**")
                            st.caption(desc)
                        with r2:
                            with st.popover(f"▼ {n_sel} sel" if n_sel else "▼ Select", use_container_width=True):
                                def _rall(ak, cols):
                                    def h():
                                        st.session_state.selected_columns[ak] = cols.copy() if st.session_state.get(f"_ra_{ak}") else []
                                    return h
                                def _rcol(ak, col):
                                    def h():
                                        sel = st.session_state.selected_columns.get(ak, [])
                                        if st.session_state.get(f"_rc_{ak}_{col}"):
                                            if col not in sel: st.session_state.selected_columns[ak] = sel + [col]
                                        else:
                                            st.session_state.selected_columns[ak] = [c for c in sel if c != col]
                                    return h
                                st.checkbox("All", key=f"_ra_{action_key}", on_change=_rall(action_key, af_cols))
                                for c in af_cols:
                                    st.checkbox(c, key=f"_rc_{action_key}_{c}", on_change=_rcol(action_key, c))
                        with r3:
                            if st.button("Fix", key=f"fix_{action_key}", disabled=n_sel == 0,
                                         type="primary" if n_sel else "secondary", use_container_width=True):
                                sel_cols = st.session_state.selected_columns.get(action_key, [])
                                try:
                                    push_history(f"Fix: {action_key}")
                                    tmp = cdf.copy()
                                    if action_key == "strip_whitespace":
                                        for c in sel_cols:
                                            if tmp[c].dtype == "object": tmp[c] = tmp[c].str.strip()
                                    elif action_key in ["convert_currency", "convert_percentage", "convert_units", "convert_duration"]:
                                        for c in sel_cols:
                                            if c not in tmp.columns: continue
                                            ne = tmp[c].astype(str).str.strip().replace("", np.nan).dropna()
                                            if ne.empty: continue
                                            if action_key == "convert_currency":
                                                cl = (ne.str.replace(r"[^\d.,\-()]", " ", regex=True)
                                                        .str.replace(r"\s+", " ", regex=True)
                                                        .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                                                        .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                                                        .str.replace(",", "", regex=False))
                                                tmp[c] = pd.to_numeric(cl, errors="coerce")
                                            elif action_key == "convert_percentage":
                                                tmp[c] = pd.to_numeric(ne.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce") / 100
                                            elif action_key == "convert_units":
                                                tmp[c] = pd.to_numeric(ne.str.extract(r"([-]?\d+\.?\d*)", expand=False), errors="coerce")
                                            elif action_key == "convert_duration":
                                                def _dur(v):
                                                    t = 0
                                                    for n, u in re.findall(r"(\d+\.?\d*)\s*(h(?:ou?r)?|m(?:in)?|s(?:ec)?)", str(v).lower()):
                                                        n = float(n)
                                                        if u.startswith("h"): t += n * 3600
                                                        elif u.startswith("m"): t += n * 60
                                                        elif u.startswith("s"): t += n
                                                    return t if t > 0 else np.nan
                                                tmp[c] = ne.apply(_dur)
                                    elif action_key == "clean_edges":
                                        for c in sel_cols:
                                            if tmp[c].dtype == "object":
                                                tmp[c] = tmp[c].astype(str).str.replace(r"^\W+", "", regex=True).str.replace(r"\W+$", "", regex=True)
                                    elif action_key == "handle_missing":
                                        vc = [c for c in sel_cols if c in tmp.columns]
                                        if vc:
                                            sub = missing_value_handler(tmp[vc].copy(), threshold=missing_threshold, numeric_strategy=numeric_strategy)
                                            for c in vc:
                                                if c in sub.columns: tmp[c] = sub[c]
                                    st.session_state.current_df = tmp
                                    st.session_state.selected_columns.pop(action_key, None)
                                    st.session_state.last_success_msg = f"Fixed {action_key} on {len(sel_cols)} column(s)!"
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    else:
                        r1, r2 = st.columns([5, 1])
                        with r1:
                            st.write(f"{icon} **{title}**"); st.caption(desc)
                        with r2:
                            if st.button("Fix", key=f"fix_{action_key}", type="primary", use_container_width=True):
                                try:
                                    push_history(f"Fix: {action_key}")
                                    if action_key == "drop_duplicates":
                                        st.session_state.current_df = drop_duplicate_rows(cdf)
                                        st.session_state.last_success_msg = "Duplicate rows removed!"
                                    elif action_key == "drop_dup_cols":
                                        st.session_state.current_df = drop_duplicate_columns(cdf)
                                        st.session_state.last_success_msg = "Duplicate columns removed!"
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                    st.write("")

                st.divider()
                if st.button("Auto-Fix All Issues", key="auto_fix_all", type="primary", use_container_width=True):
                    try:
                        push_history("Auto-Fix All")
                        with st.spinner("Running full pipeline..."):
                            tmp = stripping_whitespace(cdf)
                            tmp = drop_duplicate_rows(tmp)
                            tmp = drop_duplicate_columns(tmp)
                            tmp = clean_string_edges(tmp, threshold=0.7)
                            tmp = smart_column_cleaner(tmp, conversion_threshold=conversion_threshold)
                            tmp = missing_value_handler(tmp, threshold=missing_threshold, numeric_strategy=numeric_strategy)
                        st.session_state.current_df = tmp
                        st.session_state.selected_columns = {}
                        st.session_state.last_success_msg = "All issues fixed automatically!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # CLEAN TAB
        with tab_clean:
            st.subheader("Manual Cleaning Operations")
            st.write("**Basic Cleaning**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Strip Whitespace", key="ws_btn", use_container_width=True,
                             help="Removes leading and trailing spaces from all text columns. e.g. '  Alice ' becomes 'Alice'."):
                    try:
                        push_history("Strip Whitespace")
                        st.session_state.current_df = stripping_whitespace(cdf)
                        st.session_state["_omsg"] = ("ws_btn", "Whitespace stripped!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "ws_btn":
                    st.success(st.session_state.pop("_omsg")[1])
            with c2:
                if st.button("Drop Duplicate Rows", key="ddr_btn", use_container_width=True,
                             help="Removes rows that are completely identical to another row. Keeps the first occurrence."):
                    try:
                        push_history("Drop Duplicate Rows")
                        before = len(cdf)
                        st.session_state.current_df = drop_duplicate_rows(cdf)
                        dropped = before - len(st.session_state.current_df)
                        st.session_state["_omsg"] = ("ddr_btn", f"Dropped {dropped} duplicate rows!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "ddr_btn":
                    st.success(st.session_state.pop("_omsg")[1])
            with c3:
                if st.button("Drop Duplicate Cols", key="ddc_btn", use_container_width=True,
                             help="Removes columns that have the same name or identical values as another column."):
                    try:
                        push_history("Drop Duplicate Columns")
                        before = cdf.shape[1]
                        st.session_state.current_df = drop_duplicate_columns(cdf)
                        dropped = before - st.session_state.current_df.shape[1]
                        st.session_state["_omsg"] = ("ddc_btn", f"Dropped {dropped} duplicate columns!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "ddc_btn":
                    st.success(st.session_state.pop("_omsg")[1])
            with c4:
                if st.button("Clean String Edges", key="cse_btn", use_container_width=True,
                             help="Removes unwanted special characters from the start and end of text values. e.g. '$hello$' becomes 'hello'."):
                    try:
                        push_history("Clean String Edges")
                        st.session_state.current_df = clean_string_edges(cdf, threshold=0.7)
                        st.session_state["_omsg"] = ("cse_btn", "String edges cleaned!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "cse_btn":
                    st.success(st.session_state.pop("_omsg")[1])

            st.write("")
            st.write("**Advanced Cleaning**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Smart Column Cleaner", key="scc_btn", use_container_width=True,
                             help="Auto-detects and converts columns that look like currency, percentages, units, or durations into proper numeric values."):
                    try:
                        push_history("Smart Column Cleaner")
                        with st.spinner("Converting..."):
                            st.session_state.current_df = smart_column_cleaner(cdf, conversion_threshold=conversion_threshold)
                        st.session_state["_omsg"] = ("scc_btn", "Columns converted!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "scc_btn":
                    st.success(st.session_state.pop("_omsg")[1])
            with c2:
                if st.button("Handle Missing Values", key="hmv_btn", use_container_width=True,
                             help="Fills in missing values using KNN imputation for numeric columns (uses nearby similar rows to estimate the value) and mode for text columns."):
                    try:
                        push_history("Handle Missing Values")
                        with st.spinner("Imputing..."):
                            st.session_state.current_df = missing_value_handler(cdf, threshold=missing_threshold, numeric_strategy=numeric_strategy)
                        st.session_state["_omsg"] = ("hmv_btn", "Missing values handled!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "hmv_btn":
                    st.success(st.session_state.pop("_omsg")[1])

            st.divider()
            st.write("**Find & Replace**")
            fr1, fr2 = st.columns([3, 1])
            with fr1: fr_col = st.selectbox("Column", all_cols, key="fr_col")
            with fr2: fr_regex = st.checkbox("Use Regex", key="fr_regex",
                help=(
                    "Regex lets you match patterns instead of exact text.\n\n"
                    "Leave OFF for simple replacements like swapping 'N/A' with nothing.\n\n"
                    "Turn ON when you need patterns:\n"
                    "- Remove all digits: find '\\d+', replace with ''\n"
                    "- Remove all letters: find '[a-zA-Z]+', replace with ''\n"
                    "- Match $ or £ or €: find '[$£€]', replace with ''\n"
                    "- Remove anything in brackets: find '\\(.*?\\)', replace with ''\n"
                    "- Match extra spaces: find '\\s+', replace with ' '"
                ))
            fr3, fr4, fr5 = st.columns([2, 2, 1])
            with fr3: fr_find = st.text_input("Find", key="fr_find", placeholder="e.g. N/A")
            with fr4: fr_replace = st.text_input("Replace with", key="fr_replace", placeholder="leave blank to delete")
            with fr5:
                st.write(""); st.write("")
                if st.button("Run", key="fr_run", disabled=not fr_find,
                             type="primary" if fr_find else "secondary", use_container_width=True):
                    try:
                        push_history(f"Find & Replace in {fr_col}")
                        st.session_state.current_df = find_and_replace(cdf, fr_col, fr_find, fr_replace, fr_regex)
                        st.session_state["_omsg"] = ("fr_run", f"Find & Replace done on '{fr_col}'!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "fr_run":
                st.success(st.session_state.pop("_omsg")[1])

            st.divider()
            st.write("**Column Type Override**")
            to1, to2, to3 = st.columns([3, 2, 1])
            with to1: ov_col = st.selectbox("Column", all_cols, key="ov_col")
            with to2: ov_type = st.selectbox("Cast to", ["string (object)", "integer (int64)", "float (float64)", "datetime", "boolean", "category"], key="ov_type",
                help="string: plain text. integer: whole numbers. float: decimals. datetime: dates/times. boolean: true/false values. category: fixed set of labels (saves memory).")
            with to3:
                st.write(""); st.write("")
                if st.button("Apply", key="ov_apply", type="primary", use_container_width=True):
                    try:
                        push_history(f"Type Override: {ov_col} -> {ov_type}")
                        tmp = cdf.copy()
                        cd = tmp[ov_col]
                        if ov_type == "string (object)": tmp[ov_col] = cd.astype(str)
                        elif ov_type == "integer (int64)": tmp[ov_col] = pd.to_numeric(cd, errors="coerce").astype("Int64")
                        elif ov_type == "float (float64)": tmp[ov_col] = pd.to_numeric(cd, errors="coerce")
                        elif ov_type == "datetime": tmp[ov_col] = pd.to_datetime(cd, errors="coerce")
                        elif ov_type == "boolean":
                            tmp[ov_col] = cd.astype(str).str.lower().map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
                        elif ov_type == "category": tmp[ov_col] = cd.astype("category")
                        st.session_state.current_df = tmp
                        st.session_state["_omsg"] = ("ov_apply", f"Column '{ov_col}' cast to {ov_type}!")
                        st.rerun()
                    except Exception as e: st.error(str(e))
            if st.session_state.get("_omsg", ("",))[0] == "ov_apply":
                st.success(st.session_state.pop("_omsg")[1])

        # VALIDATE TAB
        with tab_validate:
            st.subheader("Validation & Quality")
            if text_cols:
                st.write("**Validate Email**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    st.caption("Flag adds a boolean column. Remove drops invalid rows.")
                    ea = st.radio("", ["Flag invalid", "Remove invalid rows"], key="email_radio", horizontal=True, label_visibility="collapsed",
                        help="Flag: adds an 'email_valid' boolean column so you can review invalid ones. Remove: deletes rows where the email doesn't match the standard format.")
                with v2: n_em = col_popover("email", text_cols)
                with v3:
                    if st.button("Run", key="run_email", disabled=n_em == 0, type="primary" if n_em else "secondary", use_container_width=True):
                        try:
                            push_history("Validate Email")
                            tmp = cdf.copy()
                            for c in st.session_state.val_selected.get("email", []):
                                tmp = validate_email_col(tmp, c, "flag" if "Flag" in ea else "remove")
                            st.session_state.current_df = tmp
                            st.session_state.val_selected.pop("email", None)
                            st.session_state["_omsg"] = ("run_email", f"Email validation done on {n_em} column(s)!")
                            st.rerun()
                        except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "run_email":
                    st.success(st.session_state.pop("_omsg")[1])

                st.divider()
                st.write("**Standardize Phone Numbers**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1: st.caption("Strips non-digit characters and formats to +[country code][number].")
                with v2: n_ph = col_popover("phone", text_cols)
                with v3:
                    if st.button("Run", key="run_phone", disabled=n_ph == 0, type="primary" if n_ph else "secondary", use_container_width=True):
                        try:
                            push_history("Standardize Phone")
                            tmp = cdf.copy()
                            for c in st.session_state.val_selected.get("phone", []):
                                tmp = validate_phone_col(tmp, c)
                            st.session_state.current_df = tmp
                            st.session_state.val_selected.pop("phone", None)
                            st.session_state["_omsg"] = ("run_phone", f"Phone standardized in {n_ph} column(s)!")
                            st.rerun()
                        except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "run_phone":
                    st.success(st.session_state.pop("_omsg")[1])

                st.divider()
                st.write("**Standardize Dates**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1: date_fmt = st.selectbox("Output format", ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"], key="date_fmt",
                    help="%Y-%m-%d = 2023-01-15 (recommended, sorts correctly). %d/%m/%Y = 15/01/2023 (common in Europe). %m/%d/%Y = 01/15/2023 (US format).")
                with v2: n_dt = col_popover("date", text_cols)
                with v3:
                    st.write("")
                    if st.button("Run", key="run_date", disabled=n_dt == 0, type="primary" if n_dt else "secondary", use_container_width=True):
                        try:
                            push_history("Standardize Dates")
                            tmp = cdf.copy()
                            for c in st.session_state.val_selected.get("date", []):
                                tmp = validate_date_col(tmp, c, output_format=date_fmt)
                            st.session_state.current_df = tmp
                            st.session_state.val_selected.pop("date", None)
                            st.session_state["_omsg"] = ("run_date", f"Dates standardized in {n_dt} column(s)!")
                            st.rerun()
                        except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "run_date":
                    st.success(st.session_state.pop("_omsg")[1])

            if num_cols:
                st.divider()
                st.write("**Cap / Remove Outliers**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    o1, o2, o3 = st.columns(3)
                    with o1: o_method = st.selectbox("Method", ["iqr", "zscore"], key="o_method",
                        help="IQR: uses the spread of the middle 50% of data — good for skewed data and most cases. Z-score: uses standard deviations from the mean — better for normally distributed data.")
                    with o2: o_action = st.selectbox("Action", ["cap", "remove"], key="o_action",
                        help="Cap: clips outliers to the boundary value instead of deleting them — safer, keeps row count. Remove: deletes the entire row containing the outlier.")
                    with o3: o_thresh = st.number_input("Threshold", min_value=0.5, max_value=10.0, value=1.5, step=0.5, key="o_thresh",
                        help="For IQR: multiplier of the IQR range — 1.5 is standard, 3.0 is more lenient. For Z-score: number of standard deviations — 2.0 catches ~5% of data, 3.0 catches ~0.3%.")
                with v2: n_out = col_popover("outlier", num_cols)
                with v3:
                    st.write(""); st.write("")
                    if st.button("Run", key="run_outlier", disabled=n_out == 0, type="primary" if n_out else "secondary", use_container_width=True):
                        try:
                            push_history("Cap Outliers")
                            before = len(cdf); tmp = cdf.copy()
                            for c in st.session_state.val_selected.get("outlier", []):
                                tmp = cap_outliers(tmp, c, method=o_method, action=o_action, threshold=o_thresh)
                            st.session_state.current_df = tmp
                            after = len(tmp)
                            st.session_state.val_selected.pop("outlier", None)
                            msg = f"Outliers capped in {n_out} column(s)!" if o_action == "cap" else f"Removed {before - after} outlier rows!"
                            st.session_state["_omsg"] = ("run_outlier", msg)
                            st.rerun()
                        except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "run_outlier":
                    st.success(st.session_state.pop("_omsg")[1])

                st.divider()
                st.write("**Validate Value Range**")
                v1, v2, v3 = st.columns([5, 1.4, 1])
                with v1:
                    r1, r2, r3 = st.columns(3)
                    with r1: rng_min = st.number_input("Min", value=0.0, key="rng_min")
                    with r2: rng_max = st.number_input("Max", value=100.0, key="rng_max")
                    with r3: rng_act = st.selectbox("Action", ["flag", "remove"], key="rng_act",
                        help="Flag: adds a new boolean column showing which rows are in range — non-destructive. Remove: deletes rows where the value falls outside the min/max.")
                with v2: n_rng = col_popover("range", num_cols)
                with v3:
                    st.write(""); st.write("")
                    if st.button("Run", key="run_range", disabled=n_rng == 0, type="primary" if n_rng else "secondary", use_container_width=True):
                        try:
                            push_history("Validate Range")
                            before = len(cdf); tmp = cdf.copy()
                            for c in st.session_state.val_selected.get("range", []):
                                tmp = validate_range(tmp, c, rng_min, rng_max, rng_act)
                            st.session_state.current_df = tmp
                            after = len(tmp)
                            st.session_state.val_selected.pop("range", None)
                            msg = f"Range flagged across {n_rng} column(s)!" if rng_act == "flag" else f"Removed {before - after} out-of-range rows!"
                            st.session_state["_omsg"] = ("run_range", msg)
                            st.rerun()
                        except Exception as e: st.error(str(e))
                if st.session_state.get("_omsg", ("",))[0] == "run_range":
                    st.success(st.session_state.pop("_omsg")[1])

        # PROFILE TAB
        with tab_profile:
            show_msg()
            st.subheader("Column Profiler")
            st.caption("Per-column stats: min, max, mean, median, std, skewness, and sample values.")
            profile = get_column_profile(cdf)
            st.dataframe(profile, use_container_width=True, hide_index=True)
            worst = profile[profile["Null"] > 0].sort_values("Null", ascending=False)
            if not worst.empty:
                st.caption(f"{len(worst)} column(s) have missing values. Worst: **{worst.iloc[0]['Column']}** ({worst.iloc[0]['Null %']} missing)")

            st.divider()
            st.subheader("Before / After Comparison")
            shared = [c for c in st.session_state.original_df.columns if c in cdf.columns]
            if shared:
                ba_col = st.selectbox("Select column to compare", shared, key="ba_col")
                ba_n = st.slider("Rows to preview", 5, 50, 10, key="ba_n")
                orig_s = st.session_state.original_df[ba_col].head(ba_n).reset_index(drop=True)
                curr_s = cdf[ba_col].head(ba_n).reset_index(drop=True)
                ml = min(len(orig_s), len(curr_s))
                orig_s, curr_s = orig_s.iloc[:ml], curr_s.iloc[:ml]
                changed = orig_s.fillna("").astype(str) != curr_s.fillna("").astype(str)
                st.dataframe(pd.DataFrame({
                    "Original": orig_s, "Current": curr_s,
                    "Changed": changed.map({True: "✓", False: ""}),
                }), use_container_width=True)
                n_ch = int(changed.sum())
                st.caption(f"✓ marks {n_ch} row(s) that changed." if n_ch else "No differences found in this preview window.")

        # HISTORY & EXPORT TAB
        with tab_history:

            # reset to original — big and prominent at the top
            st.subheader("Reset Data")
            st.warning("This will discard all cleaning and restore the original uploaded file.")
            if st.button("Reset to Original Data", key="reset_orig", use_container_width=True):
                st.session_state.current_df = st.session_state.original_df.copy()
                st.session_state.selected_columns = {}
                st.session_state.history = []
                st.session_state.last_success_msg = "Data reset to original!"
                st.rerun()


            st.divider()

            # download section
            st.subheader("Download Cleaned Data")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download as CSV",
                    data=cdf.to_csv(index=False).encode("utf-8"),
                    file_name="cleaned_data.csv", mime="text/csv",
                    key="dl_csv", use_container_width=True)
            with d2:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as w:
                    cdf.to_excel(w, index=False, sheet_name="Cleaned Data")
                    st.session_state.original_df.to_excel(w, index=False, sheet_name="Original Data")
                st.download_button(
                    "Download as Excel",
                    data=buf.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_xlsx", use_container_width=True)

            st.divider()

            # cleaning history
            st.subheader("Cleaning History")
            hist = st.session_state.get("history", [])
            if not hist:
                st.caption("No operations recorded yet. Every cleaning action is saved here.")
            else:
                st.caption(f"{len(hist)} operation(s) recorded. Max 20 steps kept.")
                for i, step in enumerate(reversed(hist)):
                    st.write(f"**{len(hist) - i}.** {step['label']}  —  {step['df'].shape[0]} rows × {step['df'].shape[1]} cols")
                st.write("")
                h1, h2 = st.columns(2)
                with h1:
                    if st.button("Undo Last Step", key="undo_btn", type="primary", use_container_width=True):
                        label = undo_last()
                        if label: st.session_state.last_success_msg = f"Undone: {label}"
                        st.rerun()
                with h2:
                    if st.button("Clear History", key="clear_hist", use_container_width=True):
                        st.session_state.history = []
                        st.session_state.last_success_msg = "History cleared!"
                        st.rerun()

            st.divider()

            # pipeline export
            st.subheader("Export Cleaning Pipeline")
            st.caption("Download your cleaning steps as a Python script you can rerun on any new file.")
            if not hist:
                st.caption("No steps recorded yet. Run some cleaning operations first.")
            else:
                script = build_pipeline_script(hist)
                st.download_button("Download pipeline.py", data=script.encode("utf-8"),
                    file_name="pipeline.py", mime="text/x-python",
                    key="dl_pipeline", use_container_width=True)
                with st.expander("Preview script", expanded=False):
                    st.code(script, language="python")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.info("Make sure your file is a valid CSV or Excel format.")
