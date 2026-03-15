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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)


# --- cleaning helpers ---

def checking_valid_input(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")

def drop_duplicate_rows(df: pd.DataFrame):
    checking_valid_input(df)
    return df.drop_duplicates()

def drop_duplicate_columns(df: pd.DataFrame):
    checking_valid_input(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.T.drop_duplicates().T

def stripping_whitespace(df: pd.DataFrame):
    checking_valid_input(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def clean_string_edges(df: pd.DataFrame, threshold: float = 0.7,
                       inplace: bool = False, verbose: bool = False) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    try:
        df_clean = df if inplace else df.copy()
        cleaned_cols = []
        for col in df_clean.select_dtypes(include=['object']).columns:
            col_series = df_clean[col].astype(str)
            leading_chars = col_series.str.extract(r'^([^\w\s])')[0].dropna()
            trailing_chars = col_series.str.extract(r'([^\w\s])$')[0].dropna()
            keep_leading = (leading_chars.value_counts(normalize=True).iloc[0] > threshold
                            if len(leading_chars) > 0 else False)
            keep_trailing = (trailing_chars.value_counts(normalize=True).iloc[0] > threshold
                             if len(trailing_chars) > 0 else False)
            if not keep_leading:
                df_clean[col] = col_series.str.replace(r'^\W+', '', regex=True)
                cleaned_cols.append(col)
            if not keep_trailing:
                df_clean[col] = col_series.str.replace(r'\W+$', '', regex=True)
                if col not in cleaned_cols:
                    cleaned_cols.append(col)
        return None if inplace else df_clean
    except Exception:
        raise

def smart_column_cleaner(df: pd.DataFrame, conversion_threshold: float = 0.6,
                          inplace: bool = False, verbose: bool = False) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    try:
        df_clean = df if inplace else df.copy()
        currency_symbols = r'[$€£¥₹₽₺₩฿₡₦₲₴₵₸₳₻₼₽₾₿]'
        currency_codes = r'(USD|EUR|GBP|JPY|CNY|INR|RUB|AUD|CAD|PKR|BDT|LKR|NPR|SGD|HKD|AED|CHF)'
        currency_text = r'(dollars?|euros?|pounds?|rupees?|yuan|yen|rubles?|pesos?|riyal|ringgit|baht|dinar|lei|krona|forint|złoty)'
        currency_pattern = f'{currency_symbols}|{currency_codes}|{currency_text}'
        conversions = []
        for col in df_clean.select_dtypes(include='object').columns:
            series = df_clean[col].astype(str).str.strip()
            non_empty = series.replace('', np.nan).dropna()
            if non_empty.empty:
                continue
            currency_like = (non_empty.str.contains(r'\d', regex=True) &
                             non_empty.str.contains(currency_pattern, case=False, regex=True))
            if currency_like.mean() > conversion_threshold:
                cleaned = (non_empty.str.replace(r'[^\d.,\-()]', ' ', regex=True)
                           .str.replace(r'\s+', ' ', regex=True)
                           .str.replace(r'\((.+?)\)', r'-\1', regex=True)
                           .str.extract(r'([-]?\d[\d\.,]*)', expand=False)
                           .str.replace(',', '', regex=False))
                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f"{col} (currency)")
                    continue
            if non_empty.str.contains('%').mean() > conversion_threshold:
                cleaned = non_empty.str.replace('%', '', regex=False).str.replace(r'[^\d.\-]', '', regex=True)
                converted = pd.to_numeric(cleaned, errors='coerce') / 100
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f"{col} (percentage)")
                    continue
            unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz|gal|pt|°C|°F|kWh|cal|ha|ac|sqft|m²|km²)'
            if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
                cleaned = non_empty.str.extract(r'([-]?\d+\.?\d*)', expand=False)
                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f"{col} (unit)")
                    continue
            duration_pattern = r'(h|hr|hour|min|minute|sec|second|s|m)'
            if non_empty.str.contains(duration_pattern, case=False, regex=True).mean() > conversion_threshold:
                def convert_duration(val):
                    val = str(val).lower()
                    total_seconds = 0
                    parts = re.findall(r'(\d+\.?\d*)\s*(h(?:ou?r)?|m(?:in)?|s(?:ec)?)', val)
                    for num, unit in parts:
                        num = float(num)
                        if unit.startswith('h'):
                            total_seconds += num * 3600
                        elif unit.startswith('m'):
                            total_seconds += num * 60
                        elif unit.startswith('s'):
                            total_seconds += num
                    return total_seconds if total_seconds > 0 else np.nan
                converted = non_empty.apply(convert_duration)
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f"{col} (duration)")
                    continue
            cleaned = non_empty.str.replace(r'[^\d.\-]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted
                conversions.append(f"{col} (numeric)")
        return None if inplace else df_clean
    except Exception:
        raise

def missing_value_handler(df: pd.DataFrame, threshold: float = 0.3,
                           inplace: bool = False, numeric_strategy: str = 'auto',
                           verbose: bool = False) -> Optional[pd.DataFrame]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if df.empty:
        return None if inplace else df.copy()
    try:
        df_clean = df.copy() if not inplace else df
        if numeric_strategy == 'auto':
            if df_clean.shape[1] > 50 or len(df_clean) > 5000:
                numeric_strategy = 'mice'
        missing_indicators = ['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', -999, 999, 9999, '']
        df_clean.replace(missing_indicators, np.nan, inplace=True)
        missing_percent = df_clean.isna().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index
        if len(cols_to_drop) > 0:
            df_clean.drop(columns=cols_to_drop, inplace=True)
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns
        if not numeric_cols.empty and df_clean[numeric_cols].isna().any().any():
            if numeric_strategy == 'knn' or (numeric_strategy == 'auto' and
                                              len(df_clean) <= 5000 and df_clean.shape[1] <= 50):
                imputer = KNNImputer(n_neighbors=min(5, max(3, len(df_clean) // 1000)))
            else:
                imputer = IterativeImputer(max_iter=10, random_state=42)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        for col in cat_cols:
            if df_clean[col].isna().any():
                if df_clean[col].nunique() < 0.5 * len(df_clean):
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Missing'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:
                    df_clean[col] = df_clean[col].fillna('Missing')
        return None if inplace else df_clean
    except Exception:
        raise


# --- validation helpers ---

def validate_email_col(df, col, action='flag'):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    pattern = r'^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$'
    is_valid = df_clean[col].astype(str).str.strip().str.match(pattern)
    if action == 'flag':
        df_clean[f'{col}_valid_email'] = is_valid
    elif action == 'remove':
        df_clean = df_clean[is_valid]
    return df_clean

def validate_phone_col(df, col, output_format='+1XXXXXXXXXX'):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    def standardize_phone(val):
        digits = re.sub(r'\D', '', str(val))
        if len(digits) == 10:
            return f'+1{digits}'
        elif len(digits) == 11 and digits.startswith('1'):
            return f'+{digits}'
        elif len(digits) >= 7:
            return f'+{digits}'
        return np.nan
    df_clean[col] = df_clean[col].apply(standardize_phone)
    return df_clean

def validate_date_col(df, col, output_format='%Y-%m-%d'):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    formats_to_try = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%d %b %Y', '%d %B %Y',
        '%b %d %Y', '%B %d %Y', '%b %d, %Y', '%B %d, %Y',
        '%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%Y.%m.%d', '%d.%m.%Y',
    ]
    def parse_single(val):
        if pd.isna(val) or str(val).strip() == '':
            return pd.NaT
        val_str = str(val).strip()
        for fmt in formats_to_try:
            try:
                return pd.to_datetime(val_str, format=fmt)
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(val_str)
        except (ValueError, TypeError):
            return pd.NaT
    parsed = df_clean[col].apply(parse_single)
    df_clean[col] = parsed.dt.strftime(output_format).where(parsed.notna(), other=np.nan)
    return df_clean

def cap_outliers(df, col, method='iqr', action='cap', threshold=1.5):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    series = df_clean[col].dropna()
    if method == 'iqr':
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
    else:
        mean, std = series.mean(), series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std
    if action == 'cap':
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    elif action == 'remove':
        mask = df_clean[col].isna() | ((df_clean[col] >= lower) & (df_clean[col] <= upper))
        df_clean = df_clean[mask]
    return df_clean

def validate_range(df, col, min_val, max_val, action='flag'):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    in_range = df_clean[col].between(min_val, max_val, inclusive='both') | df_clean[col].isna()
    if action == 'flag':
        df_clean[f'{col}_in_range'] = in_range
    elif action == 'remove':
        df_clean = df_clean[in_range]
    return df_clean


# --- find & replace, history ---

def find_and_replace(df: pd.DataFrame, col: str, find: str,
                     replace: str, use_regex: bool = False) -> pd.DataFrame:
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()
    df_clean[col] = df_clean[col].astype(str).str.replace(find, replace, regex=use_regex)
    return df_clean

def push_history(label: str):
    if 'history' not in st.session_state:
        st.session_state.history = []
    if len(st.session_state.history) >= 20:
        st.session_state.history.pop(0)
    st.session_state.history.append({
        'label': label,
        'df': st.session_state.current_df.copy()
    })

def undo_last():
    if st.session_state.get('history'):
        last = st.session_state.history.pop()
        st.session_state.current_df = last['df']
        return last['label']
    return None


# --- pipeline export helper ---
# builds a .py script from the history list so the user can rerun cleaning on any new file

def build_pipeline_script(history: list) -> str:
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import re",
        "",
        "# Load your file here",
        "df = pd.read_csv('your_file.csv')",
        "",
        "# Steps recorded from your cleaning session",
    ]
    for step in history:
        label = step['label']
        if label == "Strip Whitespace":
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")
        elif label == "Drop Duplicate Rows":
            lines.append("df = df.drop_duplicates()")
        elif label == "Drop Duplicate Columns":
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
            lines.append("df = df.T.drop_duplicates().T")
        elif label == "Auto-Fix All":
            lines.append("# Auto-Fix All — stripping, dedup, edge clean, smart convert, missing impute")
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")
            lines.append("df = df.drop_duplicates()")
            lines.append("df = df.loc[:, ~df.columns.duplicated()]")
        elif label.startswith("Fix: strip_whitespace"):
            lines.append("# Strip whitespace (targeted via recommendations)")
            lines.append("df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)")
        elif label.startswith("Fix: drop_duplicates"):
            lines.append("df = df.drop_duplicates()")
        elif label.startswith("Fix: handle_missing"):
            lines.append("# Missing value imputation — install sklearn if needed")
            lines.append("from sklearn.impute import KNNImputer")
            lines.append("num_cols = df.select_dtypes(include=np.number).columns")
            lines.append("df[num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[num_cols])")
        elif label.startswith("Find & Replace in"):
            col = label.replace("Find & Replace in ", "").strip()
            lines.append(f"# Find & Replace was applied to column '{col}' — update pattern/replacement below")
            lines.append(f"# df['{col}'] = df['{col}'].astype(str).str.replace('FIND', 'REPLACE', regex=False)")
        elif label.startswith("Type Override:"):
            lines.append(f"# {label}")
        elif label.startswith("Validate Email"):
            lines.append("# Email validation — add your column name below")
            lines.append("# df = df[df['email_col'].str.match(r'^[\\w\\.\\+\\-]+@[\\w\\-]+\\.[a-zA-Z]{2,}$')]")
        elif label.startswith("Cap Outliers"):
            lines.append("# Outlier capping — update col/method as needed")
            lines.append("# q1, q3 = df['col'].quantile(0.25), df['col'].quantile(0.75)")
            lines.append("# df['col'] = df['col'].clip(lower=q1 - 1.5*(q3-q1), upper=q3 + 1.5*(q3-q1))")
        elif label.startswith("Standardize Dates"):
            lines.append("# Date standardization — update col name")
            lines.append("# df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce').dt.strftime('%Y-%m-%d')")
        elif label.startswith("Standardize Phone"):
            lines.append("# Phone standardization — update col name")
            lines.append("# df['phone_col'] = df['phone_col'].astype(str).str.replace(r'\\D', '', regex=True)")
        else:
            lines.append(f"# Step: {label}")
    lines += ["", "print('Pipeline complete. Shape:', df.shape)"]
    return "\n".join(lines)


# --- cached functions ---

@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, filename: str, sheet_name=None) -> pd.DataFrame:
    ext = filename.split('.')[-1].lower()
    buf = io.BytesIO(file_bytes)
    if ext == 'csv':
        return pd.read_csv(buf, quotechar='"', skipinitialspace=True)
    else:
        return pd.read_excel(buf, sheet_name=sheet_name)

@st.cache_data(show_spinner=False)
def get_dataframe_stats(df: pd.DataFrame) -> Dict:
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_cells': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=np.number).columns),
        'categorical_cols': len(df.select_dtypes(exclude=np.number).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2
    }

@st.cache_data(show_spinner=False)
def get_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        row = {
            'Column': col,
            'Type': str(series.dtype),
            'Non-Null': int(series.notna().sum()),
            'Null': int(series.isna().sum()),
            'Null %': f"{series.isna().mean() * 100:.1f}%",
            'Unique': int(series.nunique()),
        }
        if is_numeric:
            row['Min']    = round(float(series.min()), 4) if series.notna().any() else None
            row['Max']    = round(float(series.max()), 4) if series.notna().any() else None
            row['Mean']   = round(float(series.mean()), 4) if series.notna().any() else None
            row['Median'] = round(float(series.median()), 4) if series.notna().any() else None
            row['Std']    = round(float(series.std()), 4) if series.notna().any() else None
            row['Skew']   = round(float(series.skew()), 4) if series.notna().any() else None
            row['Sample Values'] = ', '.join(str(v) for v in series.dropna().head(3).tolist())
        else:
            row['Min'] = row['Max'] = row['Mean'] = row['Median'] = row['Std'] = row['Skew'] = '-'
            top = series.dropna().value_counts().head(3)
            row['Sample Values'] = ', '.join(f'"{v}"' for v in top.index.tolist())
        rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_analysis_and_recommendations(df: pd.DataFrame, conversion_threshold: float) -> Tuple[Dict, List]:
    issues = {
        'duplicate_rows': 0, 'duplicate_cols': 0,
        'whitespace_cols': [], 'currency_cols': [], 'percentage_cols': [],
        'unit_cols': [], 'duration_cols': [],
        'missing_cells': 0, 'missing_cols': [], 'edge_char_cols': []
    }
    issues['duplicate_rows'] = df.duplicated().sum()
    dup_names = df.columns[df.columns.duplicated()].tolist()
    dup_content = len(df.columns) - len(df.T.drop_duplicates().T.columns)
    issues['duplicate_cols'] = max(len(dup_names), dup_content)
    for col in df.select_dtypes(include='object').columns:
        if df[col].astype(str).str.strip().ne(df[col].astype(str)).any():
            issues['whitespace_cols'].append(col)
    currency_symbols = r'[$€£¥₹₽₺₩฿₡₦₲₴₵₸₳₻₼₽₾₿]'
    currency_codes = r'(USD|EUR|GBP|JPY|CNY|INR|RUB|AUD|CAD|PKR)'
    currency_pattern = f'{currency_symbols}|{currency_codes}'
    for col in df.select_dtypes(include='object').columns:
        series = df[col].astype(str).str.strip()
        non_empty = series.replace('', np.nan).dropna()
        if non_empty.empty:
            continue
        currency_like = (non_empty.str.contains(r'\d', regex=True) &
                         non_empty.str.contains(currency_pattern, case=False, regex=True))
        if currency_like.mean() > conversion_threshold:
            issues['currency_cols'].append(col)
            continue
        if non_empty.str.contains('%').mean() > conversion_threshold:
            issues['percentage_cols'].append(col)
            continue
        unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz)'
        if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
            issues['unit_cols'].append(col)
            continue
        duration_pattern = r'(h|hr|hour|min|minute|sec|second)'
        if non_empty.str.contains(duration_pattern, case=False, regex=True).mean() > conversion_threshold:
            issues['duration_cols'].append(col)
    issues['missing_cells'] = df.isna().sum().sum()
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        if missing_pct > 0:
            issues['missing_cols'].append((col, missing_pct))
    for col in df.select_dtypes(include='object').columns:
        col_series = df[col].astype(str)
        has_edge_chars = col_series.str.match(r'^\W') | col_series.str.match(r'\W$')
        if has_edge_chars.sum() > 0:
            issues['edge_char_cols'].append(col)

    recommendations = []
    if issues['duplicate_rows'] > 0:
        recommendations.append(("", f"Found {issues['duplicate_rows']} duplicate rows",
            "Removing duplicates will reduce dataset size and prevent skewed analysis.", "drop_duplicates"))
    if issues['duplicate_cols'] > 0:
        recommendations.append(("", f"Found {issues['duplicate_cols']} duplicate columns",
            "These columns are wasting memory and processing power.", "drop_dup_cols"))
    if issues['whitespace_cols']:
        recommendations.append(("", f"{len(issues['whitespace_cols'])} columns have extra whitespace",
            f"Columns: {', '.join(issues['whitespace_cols'][:3])}{'...' if len(issues['whitespace_cols']) > 3 else ''}",
            "strip_whitespace"))
    if issues['currency_cols']:
        recommendations.append(("", f"{len(issues['currency_cols'])} columns look like currency but aren't numeric",
            f"Columns: {', '.join(issues['currency_cols'][:3])}{'...' if len(issues['currency_cols']) > 3 else ''}.",
            "convert_currency"))
    if issues['percentage_cols']:
        recommendations.append(("", f"{len(issues['percentage_cols'])} columns contain percentages as text",
            f"Columns: {', '.join(issues['percentage_cols'][:3])}{'...' if len(issues['percentage_cols']) > 3 else ''}.",
            "convert_percentage"))
    if issues['unit_cols']:
        recommendations.append(("", f"{len(issues['unit_cols'])} columns have measurement units mixed in",
            f"Columns: {', '.join(issues['unit_cols'][:3])}{'...' if len(issues['unit_cols']) > 3 else ''}",
            "convert_units"))
    if issues['duration_cols']:
        recommendations.append(("", f"{len(issues['duration_cols'])} columns contain time durations",
            f"Columns: {', '.join(issues['duration_cols'][:3])}{'...' if len(issues['duration_cols']) > 3 else ''}.",
            "convert_duration"))
    if issues['missing_cells'] > 0:
        top_missing = sorted(issues['missing_cols'], key=lambda x: x[1], reverse=True)[:3]
        col_details = ', '.join([f"{c} ({p * 100:.0f}%)" for c, p in top_missing])
        recommendations.append(("", f"{issues['missing_cells']} missing values found",
            f"Worst: {col_details}. Use ML imputation to fill intelligently.", "handle_missing"))
    if issues['edge_char_cols']:
        recommendations.append(("", f"{len(issues['edge_char_cols'])} columns have unwanted edge characters",
            f"Columns: {', '.join(issues['edge_char_cols'][:3])}{'...' if len(issues['edge_char_cols']) > 3 else ''}",
            "clean_edges"))
    return issues, recommendations


# --- app starts here ---

st.markdown('<p class="main-header">Advanced Data Cleaning Pipeline</p>', unsafe_allow_html=True)
st.markdown("Upload a CSV or Excel file and get smart recommendations on how to clean your data.")
st.markdown("**Don't have a file? Download the test CSV from [this GitHub repo](https://github.com/Aneezakiran07/Data-Pipelining).**")

# sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Simple", "Advanced"], horizontal=True, key="sidebar_mode")

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
            help="Columns above this missing % will be dropped"
        ) / 100
        st.session_state["missing_threshold_val"] = int(missing_threshold * 100)

        numeric_strategy = st.selectbox(
            "Numeric Imputation Strategy", ['auto', 'knn', 'mice'],
            key="numeric_strategy_select",
            help="auto: KNN for small datasets, MICE for large ones"
        )

        st.subheader("Smart Cleaner")
        conversion_threshold = st.slider(
            "Conversion Threshold %", 0, 100,
            value=st.session_state.get("conversion_threshold_val", 60),
            help="Min % of values that must be convertible"
        ) / 100
        st.session_state["conversion_threshold_val"] = int(conversion_threshold * 100)

        st.divider()
        if st.button("Reset All", key="reset_all_button", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key != "sidebar_mode":
                    del st.session_state[key]
            st.session_state["missing_threshold_val"] = 30
            st.session_state["conversion_threshold_val"] = 60
            st.cache_data.clear()
            st.rerun()

st.divider()

# file upload
st.subheader("Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'],
    key="file_uploader_main"
)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        selected_sheet = None

        if file_extension in ['xlsx', 'xls']:
            xl_bytes = uploaded_file.read()
            xl = pd.ExcelFile(io.BytesIO(xl_bytes))
            sheet_names = xl.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    f"This file has {len(sheet_names)} sheets. Select one:",
                    sheet_names, key="sheet_selector"
                )
                st.info(f"Loaded sheet: {selected_sheet}")
            else:
                selected_sheet = sheet_names[0]
            file_bytes = xl_bytes
        else:
            file_bytes = uploaded_file.read()

        df = load_file(file_bytes, uploaded_file.name, sheet_name=selected_sheet)

        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get('loaded_file_id') != file_id:
            st.session_state.original_df      = df.copy()
            st.session_state.current_df       = df.copy()
            st.session_state.original_stats   = get_dataframe_stats(df)
            st.session_state.loaded_file_id   = file_id
            st.session_state.selected_columns = {}
            st.session_state.val_selected     = {}
            st.session_state.last_success_msg = None
            st.session_state.history          = []

        current_stats = get_dataframe_stats(st.session_state.current_df)

        st.info(f"DataFrame: {current_stats['rows']} rows × {current_stats['columns']} columns | "
                f"Memory: {current_stats['memory_usage']:.2f} MB")

        # stats dashboard
        st.subheader("Data Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", current_stats['rows'],
                      delta=current_stats['rows'] - st.session_state.original_stats['rows'],
                      delta_color="inverse")
            st.metric("Columns", current_stats['columns'],
                      delta=current_stats['columns'] - st.session_state.original_stats['columns'],
                      delta_color="inverse")
        with col2:
            st.metric("Missing Cells", current_stats['missing_cells'],
                      delta=current_stats['missing_cells'] - st.session_state.original_stats['missing_cells'],
                      delta_color="inverse")
            st.metric("Duplicate Rows", current_stats['duplicate_rows'],
                      delta=current_stats['duplicate_rows'] - st.session_state.original_stats['duplicate_rows'],
                      delta_color="inverse")
        with col3:
            st.metric("Numeric Columns", current_stats['numeric_cols'])
            st.metric("Categorical Columns", current_stats['categorical_cols'])

        st.divider()

        # column profiler
        st.subheader("Column Profiler")
        st.caption("Per-column stats: min, max, mean, median, std, skewness, and sample values.")
        with st.expander("View Column Profile", expanded=False):
            profile_df = get_column_profile(st.session_state.current_df)
            st.dataframe(profile_df, use_container_width=True, hide_index=True)
            worst = profile_df[profile_df['Null'] > 0].sort_values('Null', ascending=False)
            if not worst.empty:
                st.caption(f"⚠️ {len(worst)} column(s) have missing values. "
                           f"Worst: **{worst.iloc[0]['Column']}** ({worst.iloc[0]['Null %']} missing)")

        st.divider()

        # before/after column comparison
        st.subheader("Before / After Comparison")
        st.caption("Pick a column to see its original values side by side with the current cleaned values.")

        all_shared_cols = [c for c in st.session_state.original_df.columns
                           if c in st.session_state.current_df.columns]

        if all_shared_cols:
            ba_col = st.selectbox("Select column to compare", all_shared_cols, key="ba_col_select")
            n_preview = st.slider("Rows to preview", 5, 50, 10, key="ba_rows_slider")

            orig_series   = st.session_state.original_df[ba_col].head(n_preview).reset_index(drop=True)
            current_series = st.session_state.current_df[ba_col].head(n_preview).reset_index(drop=True)

            # flag rows that actually changed so the user can spot them quickly
            changed_mask = orig_series.astype(str) != current_series.astype(str)
            n_changed = int(changed_mask.sum())

            ba_df = pd.DataFrame({
                'Original': orig_series,
                'Current': current_series,
                'Changed': changed_mask.map({True: '✓', False: ''})
            })

            st.dataframe(ba_df, use_container_width=True)
            if n_changed > 0:
                st.caption(f"✓ marks {n_changed} row(s) that changed in this preview window.")
            else:
                st.caption("No differences found in this preview window.")
        else:
            st.caption("No matching columns found between original and current data.")

        st.divider()

        # smart recommendations
        st.subheader("Smart Recommendations")

        issues, recommendations = get_analysis_and_recommendations(
            st.session_state.current_df, conversion_threshold
        )

        if len(recommendations) == 0:
            if st.session_state.get('last_success_msg'):
                st.success(st.session_state.last_success_msg)
                st.session_state.last_success_msg = None
            st.success("Your data looks clean! No issues detected.")
        else:
            st.warning(f"Found {len(recommendations)} potential issues in your data")

            if st.session_state.get('last_success_msg'):
                st.success(st.session_state.last_success_msg)
                st.session_state.last_success_msg = None

            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = {}

            for idx, (icon, title, description, action_key) in enumerate(recommendations):
                affected_columns = []
                if action_key == "strip_whitespace":   affected_columns = issues['whitespace_cols']
                elif action_key == "convert_currency": affected_columns = issues['currency_cols']
                elif action_key == "convert_percentage": affected_columns = issues['percentage_cols']
                elif action_key == "convert_units":    affected_columns = issues['unit_cols']
                elif action_key == "convert_duration": affected_columns = issues['duration_cols']
                elif action_key == "clean_edges":      affected_columns = issues['edge_char_cols']
                elif action_key == "handle_missing":   affected_columns = [c for c, _ in issues['missing_cols']]

                num_selected = len(st.session_state.selected_columns.get(action_key, []))

                if affected_columns:
                    dropdown_label = f"▼ {num_selected} selected" if num_selected > 0 else "▼ Select columns"
                    col1, col2, col3 = st.columns([5, 1.4, 1])
                    with col1:
                        st.write(f"{icon} **{title}**")
                        st.caption(description)
                    with col2:
                        with st.popover(dropdown_label, use_container_width=True):
                            st.write(f"**Columns for: {title}**")
                            st.caption("Check the columns you want to fix")

                            def make_apply_all_handler(ak, cols):
                                def handler():
                                    if st.session_state.get(f"_widget_all_{ak}", False):
                                        st.session_state.selected_columns[ak] = cols.copy()
                                    else:
                                        st.session_state.selected_columns[ak] = []
                                return handler

                            def make_col_handler(ak, col):
                                def handler():
                                    sel = st.session_state.selected_columns.get(ak, [])
                                    if st.session_state.get(f"_widget_chk_{ak}_{col}", False):
                                        if col not in sel:
                                            st.session_state.selected_columns[ak] = sel + [col]
                                    else:
                                        st.session_state.selected_columns[ak] = [c for c in sel if c != col]
                                return handler

                            st.checkbox("Apply to all columns", key=f"_widget_all_{action_key}",
                                        on_change=make_apply_all_handler(action_key, affected_columns))
                            for col in affected_columns:
                                st.checkbox(col, key=f"_widget_chk_{action_key}_{col}",
                                            on_change=make_col_handler(action_key, col))
                    with col3:
                        has_selection = num_selected > 0
                        if st.button("Fix This", key=f"fix_{action_key}", use_container_width=True,
                                     disabled=not has_selection,
                                     type="primary" if has_selection else "secondary"):
                            selected_cols = st.session_state.selected_columns.get(action_key, [])
                            try:
                                with st.spinner(f"Fixing {len(selected_cols)} column(s)..."):
                                    push_history(f"Fix: {action_key}")
                                    df_temp = st.session_state.current_df.copy()

                                    if action_key == "strip_whitespace":
                                        for col in selected_cols:
                                            if col in df_temp.columns and df_temp[col].dtype == 'object':
                                                df_temp[col] = df_temp[col].str.strip()
                                    elif action_key in ["convert_currency", "convert_percentage",
                                                        "convert_units", "convert_duration"]:
                                        for col in selected_cols:
                                            if col not in df_temp.columns:
                                                continue
                                            series = df_temp[col].astype(str).str.strip()
                                            non_empty = series.replace('', np.nan).dropna()
                                            if non_empty.empty:
                                                continue
                                            if action_key == "convert_currency":
                                                cleaned = (non_empty.str.replace(r'[^\d.,\-()]', ' ', regex=True)
                                                           .str.replace(r'\s+', ' ', regex=True)
                                                           .str.replace(r'\((.+?)\)', r'-\1', regex=True)
                                                           .str.extract(r'([-]?\d[\d\.,]*)', expand=False)
                                                           .str.replace(',', '', regex=False))
                                                df_temp[col] = pd.to_numeric(cleaned, errors='coerce')
                                            elif action_key == "convert_percentage":
                                                cleaned = non_empty.str.replace('%', '', regex=False).str.replace(r'[^\d.\-]', '', regex=True)
                                                df_temp[col] = pd.to_numeric(cleaned, errors='coerce') / 100
                                            elif action_key == "convert_units":
                                                cleaned = non_empty.str.extract(r'([-]?\d+\.?\d*)', expand=False)
                                                df_temp[col] = pd.to_numeric(cleaned, errors='coerce')
                                            elif action_key == "convert_duration":
                                                def convert_dur(val):
                                                    val = str(val).lower()
                                                    total = 0
                                                    for num, unit in re.findall(r'(\d+\.?\d*)\s*(h(?:ou?r)?|m(?:in)?|s(?:ec)?)', val):
                                                        num = float(num)
                                                        if unit.startswith('h'): total += num * 3600
                                                        elif unit.startswith('m'): total += num * 60
                                                        elif unit.startswith('s'): total += num
                                                    return total if total > 0 else np.nan
                                                df_temp[col] = non_empty.apply(convert_dur)
                                    elif action_key == "clean_edges":
                                        for col in selected_cols:
                                            if col in df_temp.columns and df_temp[col].dtype == 'object':
                                                df_temp[col] = (df_temp[col].astype(str)
                                                                .str.replace(r'^\W+', '', regex=True)
                                                                .str.replace(r'\W+$', '', regex=True))
                                    elif action_key == "handle_missing":
                                        valid_cols = [c for c in selected_cols if c in df_temp.columns]
                                        if valid_cols:
                                            df_subset = df_temp[valid_cols].copy()
                                            df_subset = missing_value_handler(
                                                df_subset, threshold=missing_threshold,
                                                numeric_strategy=numeric_strategy, verbose=False)
                                            for c in valid_cols:
                                                if c in df_subset.columns:
                                                    df_temp[c] = df_subset[c]

                                    st.session_state.current_df = df_temp
                                    st.session_state.selected_columns.pop(action_key, None)
                                    if action_key == "strip_whitespace":
                                        st.session_state.last_success_msg = f"Stripped whitespace from {len(selected_cols)} column(s)!"
                                    elif action_key in ["convert_currency", "convert_percentage",
                                                        "convert_units", "convert_duration"]:
                                        st.session_state.last_success_msg = f"Converted {len(selected_cols)} column(s)!"
                                    elif action_key == "clean_edges":
                                        st.session_state.last_success_msg = f"Cleaned edges in {len(selected_cols)} column(s)!"
                                    elif action_key == "handle_missing":
                                        st.session_state.last_success_msg = f"Handled missing values in {len(valid_cols)} column(s)!"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"{icon} **{title}**")
                        st.caption(description)
                    with col2:
                        if st.button("Fix This", key=f"fix_{action_key}", use_container_width=True, type="primary"):
                            try:
                                with st.spinner("Fixing..."):
                                    push_history(f"Fix: {action_key}")
                                    if action_key == "drop_duplicates":
                                        st.session_state.current_df = drop_duplicate_rows(st.session_state.current_df)
                                        st.session_state.last_success_msg = "Duplicate rows removed!"
                                    elif action_key == "drop_dup_cols":
                                        st.session_state.current_df = drop_duplicate_columns(st.session_state.current_df)
                                        st.session_state.last_success_msg = "Duplicate columns removed!"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                st.write("")

            if st.button("Auto-Fix All Issues", key="auto_fix_all", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Running complete cleaning pipeline..."):
                        push_history("Auto-Fix All")
                        df_temp = st.session_state.current_df.copy()
                        df_temp = stripping_whitespace(df_temp)
                        df_temp = drop_duplicate_rows(df_temp)
                        df_temp = drop_duplicate_columns(df_temp)
                        df_temp = clean_string_edges(df_temp, threshold=0.7, verbose=False)
                        df_temp = smart_column_cleaner(df_temp, conversion_threshold=conversion_threshold, verbose=False)
                        df_temp = missing_value_handler(df_temp, threshold=missing_threshold,
                                                        numeric_strategy=numeric_strategy, verbose=False)
                        st.session_state.current_df = df_temp
                        st.session_state.selected_columns = {}
                        st.session_state.last_success_msg = "All issues fixed automatically!"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()

        # manual cleaning operations
        st.subheader("Manual Cleaning Operations")
        st.markdown("Or choose specific operations to perform manually:")

        st.write("**Basic Cleaning**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Strip Whitespace", key="manual_strip_ws", use_container_width=True):
                try:
                    push_history("Strip Whitespace")
                    st.session_state.current_df = stripping_whitespace(st.session_state.current_df)
                    st.success("Whitespace stripped!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col2:
            if st.button("Drop Duplicate Rows", key="manual_drop_dup_rows", use_container_width=True):
                try:
                    before_rows = st.session_state.current_df.shape[0]
                    push_history("Drop Duplicate Rows")
                    st.session_state.current_df = drop_duplicate_rows(st.session_state.current_df)
                    st.success(f"Dropped {before_rows - st.session_state.current_df.shape[0]} duplicate rows!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col3:
            if st.button("Drop Duplicate Columns", key="manual_drop_dup_cols", use_container_width=True):
                try:
                    before_cols = st.session_state.current_df.shape[1]
                    push_history("Drop Duplicate Columns")
                    st.session_state.current_df = drop_duplicate_columns(st.session_state.current_df)
                    st.success(f"Dropped {before_cols - st.session_state.current_df.shape[1]} duplicate columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col4:
            if st.button("Clean String Edges", key="manual_clean_edges", use_container_width=True):
                try:
                    push_history("Clean String Edges")
                    with st.spinner("Cleaning..."):
                        st.session_state.current_df = clean_string_edges(
                            st.session_state.current_df, threshold=0.7, verbose=False)
                    st.success("String edges cleaned!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("")
        st.write("**Advanced Cleaning**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Smart Column Cleaner", key="manual_smart_cleaner", use_container_width=True):
                try:
                    push_history("Smart Column Cleaner")
                    with st.spinner("Analyzing and converting columns..."):
                        st.session_state.current_df = smart_column_cleaner(
                            st.session_state.current_df, conversion_threshold=conversion_threshold, verbose=False)
                    st.success("Columns converted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col2:
            if st.button("Handle Missing Values", key="manual_missing_handler", use_container_width=True):
                try:
                    push_history("Handle Missing Values")
                    with st.spinner("Handling missing values..."):
                        st.session_state.current_df = missing_value_handler(
                            st.session_state.current_df, threshold=missing_threshold,
                            numeric_strategy=numeric_strategy, verbose=False)
                    st.success("Missing values handled!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("")
        st.write("**Find & Replace**")
        all_cols = list(st.session_state.current_df.columns)

        fr_c1, fr_c2 = st.columns([3, 1])
        with fr_c1:
            fr_col = st.selectbox("Column to search in", all_cols, key="fr_col")
        with fr_c2:
            fr_regex = st.checkbox("Use Regex", key="fr_regex", value=False)

        fr_c3, fr_c4, fr_c5 = st.columns([2, 2, 1])
        with fr_c3:
            fr_find = st.text_input("Find", key="fr_find", placeholder='e.g. N/A')
        with fr_c4:
            fr_replace = st.text_input("Replace with", key="fr_replace", placeholder='leave blank to delete')
        with fr_c5:
            st.write("")
            st.write("")
            if st.button("Run", key="run_find_replace", use_container_width=True,
                         type="primary" if fr_find else "secondary", disabled=not fr_find):
                try:
                    push_history(f"Find & Replace in {fr_col}")
                    st.session_state.current_df = find_and_replace(
                        st.session_state.current_df, col=fr_col,
                        find=fr_find, replace=fr_replace, use_regex=fr_regex)
                    st.session_state.last_success_msg = f"Find & Replace done on column '{fr_col}'!"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("")

        # column type override
        # lets user manually cast a column to a different type — prevents silent bugs from wrong inferred types
        st.write("**Column Type Override**")
        st.caption("Force a column to a specific type. Useful when auto-detection gets it wrong.")

        to_c1, to_c2, to_c3 = st.columns([3, 2, 1])
        type_options = ['string (object)', 'integer (int64)', 'float (float64)',
                        'datetime', 'boolean', 'category']
        with to_c1:
            override_col = st.selectbox("Column", all_cols, key="override_col")
        with to_c2:
            override_type = st.selectbox("Cast to", type_options, key="override_type")
        with to_c3:
            st.write("")
            st.write("")
            if st.button("Apply", key="apply_type_override", use_container_width=True, type="primary"):
                try:
                    push_history(f"Type Override: {override_col} → {override_type}")
                    df_temp = st.session_state.current_df.copy()
                    col_data = df_temp[override_col]

                    if override_type == 'string (object)':
                        df_temp[override_col] = col_data.astype(str)
                    elif override_type == 'integer (int64)':
                        df_temp[override_col] = pd.to_numeric(col_data, errors='coerce').astype('Int64')
                    elif override_type == 'float (float64)':
                        df_temp[override_col] = pd.to_numeric(col_data, errors='coerce').astype('float64')
                    elif override_type == 'datetime':
                        df_temp[override_col] = pd.to_datetime(col_data, errors='coerce')
                    elif override_type == 'boolean':
                        # map common truthy/falsy strings, then cast
                        bool_map = {'true': True, '1': True, 'yes': True,
                                    'false': False, '0': False, 'no': False}
                        df_temp[override_col] = col_data.astype(str).str.lower().map(bool_map)
                    elif override_type == 'category':
                        df_temp[override_col] = col_data.astype('category')

                    st.session_state.current_df = df_temp
                    st.session_state.last_success_msg = f"Column '{override_col}' cast to {override_type}!"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("")

        # validation & quality
        st.write("**Validation & Quality**")

        if 'val_selected' not in st.session_state:
            st.session_state.val_selected = {}

        text_cols = list(st.session_state.current_df.select_dtypes(include='object').columns)
        num_cols  = list(st.session_state.current_df.select_dtypes(include=np.number).columns)

        def make_val_all_handler(section, cols):
            def handler():
                if st.session_state.get(f"_vall_{section}", False):
                    st.session_state.val_selected[section] = cols.copy()
                else:
                    st.session_state.val_selected[section] = []
            return handler

        def make_val_col_handler(section, col):
            def handler():
                sel = st.session_state.val_selected.get(section, [])
                if st.session_state.get(f"_valc_{section}_{col}", False):
                    if col not in sel:
                        st.session_state.val_selected[section] = sel + [col]
                else:
                    st.session_state.val_selected[section] = [c for c in sel if c != col]
            return handler

        def col_popover(section, available_cols):
            n = len(st.session_state.val_selected.get(section, []))
            label = f"v {n} selected" if n > 0 else "v Select columns"
            with st.popover(label, use_container_width=True):
                st.caption("Select columns to apply this operation")
                st.checkbox("Apply to all", key=f"_vall_{section}",
                            on_change=make_val_all_handler(section, available_cols))
                for col in available_cols:
                    st.checkbox(col, key=f"_valc_{section}_{col}",
                                on_change=make_val_col_handler(section, col))
            return n

        if text_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Validate Email**")
                st.caption("Flag adds a boolean column. Remove drops invalid rows.")
                email_action = st.radio("Action", ["Flag invalid", "Remove invalid rows"],
                                        key="email_action_radio", horizontal=True, label_visibility="collapsed")
            with c2:
                n_email = col_popover("email", text_cols)
            with c3:
                if st.button("Run", key="run_email_val", use_container_width=True,
                             disabled=n_email == 0, type="primary" if n_email > 0 else "secondary"):
                    try:
                        push_history("Validate Email")
                        action_key = 'flag' if 'Flag' in email_action else 'remove'
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["email"]:
                            df_temp = validate_email_col(df_temp, col, action=action_key)
                        st.session_state.current_df = df_temp
                        st.session_state.val_selected.pop("email", None)
                        st.session_state.last_success_msg = f"Email validation done on {n_email} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        if text_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Standardize Phone Numbers**")
                st.caption("Strips non-digit characters and formats to +[country code][number].")
            with c2:
                n_phone = col_popover("phone", text_cols)
            with c3:
                if st.button("Run", key="run_phone_val", use_container_width=True,
                             disabled=n_phone == 0, type="primary" if n_phone > 0 else "secondary"):
                    try:
                        push_history("Standardize Phone")
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["phone"]:
                            df_temp = validate_phone_col(df_temp, col)
                        st.session_state.current_df = df_temp
                        st.session_state.val_selected.pop("phone", None)
                        st.session_state.last_success_msg = f"Phone standardized in {n_phone} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        if text_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Standardize Dates**")
                date_fmt = st.selectbox("Output format",
                    ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y'], key="date_fmt_select")
            with c2:
                n_date = col_popover("date", text_cols)
            with c3:
                st.write("")
                if st.button("Run", key="run_date_val", use_container_width=True,
                             disabled=n_date == 0, type="primary" if n_date > 0 else "secondary"):
                    try:
                        push_history("Standardize Dates")
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["date"]:
                            df_temp = validate_date_col(df_temp, col, output_format=date_fmt)
                        st.session_state.current_df = df_temp
                        st.session_state.val_selected.pop("date", None)
                        st.session_state.last_success_msg = f"Dates standardized in {n_date} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        if num_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Cap / Remove Outliers**")
                oc1, oc2, oc3 = st.columns(3)
                with oc1:
                    outlier_method = st.selectbox("Method", ["iqr", "zscore"], key="outlier_method")
                with oc2:
                    outlier_action = st.selectbox("Action", ["cap", "remove"], key="outlier_action")
                with oc3:
                    outlier_thresh = st.number_input("Threshold", min_value=0.5, max_value=10.0,
                                                     value=1.5, step=0.5, key="outlier_thresh")
            with c2:
                n_outlier = col_popover("outlier", num_cols)
            with c3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_outlier", use_container_width=True,
                             disabled=n_outlier == 0, type="primary" if n_outlier > 0 else "secondary"):
                    try:
                        push_history("Cap Outliers")
                        before = len(st.session_state.current_df)
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["outlier"]:
                            df_temp = cap_outliers(df_temp, col=col, method=outlier_method,
                                                   action=outlier_action, threshold=outlier_thresh)
                        st.session_state.current_df = df_temp
                        after = len(st.session_state.current_df)
                        st.session_state.val_selected.pop("outlier", None)
                        if outlier_action == 'cap':
                            st.session_state.last_success_msg = f"Outliers capped in {n_outlier} column(s)!"
                        else:
                            st.session_state.last_success_msg = f"Removed {before - after} outlier rows!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        if num_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Validate Value Range**")
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    range_min = st.number_input("Min value", value=0.0, key="range_min")
                with rc2:
                    range_max = st.number_input("Max value", value=100.0, key="range_max")
                with rc3:
                    range_action = st.selectbox("Action", ["flag", "remove"], key="range_action")
            with c2:
                n_range = col_popover("range", num_cols)
            with c3:
                st.write("")
                st.write("")
                if st.button("Run", key="run_range_val", use_container_width=True,
                             disabled=n_range == 0, type="primary" if n_range > 0 else "secondary"):
                    try:
                        push_history("Validate Range")
                        before = len(st.session_state.current_df)
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["range"]:
                            df_temp = validate_range(df_temp, col=col,
                                                     min_val=range_min, max_val=range_max,
                                                     action=range_action)
                        st.session_state.current_df = df_temp
                        after = len(st.session_state.current_df)
                        st.session_state.val_selected.pop("range", None)
                        if range_action == 'flag':
                            st.session_state.last_success_msg = f"Range flagged across {n_range} column(s)!"
                        else:
                            st.session_state.last_success_msg = f"Removed {before - after} out-of-range rows!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        # cleaning history & undo
        st.subheader("Cleaning History")
        history = st.session_state.get('history', [])

        if not history:
            st.caption("No operations recorded yet. Every cleaning action is saved here.")
        else:
            st.caption(f"{len(history)} operation(s) recorded. Max 20 steps kept.")
            for i, step in enumerate(reversed(history)):
                step_num = len(history) - i
                st.write(f"**{step_num}.** {step['label']}  —  "
                         f"{step['df'].shape[0]} rows × {step['df'].shape[1]} cols")

            st.write("")
            col_undo, col_clear = st.columns(2)
            with col_undo:
                if st.button("⬅ Undo Last Step", key="undo_btn", use_container_width=True, type="primary"):
                    label = undo_last()
                    if label:
                        st.session_state.last_success_msg = f"Undone: {label}"
                    st.rerun()
            with col_clear:
                if st.button("Clear History", key="clear_history_btn", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

        st.divider()

        # data preview
        st.subheader("Data Preview")
        preview_rows = st.slider("Rows to display", 5, 50, 10, key="preview_rows_slider")
        st.dataframe(st.session_state.current_df.head(preview_rows), use_container_width=True)

        st.divider()

        with st.expander("Column Data Types & Information", expanded=False):
            col_types = pd.DataFrame({
                'Column': st.session_state.current_df.columns,
                'Type': st.session_state.current_df.dtypes.values,
                'Non-Null': st.session_state.current_df.count().values,
                'Null': st.session_state.current_df.isna().sum().values,
                'Unique': st.session_state.current_df.nunique().values
            })
            st.dataframe(col_types, use_container_width=True)

        st.divider()

        # pipeline export
        # turns the history into a reusable .py script the user can download and run on new files
        st.subheader("Export Cleaning Pipeline")
        st.caption("Download your cleaning steps as a Python script you can rerun on any new file.")

        history_for_export = st.session_state.get('history', [])
        if not history_for_export:
            st.caption("No steps recorded yet. Run some cleaning operations first.")
        else:
            script = build_pipeline_script(history_for_export)
            st.download_button(
                label="Download pipeline.py",
                data=script.encode('utf-8'),
                file_name="pipeline.py",
                mime="text/x-python",
                use_container_width=True,
                key="download_pipeline_btn"
            )
            with st.expander("Preview script", expanded=False):
                st.code(script, language='python')

        st.divider()

        # download cleaned data
        st.subheader("Download Cleaned Data")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV", data=csv,
                file_name="cleaned_data.csv", mime="text/csv",
                key="download_csv_button", use_container_width=True
            )
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.current_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                st.session_state.original_df.to_excel(writer, index=False, sheet_name='Original Data')
            st.download_button(
                label="Download as Excel", data=buffer.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button", use_container_width=True
            )
        with col3:
            if st.button("Reset to Original", key="reset_to_original_button", use_container_width=True):
                if 'original_df' in st.session_state:
                    st.session_state.current_df = st.session_state.original_df.copy()
                    st.session_state.selected_columns = {}
                    st.session_state.history = []
                    st.success("Data reset to original!")
                    st.rerun()

    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.info("Make sure your file is a valid CSV or Excel format.")

else:
    st.warning("Please upload a CSV or Excel file to begin cleaning.")

    with st.expander("Sample Data Format"):
        sample_df = pd.DataFrame({
            'name': [' Alice ', 'Bob', 'Charlie', 'Alice'],
            'price': ['$100', '$200.50', '€300', '$100'],
            'percentage': ['75%', '80.5%', '99%', '75%'],
            'weight': ['100kg', '150.5 lbs', '?', '100kg'],
            'duration': ['1h30m', '90min', 'NA', '1h30m']
        })
        st.dataframe(sample_df, use_container_width=True)
        st.caption("The pipeline can handle currency symbols, percentages, units, and missing values automatically!")