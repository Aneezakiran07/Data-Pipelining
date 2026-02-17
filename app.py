#Importing dependencies
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from typing import Optional, Dict, List, Tuple
import io

#setting up the page configuration
st.set_page_config(
    page_title="Advanced Data Cleaning Pipeline",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# ============================================================================
# DATA CLEANING FUNCTIONS (FROM PHASE 4)
# ============================================================================

def checking_valid_input(df:pd.DataFrame):
    if not isinstance(df,pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")

def drop_duplicate_rows(df:pd.DataFrame):
    checking_valid_input(df)  
    return df.drop_duplicates()

def drop_duplicate_columns(df:pd.DataFrame):
    checking_valid_input(df)  
    return df.drop_duplicates().T.drop_duplicates().T

def stripping_whitespace(df:pd.DataFrame):
    checking_valid_input(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

def clean_string_edges(
    df: pd.DataFrame,
    threshold: float = 0.7,
    inplace: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Intelligently trims edge characters when conditions are met."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        if verbose:
            st.warning("Warning: Empty DataFrame received")
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
                if keep_leading is False:
                    df_clean[col] = col_series.str.replace(r'^\W+', '', regex=True)
                    cleaned_cols.append(col)
            
            if not keep_trailing:
                if keep_trailing is False:
                    df_clean[col] = col_series.str.replace(r'\W+$', '', regex=True)
                    if col not in cleaned_cols:
                        cleaned_cols.append(col)
        
        if verbose and cleaned_cols:
            st.success(f"Cleaned string edges in {len(cleaned_cols)} columns")
        
        return None if inplace else df_clean
    except Exception as e:
        if verbose:
            st.error(f"Error during string cleaning: {str(e)}")
        raise

def smart_column_cleaner(
    df: pd.DataFrame,
    conversion_threshold: float = 0.6,
    inplace: bool = False,
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Ultimate smart column cleaner for numeric formats."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        if verbose:
            st.warning("Empty DataFrame received")
        return None if inplace else df.copy()

    try:
        df_clean = df if inplace else df.copy()

        # Define currency pattern
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

            # Currency Detection
            currency_like = non_empty.str.contains(r'\d', regex=True) & non_empty.str.contains(currency_pattern, case=False, regex=True)
            if currency_like.mean() > conversion_threshold:
                cleaned = (
                    non_empty.str.replace(r'[^\d.,\-()]', ' ', regex=True)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.replace(r'\((.+?)\)', r'-\1', regex=True)
                    .str.extract(r'([-]?\d[\d\.,]*)', expand=False)
                    .str.replace(',', '', regex=False)
                )

                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f" {col} (currency)")
                    continue

            # Percentage Detection
            if non_empty.str.contains('%').mean() > conversion_threshold:
                cleaned = non_empty.str.replace('%', '', regex=False)
                cleaned = cleaned.str.replace(r'[^\d.\-]', '', regex=True)
                converted = pd.to_numeric(cleaned, errors='coerce') / 100
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f" {col} (percentage)")
                    continue

            # Unit Detection
            unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz|gal|pt|°C|°F|kWh|cal|ha|ac|sqft|m²|km²)'
            if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
                cleaned = non_empty.str.extract(r'([-]?\d+\.?\d*)', expand=False)
                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f" {col} (unit)")
                    continue

            # Duration Detection
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
                    conversions.append(f" {col} (duration)")
                    continue

            # Generic Numeric Detection
            cleaned = non_empty.str.replace(r'[^\d.\-]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted
                conversions.append(f" {col} (numeric)")

        if verbose and conversions:
            st.success(f"Converted {len(conversions)} columns:")
            for conv in conversions:
                st.write(f" • {conv}")
        
        return None if inplace else df_clean

    except Exception as e:
        if verbose:
            st.error(f"Error during smart cleaning: {str(e)}")
        raise

def missing_value_handler(
    df: pd.DataFrame,
    threshold: float = 0.3,
    inplace: bool = False,
    numeric_strategy: str = 'auto',
    verbose: bool = False
) -> Optional[pd.DataFrame]:
    """Enhanced missing value handler with KNN/MICE imputation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    
    if df.empty:
        if verbose:
            st.warning("Warning: Empty DataFrame received")
        return None if inplace else df.copy()
    
    try:
        if not inplace:
            df_clean = df.copy()
        else:
            df_clean = df
        
        # Auto-switch to MICE if dataset is too large
        if numeric_strategy == 'auto':
            if df_clean.shape[1] > 50 or len(df_clean) > 5000:
                numeric_strategy = 'mice'
                if verbose:
                    st.info("Auto-switched to MICE (dataset exceeds 50 columns or 5,000 rows)")
        
        # Convert common missing indicators to NaN
        missing_indicators = ['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', -999, 999, 9999, '']
        df_clean.replace(missing_indicators, np.nan, inplace=True)
        
        # Drop columns with too many missing values
        missing_percent = df_clean.isna().mean()
        cols_to_drop = missing_percent[missing_percent > threshold].index
        if len(cols_to_drop) > 0:
            df_clean.drop(columns=cols_to_drop, inplace=True)
            if verbose:
                st.warning(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        cat_cols = df_clean.select_dtypes(exclude=np.number).columns
        
        # Numeric imputation
        if not numeric_cols.empty and df_clean[numeric_cols].isna().any().any():
            if numeric_strategy == 'knn' or (numeric_strategy == 'auto' and len(df_clean) <= 5000 and df_clean.shape[1] <= 50):
                if verbose:
                    st.info("Using KNN imputer for numeric columns")
                imputer = KNNImputer(n_neighbors=min(5, max(3, len(df_clean)//1000)))
            else:
                if verbose:
                    st.info("Using MICE imputer for numeric columns")
                imputer = IterativeImputer(max_iter=10, random_state=42)
            
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
        
        # Categorical imputation
        for col in cat_cols:
            if df_clean[col].isna().any():
                if df_clean[col].nunique() < 0.5 * len(df_clean):
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Missing'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                else:
                    df_clean[col] = df_clean[col].fillna('Missing')
        
        if verbose:
            st.success(f"Imputed {len(numeric_cols)} numeric and {len(cat_cols)} categorical columns")
        
        return None if inplace else df_clean
    
    except Exception as e:
        if verbose:
            st.error(f"Error during missing value handling: {str(e)}")
        raise

# ============================================================================
# PHASE 6: VALIDATION & QUALITY FUNCTIONS
# ============================================================================

def validate_email_col(
    df: pd.DataFrame,
    col: str,
    action: str = 'flag'   # 'flag' adds a boolean column, 'remove' drops invalid rows
) -> pd.DataFrame:
    """Validates email format in a column. Flags or removes invalid entries."""
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


def validate_phone_col(
    df: pd.DataFrame,
    col: str,
    output_format: str = '+1XXXXXXXXXX'  # target format hint shown in UI
) -> pd.DataFrame:
    """Strips all non-digit characters and standardizes phone numbers."""
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

    df_clean = df.copy()

    def standardize_phone(val):
        digits = re.sub(r'\D', '', str(val))
        if len(digits) == 10:
            return f'+1{digits}'           # assume US if 10 digits
        elif len(digits) == 11 and digits.startswith('1'):
            return f'+{digits}'
        elif len(digits) >= 7:
            return f'+{digits}'
        return np.nan                       # too short to be a real number

    df_clean[col] = df_clean[col].apply(standardize_phone)
    return df_clean


def validate_date_col(
    df: pd.DataFrame,
    col: str,
    output_format: str = '%Y-%m-%d'
) -> pd.DataFrame:
    """Detects and parses mixed date formats into one consistent format."""
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")

    df_clean = df.copy()

    # Try every common format explicitly — most reliable approach
    formats_to_try = [
        '%Y-%m-%d',       # 2023-01-15
        '%d/%m/%Y',       # 15/03/2022
        '%m/%d/%Y',       # 03/15/2022
        '%Y/%m/%d',       # 2022/08/30
        '%d-%m-%Y',       # 28-07-2020
        '%m-%d-%Y',       # 03-15-2022
        '%d %b %Y',       # 15 Jan 2023
        '%d %B %Y',       # 15 January 2023
        '%b %d %Y',       # Jan 15 2023
        '%B %d %Y',       # January 15 2023
        '%b %d, %Y',      # Jan 15, 2023
        '%B %d, %Y',      # January 15, 2023
        '%d/%m/%y',       # 15/03/22
        '%m/%d/%y',       # 03/15/22
        '%d-%m-%y',       # 15-03-22
        '%Y.%m.%d',       # 2023.01.15
        '%d.%m.%Y',       # 15.01.2023
    ]

    def parse_single(val):
        if pd.isna(val) or str(val).strip() == '':
            return pd.NaT
        val_str = str(val).strip()
        # Try each explicit format first
        for fmt in formats_to_try:
            try:
                return pd.to_datetime(val_str, format=fmt)
            except (ValueError, TypeError):
                continue
        # Last resort: let pandas guess
        try:
            return pd.to_datetime(val_str)
        except (ValueError, TypeError):
            return pd.NaT

    parsed = df_clean[col].apply(parse_single)
    df_clean[col] = parsed.dt.strftime(output_format).where(parsed.notna(), other=np.nan)
    return df_clean


def cap_outliers(
    df: pd.DataFrame,
    col: str,
    method: str = 'iqr',    # 'iqr' or 'zscore'
    action: str = 'cap',    # 'cap' clips to boundary, 'remove' drops the row
    threshold: float = 1.5  # IQR multiplier or Z-score cutoff
) -> pd.DataFrame:
    """Detects outliers via IQR or Z-score and caps or removes them."""
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
    else:  # zscore
        mean, std = series.mean(), series.std()
        lower = mean - threshold * std
        upper = mean + threshold * std

    if action == 'cap':
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    elif action == 'remove':
        mask = df_clean[col].isna() | ((df_clean[col] >= lower) & (df_clean[col] <= upper))
        df_clean = df_clean[mask]

    return df_clean


def validate_range(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    action: str = 'flag'    # 'flag' adds boolean col, 'remove' drops invalid rows
) -> pd.DataFrame:
    """Flags or removes rows where column value is outside [min_val, max_val]."""
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


# ============================================================================
# PHASE 5: SMART RECOMMENDATIONS ENGINE
# ============================================================================

def analyze_data_issues(df: pd.DataFrame, conversion_threshold: float = 0.6) -> Dict[str, any]:
    """Analyzes the dataframe and identifies potential issues."""
    issues = {
        'duplicate_rows': 0,
        'duplicate_cols': 0,
        'whitespace_cols': [],
        'currency_cols': [],
        'percentage_cols': [],
        'unit_cols': [],
        'duration_cols': [],
        'missing_cells': 0,
        'missing_cols': [],
        'edge_char_cols': []
    }
    
    # Check duplicates
    issues['duplicate_rows'] = df.duplicated().sum()
    
    # Check duplicate columns
    issues['duplicate_cols'] = len(df.columns) - len(df.T.drop_duplicates().T.columns)
    
    # Check whitespace
    for col in df.select_dtypes(include='object').columns:
        if df[col].astype(str).str.strip().ne(df[col].astype(str)).any():
            issues['whitespace_cols'].append(col)
    
    # Check for currency, percentage, units, duration
    currency_symbols = r'[$€£¥₹₽₺₩฿₡₦₲₴₵₸₳₻₼₽₾₿]'
    currency_codes = r'(USD|EUR|GBP|JPY|CNY|INR|RUB|AUD|CAD|PKR)'
    currency_pattern = f'{currency_symbols}|{currency_codes}'
    
    for col in df.select_dtypes(include='object').columns:
        series = df[col].astype(str).str.strip()
        non_empty = series.replace('', np.nan).dropna()
        
        if non_empty.empty:
            continue
        
        # Currency check
        currency_like = non_empty.str.contains(r'\d', regex=True) & non_empty.str.contains(currency_pattern, case=False, regex=True)
        if currency_like.mean() > conversion_threshold:
            issues['currency_cols'].append(col)
            continue
        
        # Percentage check
        if non_empty.str.contains('%').mean() > conversion_threshold:
            issues['percentage_cols'].append(col)
            continue
        
        # Unit check
        unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz)'
        if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
            issues['unit_cols'].append(col)
            continue
        
        # Duration check
        duration_pattern = r'(h|hr|hour|min|minute|sec|second)'
        if non_empty.str.contains(duration_pattern, case=False, regex=True).mean() > conversion_threshold:
            issues['duration_cols'].append(col)
    
    # Check missing values
    issues['missing_cells'] = df.isna().sum().sum()
    for col in df.columns:
        missing_pct = df[col].isna().mean()
        if missing_pct > 0:
            issues['missing_cols'].append((col, missing_pct))
    
    # Check edge characters
    for col in df.select_dtypes(include='object').columns:
        col_series = df[col].astype(str)
        has_edge_chars = col_series.str.match(r'^\W') | col_series.str.match(r'\W$')
        if has_edge_chars.sum() > 0:
            issues['edge_char_cols'].append(col)
    
    return issues

def generate_recommendations(issues: Dict[str, any]) -> List[Tuple[str, str, str, str]]:
    """Generates actionable recommendations based on detected issues.
    Returns list of (icon, title, description, action_key) tuples."""
    recommendations = []
    
    # Duplicate rows
    if issues['duplicate_rows'] > 0:
        recommendations.append((
            "",
            f"Found {issues['duplicate_rows']} duplicate rows",
            f"Removing duplicates will reduce your dataset size and prevent skewed analysis.",
            "drop_duplicates"
        ))
    
    # Duplicate columns
    if issues['duplicate_cols'] > 0:
        recommendations.append((
            "",
            f"Found {issues['duplicate_cols']} duplicate columns",
            "These columns are wasting memory and processing power.",
            "drop_dup_cols"
        ))
    
    # Whitespace
    if len(issues['whitespace_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['whitespace_cols'])} columns have extra whitespace",
            f"Columns: {', '.join(issues['whitespace_cols'][:3])}{'...' if len(issues['whitespace_cols']) > 3 else ''}",
            "strip_whitespace"
        ))
    
    # Currency columns
    if len(issues['currency_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['currency_cols'])} columns look like currency but aren't numeric",
            f"Columns: {', '.join(issues['currency_cols'][:3])}{'...' if len(issues['currency_cols']) > 3 else ''}. Convert them to do calculations.",
            "convert_currency"
        ))
    
    # Percentage columns
    if len(issues['percentage_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['percentage_cols'])} columns contain percentages as text",
            f"Columns: {', '.join(issues['percentage_cols'][:3])}{'...' if len(issues['percentage_cols']) > 3 else ''}. Should be decimals for math.",
            "convert_percentage"
        ))
    
    # Unit columns
    if len(issues['unit_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['unit_cols'])} columns have measurement units mixed with numbers",
            f"Columns: {', '.join(issues['unit_cols'][:3])}{'...' if len(issues['unit_cols']) > 3 else ''}",
            "convert_units"
        ))
    
    # Duration columns
    if len(issues['duration_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['duration_cols'])} columns contain time durations",
            f"Columns: {', '.join(issues['duration_cols'][:3])}{'...' if len(issues['duration_cols']) > 3 else ''}. Convert to seconds for consistency.",
            "convert_duration"
        ))
    
    # Missing values
    if issues['missing_cells'] > 0:
        top_missing = sorted(issues['missing_cols'], key=lambda x: x[1], reverse=True)[:3]
        col_details = ', '.join([f"{col} ({pct*100:.0f}%)" for col, pct in top_missing])
        recommendations.append((
            "",
            f"{issues['missing_cells']} missing values found",
            f"Worst affected: {col_details}. Use ML imputation to fill them intelligently.",
            "handle_missing"
        ))
    
    # Edge characters
    if len(issues['edge_char_cols']) > 0:
        recommendations.append((
            "",
            f"{len(issues['edge_char_cols'])} columns have unwanted edge characters",
            f"Columns: {', '.join(issues['edge_char_cols'][:3])}{'...' if len(issues['edge_char_cols']) > 3 else ''}",
            "clean_edges"
        ))
    
    return recommendations

# Helper function to get statistics
def get_dataframe_stats(df):
    """Returns key statistics about the DataFrame."""
    return {
        'rows': df.shape[0],
        'columns': df.shape[1],
        'missing_cells': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=np.number).columns),
        'categorical_cols': len(df.select_dtypes(exclude=np.number).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown('<p class="main-header">Advanced Data Cleaning Pipeline</p>', unsafe_allow_html=True)
st.markdown("Upload a CSV or Excel file and get smart recommendations on how to clean your data.")
st.markdown("**Don't have a file? Download the test_data CSV from [this GitHub repo](https://github.com/Aneezakiran07/Data-Pipelining) and upload it here to check all functionalities.**")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Missing Value Handler")
    missing_threshold = st.slider(
        "Drop columns with missing % >",
        0, 100, 30,
        key="missing_threshold_slider",
        help="Columns with more than this percentage of missing values will be dropped"
    ) / 100
    
    numeric_strategy = st.selectbox(
        "Numeric Imputation Strategy",
        ['auto', 'knn', 'mice'],
        key="numeric_strategy_select",
        help="auto: KNN for small datasets, MICE for large ones"
    )
    
    st.subheader("Smart Cleaner")
    conversion_threshold = st.slider(
        "Conversion Threshold %",
        0, 100, 60,
        key="conversion_threshold_slider",
        help="Minimum percentage of values that must be convertible"
    ) / 100
    
    st.divider()
    
    if st.button("Reset All", key="reset_all_button", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.divider()

st.subheader("Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx', 'xls'],
    key="file_uploader_main",
    help="Upload a CSV or Excel file that needs cleaning"
)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    try:
        # Read file based on extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file, quotechar='"', skipinitialspace=True)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            st.stop()

        # Initialize session state
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()
            st.session_state.current_df = df.copy()
            st.session_state.original_stats = get_dataframe_stats(df)
        
        # Get current statistics
        current_stats = get_dataframe_stats(st.session_state.current_df)
        
        st.info(f"DataFrame: {current_stats['rows']} rows × {current_stats['columns']} columns | "
               f"Memory: {current_stats['memory_usage']:.2f} MB")

        # Statistics Dashboard
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

        # ============================================================================
        # PHASE 5: SMART RECOMMENDATIONS SECTION
        # ============================================================================
        
        st.subheader("Smart Recommendations")
        
        # Analyze data and generate recommendations
        with st.spinner("Analyzing your data..."):
            issues = analyze_data_issues(st.session_state.current_df, conversion_threshold)
            recommendations = generate_recommendations(issues)
        
        if len(recommendations) == 0:
            if st.session_state.get('last_success_msg'):
                st.success(st.session_state.last_success_msg)
                st.session_state.last_success_msg = None
            st.success("Your data looks clean! No issues detected.")
        else:
            st.warning(f"Found {len(recommendations)} potential issues in your data")
            
            # Show success message from previous fix action
            if st.session_state.get('last_success_msg'):
                st.success(st.session_state.last_success_msg)
                st.session_state.last_success_msg = None
            
            # Initialize session state for column selections
            if 'selected_columns' not in st.session_state:
                st.session_state.selected_columns = {}

            for idx, (icon, title, description, action_key) in enumerate(recommendations):

                # Get affected columns for this recommendation
                affected_columns = []
                if action_key == "strip_whitespace":
                    affected_columns = issues['whitespace_cols']
                elif action_key == "convert_currency":
                    affected_columns = issues['currency_cols']
                elif action_key == "convert_percentage":
                    affected_columns = issues['percentage_cols']
                elif action_key == "convert_units":
                    affected_columns = issues['unit_cols']
                elif action_key == "convert_duration":
                    affected_columns = issues['duration_cols']
                elif action_key == "clean_edges":
                    affected_columns = issues['edge_char_cols']
                elif action_key == "handle_missing":
                    affected_columns = [col for col, pct in issues['missing_cols']]

                # How many columns user has selected for this action
                # Read from clean tracking dict, not widget keys
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

                            # on_change handlers write to selected_columns dict only
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

                            # Apply to all — no value= so Streamlit owns state fully
                            st.checkbox(
                                "Apply to all columns",
                                key=f"_widget_all_{action_key}",
                                on_change=make_apply_all_handler(action_key, affected_columns)
                            )

                            # Individual checkboxes — no value= 
                            for col in affected_columns:
                                st.checkbox(
                                    col,
                                    key=f"_widget_chk_{action_key}_{col}",
                                    on_change=make_col_handler(action_key, col)
                                )

                    with col3:
                        has_selection = num_selected > 0
                        if st.button(
                            "Fix This",
                            key=f"fix_{action_key}",
                            use_container_width=True,
                            disabled=not has_selection,
                            type="primary" if has_selection else "secondary"
                        ):
                            selected_cols = st.session_state.selected_columns.get(action_key, [])
                            try:
                                with st.spinner(f"Fixing {len(selected_cols)} column(s)..."):
                                    df_temp = st.session_state.current_df.copy()

                                    if action_key == "strip_whitespace":
                                        for col in selected_cols:
                                            if col in df_temp.columns and df_temp[col].dtype == 'object':
                                                df_temp[col] = df_temp[col].str.strip()

                                    elif action_key in ["convert_currency", "convert_percentage", "convert_units", "convert_duration"]:
                                        for col in selected_cols:
                                            if col not in df_temp.columns:
                                                continue
                                            series = df_temp[col].astype(str).str.strip()
                                            non_empty = series.replace('', np.nan).dropna()
                                            if non_empty.empty:
                                                continue

                                            if action_key == "convert_currency":
                                                cleaned = (
                                                    non_empty.str.replace(r'[^\d.,\-()]', ' ', regex=True)
                                                    .str.replace(r'\s+', ' ', regex=True)
                                                    .str.replace(r'\((.+?)\)', r'-\1', regex=True)
                                                    .str.extract(r'([-]?\d[\d\.,]*)', expand=False)
                                                    .str.replace(',', '', regex=False)
                                                )
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
                                                df_temp[col] = df_temp[col].astype(str).str.replace(r'^\W+', '', regex=True).str.replace(r'\W+$', '', regex=True)

                                    elif action_key == "handle_missing":
                                        # Guard: only keep cols that still exist in df
                                        valid_cols = [c for c in selected_cols if c in df_temp.columns]
                                        if valid_cols:
                                            df_subset = df_temp[valid_cols].copy()
                                            df_subset = missing_value_handler(df_subset, threshold=missing_threshold, numeric_strategy=numeric_strategy, verbose=False)
                                            for c in valid_cols:
                                                if c in df_subset.columns:
                                                    df_temp[c] = df_subset[c]

                                    st.session_state.current_df = df_temp
                                    st.session_state.selected_columns.pop(action_key, None)
                                    # Store message before rerun so it survives
                                    if action_key == "strip_whitespace":
                                        st.session_state.last_success_msg = f"Stripped whitespace from {len(selected_cols)} column(s)!"
                                    elif action_key in ["convert_currency", "convert_percentage", "convert_units", "convert_duration"]:
                                        st.session_state.last_success_msg = f"Converted {len(selected_cols)} column(s)!"
                                    elif action_key == "clean_edges":
                                        st.session_state.last_success_msg = f"Cleaned edges in {len(selected_cols)} column(s)!"
                                    elif action_key == "handle_missing":
                                        st.session_state.last_success_msg = f"Handled missing values in {len(valid_cols)} column(s)!"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                else:
                    # No specific columns (duplicate rows/cols) — just Fix This directly
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"{icon} **{title}**")
                        st.caption(description)
                    with col2:
                        if st.button("Fix This", key=f"fix_{action_key}", use_container_width=True, type="primary"):
                            try:
                                with st.spinner("Fixing..."):
                                    if action_key == "drop_duplicates":
                                        st.session_state.current_df = drop_duplicate_rows(st.session_state.current_df)
                                        st.session_state.last_success_msg = "Duplicate rows removed!"
                                    elif action_key == "drop_dup_cols":
                                        st.session_state.current_df = drop_duplicate_columns(st.session_state.current_df)
                                        st.session_state.last_success_msg = "Duplicate columns removed!"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                st.write("")  # spacing

            # Auto-fix all button
            if st.button("Auto-Fix All Issues", key="auto_fix_all", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Running complete cleaning pipeline..."):
                        df_temp = st.session_state.current_df.copy()
                        df_temp = stripping_whitespace(df_temp)
                        df_temp = drop_duplicate_rows(df_temp)
                        df_temp = drop_duplicate_columns(df_temp)
                        df_temp = clean_string_edges(df_temp, threshold=0.7, verbose=False)
                        df_temp = smart_column_cleaner(df_temp, conversion_threshold=conversion_threshold, verbose=False)
                        df_temp = missing_value_handler(df_temp, threshold=missing_threshold, numeric_strategy=numeric_strategy, verbose=False)
                        st.session_state.current_df = df_temp
                        st.session_state.selected_columns = {}
                        st.session_state.last_success_msg = "All issues fixed automatically!"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()

        # ============================================================================
        # MANUAL CLEANING OPERATIONS (FROM PHASE 4)
        # ============================================================================

        st.subheader("Manual Cleaning Operations")
        st.markdown("Or choose specific operations to perform manually:")

        # Row 1: Basic cleaning
        st.write("**Basic Cleaning**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Strip Whitespace", key="manual_strip_ws", use_container_width=True, help="Remove leading/trailing spaces"):
                try:
                    st.session_state.current_df = stripping_whitespace(st.session_state.current_df)
                    st.success("Whitespace stripped!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Drop Duplicate Rows", key="manual_drop_dup_rows", use_container_width=True):
                try:
                    before_rows = st.session_state.current_df.shape[0]
                    st.session_state.current_df = drop_duplicate_rows(st.session_state.current_df)
                    after_rows = st.session_state.current_df.shape[0]
                    st.success(f"Dropped {before_rows - after_rows} duplicate rows!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col3:
            if st.button("Drop Duplicate Columns", key="manual_drop_dup_cols", use_container_width=True):
                try:
                    before_cols = st.session_state.current_df.shape[1]
                    st.session_state.current_df = drop_duplicate_columns(st.session_state.current_df)
                    after_cols = st.session_state.current_df.shape[1]
                    st.success(f"Dropped {before_cols - after_cols} duplicate columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col4:
            if st.button("Clean String Edges", key="manual_clean_edges", use_container_width=True, help="Remove unwanted edge characters"):
                try:
                    with st.spinner("Cleaning string edges..."):
                        st.session_state.current_df = clean_string_edges(
                            st.session_state.current_df,
                            threshold=0.7,
                            verbose=True
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("")  # Spacing

        # Row 2: Advanced cleaning
        st.write("**Advanced Cleaning**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Smart Column Cleaner", key="manual_smart_cleaner", use_container_width=True,
                        help="Auto-detect and convert currency, percentages, units, durations"):
                try:
                    with st.spinner("Analyzing and converting columns..."):
                        st.session_state.current_df = smart_column_cleaner(
                            st.session_state.current_df,
                            conversion_threshold=conversion_threshold,
                            verbose=True
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Handle Missing Values", key="manual_missing_handler", use_container_width=True,
                        help="Intelligent imputation using KNN/MICE"):
                try:
                    with st.spinner("Handling missing values (this may take a moment)..."):
                        st.session_state.current_df = missing_value_handler(
                            st.session_state.current_df,
                            threshold=missing_threshold,
                            numeric_strategy=numeric_strategy,
                            verbose=True
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.write("")

        # Row 3: Validation & Quality
        st.write("**Validation & Quality**")

        # Initialize validation selection state
        if 'val_selected' not in st.session_state:
            st.session_state.val_selected = {}

        text_cols  = list(st.session_state.current_df.select_dtypes(include='object').columns)
        num_cols   = list(st.session_state.current_df.select_dtypes(include=np.number).columns)

        # Helper to build on_change handlers for validation popovers
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
            """Renders the column selector popover and returns selected count."""
            n = len(st.session_state.val_selected.get(section, []))
            label = f"v {n} selected" if n > 0 else "v Select columns"
            with st.popover(label, use_container_width=True):
                st.caption("Select columns to apply this operation")
                st.checkbox(
                    "Apply to all",
                    key=f"_vall_{section}",
                    on_change=make_val_all_handler(section, available_cols)
                )
                for col in available_cols:
                    st.checkbox(
                        col,
                        key=f"_valc_{section}_{col}",
                        on_change=make_val_col_handler(section, col)
                    )
            return n

        # --- Email Validation ---
        if text_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Validate Email**")
                st.caption("Checks email format. Flag adds a boolean column, Remove drops invalid rows.")
                email_action = st.radio("Action", ["Flag invalid", "Remove invalid rows"],
                                        key="email_action_radio", horizontal=True, label_visibility="collapsed")
            with c2:
                n_email = col_popover("email", text_cols)
            with c3:
                if st.button("Run", key="run_email_val", use_container_width=True,
                             disabled=n_email == 0, type="primary" if n_email > 0 else "secondary"):
                    try:
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

        # --- Phone Standardization ---
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
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["phone"]:
                            df_temp = validate_phone_col(df_temp, col)
                        st.session_state.current_df = df_temp
                        st.session_state.val_selected.pop("phone", None)
                        st.session_state.last_success_msg = f"Phone numbers standardized in {n_phone} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        # --- Date Standardization ---
        if text_cols:
            c1, c2, c3 = st.columns([5, 1.4, 1])
            with c1:
                st.write("**Standardize Dates**")
                date_fmt = st.selectbox(
                    "Output format",
                    ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y'],
                    key="date_fmt_select"
                )
            with c2:
                n_date = col_popover("date", text_cols)
            with c3:
                st.write("")
                if st.button("Run", key="run_date_val", use_container_width=True,
                             disabled=n_date == 0, type="primary" if n_date > 0 else "secondary"):
                    try:
                        df_temp = st.session_state.current_df.copy()
                        for col in st.session_state.val_selected["date"]:
                            df_temp = validate_date_col(df_temp, col, output_format=date_fmt)
                        st.session_state.current_df = df_temp
                        st.session_state.val_selected.pop("date", None)
                        st.session_state.last_success_msg = f"Dates standardized to {date_fmt} in {n_date} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        # --- Cap Outliers ---
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
                            st.session_state.last_success_msg = f"Removed {before - after} outlier rows across {n_outlier} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.write("")

        # --- Validate Range ---
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
                            st.session_state.last_success_msg = f"Removed {before - after} out-of-range rows across {n_range} column(s)!"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.divider()

        # Data preview
        st.subheader("Data Preview")
        preview_rows = st.slider("Rows to display", 5, 50, 10, key="preview_rows_slider")
        st.dataframe(st.session_state.current_df.head(preview_rows), use_container_width=True)

        st.divider()
        
        # Column information expander
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

        # Download section
        st.subheader("Download Cleaned Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                key="download_csv_button",
                use_container_width=True
            )
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.current_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                st.session_state.original_df.to_excel(writer, index=False, sheet_name='Original Data')
            
            st.download_button(
                label="Download as Excel",
                data=buffer.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button",
                use_container_width=True
            )
        
        with col3:
            if st.button("Reset to Original", key="reset_to_original_button", use_container_width=True):
                # Only reset the dataframe, don't clear the file uploader
                if 'original_df' in st.session_state:
                    st.session_state.current_df = st.session_state.original_df.copy()
                    # Clear column selection state
                    if 'show_columns_for' in st.session_state:
                        st.session_state.show_columns_for = None
                    if 'selected_columns' in st.session_state:
                        st.session_state.selected_columns = {}
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