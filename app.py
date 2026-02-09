#Importing dependencies
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional
import io

#setting up the page configuration
st.set_page_config(
    page_title="Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide"
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

#python functions for data cleaning
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
            st.success(f"✅ Cleaned string edges in {len(cleaned_cols)} columns: {', '.join(cleaned_cols[:5])}")
        
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
            st.warning("⚠️ Empty DataFrame received")
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
                    conversions.append(f"💰 {col} (currency)")
                    continue

            # Percentage Detection
            if non_empty.str.contains('%').mean() > conversion_threshold:
                cleaned = non_empty.str.replace('%', '', regex=False)
                cleaned = cleaned.str.replace(r'[^\d.\-]', '', regex=True)
                converted = pd.to_numeric(cleaned, errors='coerce') / 100
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f"📉 {col} (percentage)")
                    continue

            # Unit Detection
            unit_pattern = r'\d+\s?(kg|g|mg|cm|mm|m|km|ml|l|lb|oz|gal|pt|°C|°F|kWh|cal|ha|ac|sqft|m²|km²)'
            if non_empty.str.contains(unit_pattern, case=False, regex=True).mean() > conversion_threshold:
                cleaned = non_empty.str.extract(r'([-]?\d+\.?\d*)', expand=False)
                converted = pd.to_numeric(cleaned, errors='coerce')
                if converted.notna().mean() > conversion_threshold:
                    df_clean[col] = converted
                    conversions.append(f"📏 {col} (unit)")
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
                    conversions.append(f"⏱️ {col} (duration → seconds)")
                    continue

            # Generic Numeric Detection
            cleaned = non_empty.str.replace(r'[^\d.\-]', '', regex=True)
            converted = pd.to_numeric(cleaned, errors='coerce')
            if converted.notna().mean() > conversion_threshold:
                df_clean[col] = converted
                conversions.append(f"🔢 {col} (numeric)")

        if verbose and conversions:
            st.success(f"✅ Converted {len(conversions)} columns:")
            for conv in conversions:
                st.write(f"  • {conv}")
        
        return None if inplace else df_clean

    except Exception as e:
        if verbose:
            st.error(f"🚨 Error during smart cleaning: {str(e)}")
        raise

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

#streamlit logic
st.markdown('<p class="main-header">🧹 Advanced Data Cleaning Pipeline</p>', unsafe_allow_html=True)
st.markdown("Upload a CSV file and apply intelligent cleaning operations to prepare your data for analysis.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Smart Cleaner")
    conversion_threshold = st.slider(
        "Conversion Threshold %",
        0, 100, 60,
        help="Minimum percentage of values that must be convertible"
    ) / 100
    
    st.divider()
    
    if st.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

st.divider()

st.subheader("📁 Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file that needs cleaning"
)

if uploaded_file is not None:

    st.success("✅ File uploaded successfully!")

    try:
        df = pd.read_csv(uploaded_file)

        # Initialize session state
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()
            st.session_state.current_df = df.copy()
            st.session_state.original_stats = get_dataframe_stats(df)
        
        # Get current statistics
        current_stats = get_dataframe_stats(st.session_state.current_df)
        
        st.info(f"📊 DataFrame: {current_stats['rows']} rows × {current_stats['columns']} columns | "
               f"💾 Memory: {current_stats['memory_usage']:.2f} MB")

        # Statistics Dashboard
        st.subheader("📈 Data Statistics")
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

        st.subheader("👀 Data Preview")
        preview_rows = st.slider("Rows to display", 5, 50, 10)
        st.dataframe(st.session_state.current_df.head(preview_rows), use_container_width=True)

        st.divider()

        # Cleaning operations
        st.subheader("🛠️ Cleaning Operations")
        st.markdown("Select the cleaning operations you want to perform:")

        # Row 1: Basic cleaning
        st.write("**Basic Cleaning**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("✂️ Strip Whitespace", use_container_width=True):
                try:
                    st.session_state.current_df = stripping_whitespace(st.session_state.current_df)
                    st.success("✅ Whitespace stripped!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Drop Duplicate Rows", use_container_width=True):
                try:
                    before_rows = st.session_state.current_df.shape[0]
                    st.session_state.current_df = drop_duplicate_rows(st.session_state.current_df)
                    after_rows = st.session_state.current_df.shape[0]
                    st.success(f"✅ Dropped {before_rows - after_rows} duplicate rows!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button("📋 Drop Duplicate Columns", use_container_width=True):
                try:
                    before_cols = st.session_state.current_df.shape[1]
                    st.session_state.current_df = drop_duplicate_columns(st.session_state.current_df)
                    after_cols = st.session_state.current_df.shape[1]
                    st.success(f"✅ Dropped {before_cols - after_cols} duplicate columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col4:
            if st.button("🧽 Clean String Edges", use_container_width=True):
                try:
                    with st.spinner("Cleaning string edges..."):
                        st.session_state.current_df = clean_string_edges(
                            st.session_state.current_df,
                            threshold=0.7,
                            verbose=True
                        )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        st.write("")  # Spacing

        # Row 2: Advanced cleaning
        st.write("**Advanced Cleaning**")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔢 Smart Column Cleaner", use_container_width=True,
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
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🚀 Run Basic Pipeline", use_container_width=True,
                        help="Strip whitespace → Remove duplicates → Clean strings → Smart convert"):
                try:
                    with st.spinner("Running basic cleaning pipeline..."):
                        df_temp = st.session_state.current_df.copy()
                        df_temp = stripping_whitespace(df_temp)
                        df_temp = drop_duplicate_rows(df_temp)
                        df_temp = drop_duplicate_columns(df_temp)
                        df_temp = clean_string_edges(df_temp, threshold=0.7, verbose=False)
                        df_temp = smart_column_cleaner(df_temp, conversion_threshold=conversion_threshold, verbose=False)
                        st.session_state.current_df = df_temp
                        st.success("✅ Basic pipeline completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        st.divider()
        
        # Column information expander
        with st.expander("📊 Column Data Types & Information", expanded=False):
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
        st.subheader("💾 Download Cleaned Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.current_df.to_excel(writer, index=False, sheet_name='Cleaned Data')
                st.session_state.original_df.to_excel(writer, index=False, sheet_name='Original Data')
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name="cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Reset to Original", use_container_width=True):
                st.session_state.current_df = st.session_state.original_df.copy()
                st.success("✅ Data reset!")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.info("💡 Make sure your file is a valid CSV format.")

else:
    st.warning("⏳ Please upload a CSV file to begin cleaning.")
    
    with st.expander("📋 Sample Data Format"):
        sample_df = pd.DataFrame({
            'name': ['  Alice  ', 'Bob', 'Charlie'],
            'price': ['$100', '€200.50', '£300'],
            'percentage': ['75%', '80.5%', '99%'],
            'weight': ['100kg', '150.5 lbs', '200g'],
            'duration': ['1h30m', '90min', '2 hours']
        })
        st.dataframe(sample_df, use_container_width=True)
        st.caption("The pipeline can handle currency symbols, percentages, units, and durations automatically!")

         