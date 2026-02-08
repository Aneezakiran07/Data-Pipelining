#Importing dependencies
import streamlit as st
import pandas as pd
import numpy as np
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
def checking_valid_input(
        df:pd.DataFrame
):
    if not isinstance(df,pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame is empty")
    

def drop_duplicate_rows(
        df:pd.DataFrame
):
    checking_valid_input(df)  
    return df.drop_duplicates()

def drop_duplicate_columns(
        df:pd.DataFrame
):
    checking_valid_input(df)  
    return df.drop_duplicates().T.drop_duplicates().T

def stripping_whitespace(
        df:pd.DataFrame
):
    checking_valid_input(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    

def stripping_character_user_wants(
        df:pd.DataFrame,
        colname:str,
        ch_toremove:str
):
    checking_valid_input(df)
    if colname in df.columns:
        df[colname] = df[colname].astype(str).str.replace(ch_toremove, "", regex=False)
    return df

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
st.markdown('<p class="main-header">🧹 Data Cleaning Pipeline</p>', unsafe_allow_html=True)
st.markdown("Upload a csv file to clean it using the pipeline.")

st.divider()

st.subheader("📁 Upload your data file")
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing missing values etc"
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
            st.metric(
                "Rows", 
                current_stats['rows'],
                delta=current_stats['rows'] - st.session_state.original_stats['rows'],
                delta_color="inverse"
            )
            st.metric(
                "Columns", 
                current_stats['columns'],
                delta=current_stats['columns'] - st.session_state.original_stats['columns'],
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                "Missing Cells", 
                current_stats['missing_cells'],
                delta=current_stats['missing_cells'] - st.session_state.original_stats['missing_cells'],
                delta_color="inverse"
            )
            st.metric(
                "Duplicate Rows", 
                current_stats['duplicate_rows'],
                delta=current_stats['duplicate_rows'] - st.session_state.original_stats['duplicate_rows'],
                delta_color="inverse"
            )
        
        with col3:
            st.metric("Numeric Columns", current_stats['numeric_cols'])
            st.metric("Categorical Columns", current_stats['categorical_cols'])

        st.divider()

        st.subheader("👀 Data Preview")
        preview_rows = st.slider("Rows to display", 5, 50, 10)
        st.dataframe(st.session_state.current_df.head(preview_rows), use_container_width=True)

        st.divider()

        st.subheader("🛠️ Cleaning Operations")
        st.markdown("Choose the cleaning operations to perform on your data")

        # Create columns for buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✂️ Strip Whitespace", use_container_width=True):
                try:
                    st.session_state.current_df = stripping_whitespace(st.session_state.current_df)
                    st.success("✅ Whitespace stripped from text columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🗑️ Drop Duplicate Rows", use_container_width=True):
                try:
                    before_rows = st.session_state.current_df.shape[0]
                    st.session_state.current_df = drop_duplicate_rows(st.session_state.current_df)
                    after_rows = st.session_state.current_df.shape[0]
                    st.success(f"✅ Dropped {before_rows-after_rows} duplicate rows!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col3:
            if st.button("📋 Drop Duplicate Columns", use_container_width=True):
                try:
                    before_cols = st.session_state.current_df.shape[1]
                    st.session_state.current_df = drop_duplicate_columns(st.session_state.current_df)
                    after_cols = st.session_state.current_df.shape[1]
                    st.success(f"✅ Dropped {before_cols-after_cols} duplicate columns!")
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
            # Download as CSV
            csv = st.session_state.current_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Download as Excel
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
            # Reset button
            if st.button("🔄 Reset to Original", use_container_width=True):
                st.session_state.current_df = st.session_state.original_df.copy()
                st.success("✅ Data reset to original!")
                st.rerun()

    except Exception as e:
        st.error(f"❌ Error reading the file: {e}")
        st.info("💡 Make sure your file is a valid CSV format.")

else:
    st.warning("⏳ Please upload a CSV file to begin cleaning.")
    
    # Show sample data format
    with st.expander("📋 Sample Data Format"):
        sample_df = pd.DataFrame({
            'name': ['  Alice  ', 'Bob', 'Charlie', 'Alice'],
            'price': ['$100', '$200.50', '€300', '$100'],
            'status': ['active', 'active', 'inactive', 'active']
        })
        st.dataframe(sample_df, use_container_width=True)
        st.caption("Upload a CSV file with similar structure to get started!")