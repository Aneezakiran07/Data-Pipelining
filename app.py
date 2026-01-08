#Importing dependencies
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

#setting up the page configuration
st.set_page_config(
    page_title="Data Cleaning Pipeline",
    page_icon="🧹",
    layout="wide"
)
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
    #strip() function is used to remove leading and trailing whitespaces
    #lambda func is being used to apply strip func to each element , also int type dont need stripping
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    

def stripping_character_user_wants(
        df:pd.DataFrame,
        colname:str ,# tells us column name 
        ch_toremove:str #s2 tells us character to remove
):
    checking_valid_input(df)
    if colname in df.columns:
        # regex=False tells pandas "this is just a normal symbol, not a complex code"
        df[colname] = df[colname].astype(str).str.replace(ch_toremove, "", regex=False)
    
    return df

#streamlit logic
st.title("Data Cleaning Pipeline")
st.markdown(
    "Upload a csv file to clean it using the pipeline."
)

#this divider creates a horizontal line in the app
st.divider()

st.subheader("Upload your data file")
#uploading file section
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=['csv'],
    help="Upload a CSV file containing missing values etc"
)

if uploaded_file is not None:

    st.success("File uploaded successfully!")

    try:
        df = pd.read_csv(uploaded_file)

        st.info(f"DataFrame loaded with {df.shape[0]} rows and {df.shape[1]} columns")
         # Store original df in session state for comparison
        if 'original_df' not in st.session_state:
            st.session_state.original_df = df.copy()
        
        # Store current working df in session state
        if 'current_df' not in st.session_state:
            st.session_state.current_df = df.copy()

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.divider()

        st.subheader("Cleaning Data")
        st.markdown("Choose the cleaning operations to perform on your data")

          # Create columns for buttons
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("Removing trailing whitespaces", use_container_width=True):
                try:
                    #session state is used to persist data across all interactions
                    st.session_state.current_df = stripping_whitespace(st.session_state.current_df)
                    st.success("Whitespace stripped from text columns!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col2:
            if st.button("Drop Duplicate Rows", use_container_width=True):#container width is used for making button full width of col
                try:
                    before_rows=st.session_state.current_df.shape[0] #shape gives us (rows,cols)
                    st.session_state.current_df= drop_duplicate_rows(st.session_state.current_df)
                    after_rows=st.session_state.current_df.shape[0]
                    st.success(f"Dropped {before_rows-after_rows} duplicate rows!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col3:
            if st.button("Drop Duplicate Columns", use_container_width=True):
                try:
                    before_cols=st.session_state.current_df.shape[1] #shape 1 for cols
                    st.session_state.current_df= drop_duplicate_columns(st.session_state.current_df)
                    after_cols=st.session_state.current_df.shape[1]
                    st.success(f"Dropped {before_cols-after_cols} duplicate columns!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    except Exception as e:
        st.error(f"Error reading the file: {e}")
    # Here, you can add more steps of the data cleaning pipeline
else:
    st.warning("Please upload a file to proceed")


