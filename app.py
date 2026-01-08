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

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)


    except Exception as e:
        st.error(f"Error reading the file: {e}")
    # Here, you can add more steps of the data cleaning pipeline
else:
    st.warning("Please upload a file to proceed")



def checking_valid_input(
        pd:pd.DataFrame
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
    

def fix_column_types(
        df:pd.DataFrame
):
    checking_valid_input(df)


