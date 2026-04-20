import re

import numpy as np
import pandas as pd

# maxi char length for a user supplied regex pattern in find and replace
# patterns longer than this are rejected before compilation to prevent ReDoS
_MAX_FIND_PATTERN_LEN = 300


def _safe_find_pattern(pattern: str, use_regex: bool) -> str:
    """
    Validates a user supplied find pattern when regex mode is on.
    Rejects patterns that are too long or fail re.compile to prevent ReDoS.
    Returns the original pattern if it passes, raises ValueError if it does not.
    Literal patterns are returned unchanged since pandas escapes them internally.
    """
    if not use_regex:
        return pattern
    if len(pattern) > _MAX_FIND_PATTERN_LEN:
        raise ValueError(
            f"Regex pattern is too long ({len(pattern)} chars, max {_MAX_FIND_PATTERN_LEN}). "
            "Shorten the pattern or turn off Use Regex."
        )
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc
    return pattern


def checking_valid_input(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")


def drop_duplicate_rows(df):
    checking_valid_input(df)
    # reset_index so downstream positional operations see a clean 0-based index
    return df.drop_duplicates().reset_index(drop=True)


def drop_duplicate_columns(df):
    checking_valid_input(df)
    df = df.loc[:, ~df.columns.duplicated()]
    return df.T.drop_duplicates().T


def stripping_whitespace(df):
    checking_valid_input(df)
    return df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)


def clean_string_edges(df, threshold=0.7, inplace=False):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
        return None if inplace else df.copy()
    df_clean = df if inplace else df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        col_series = df_clean[col].astype(str)
        leading = col_series.str.extract(r"^([^\w\s])")[0].dropna()
        trailing = col_series.str.extract(r"([^\w\s])$")[0].dropna()
        keep_leading = (leading.value_counts(normalize=True).iloc[0] > threshold if len(leading) > 0 else False)
        keep_trailing = (trailing.value_counts(normalize=True).iloc[0] > threshold if len(trailing) > 0 else False)
        if not keep_leading:
            df_clean[col] = col_series.str.replace(r"^\W+", "", regex=True)
        if not keep_trailing:
            df_clean[col] = df_clean[col].astype(str).str.replace(r"\W+$", "", regex=True)
    return None if inplace else df_clean


def find_and_replace(df, col, find, replace, use_regex=False):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    # validate the find pattern before passing it to pandas to prevent ReDoS
    validated_find = _safe_find_pattern(find, use_regex)
    df_clean = df.copy()
    mask = df_clean[col].notna()
    df_clean.loc[mask, col] = (
        df_clean.loc[mask, col]
        .astype(str)
        .str.replace(validated_find, replace, regex=use_regex)
    )
    return df_clean