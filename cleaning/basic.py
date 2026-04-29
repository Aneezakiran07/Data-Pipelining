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


def find_and_replace_multi(df, col, pairs, use_regex=False):
    """
    Applies multiple find and replace pairs to a single column in order.
    pairs is a list of dicts with find and replace keys.
    Each find value is validated against the ReDoS guard before use.
    Pairs with an empty find string are skipped silently so the UI can
    pass the full table including blank rows without crashing.
    Returns the modified dataframe and a list of per-pair change counts
    so the caller can build an informative toast message.
    """
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pairs:
        raise ValueError("At least one find and replace pair is required")

    df_result = df.copy()
    counts = []

    mask = df_result[col].notna()
    # snapshot the column before any pair runs so each pair sees the original values
    # this prevents pair 2 from matching text that pair 1 just wrote into the column
    original_series = df_result.loc[mask, col].astype(str).copy()
    working_series = original_series.copy()

    for pair in pairs:
        find_val = pair.get("find", "")
        replace_val = pair.get("replace", "")

        # skip rows where the user left the find field empty
        if not find_val:
            counts.append(0)
            continue

        # validate pattern before it reaches pandas to prevent ReDoS
        validated_find = _safe_find_pattern(find_val, use_regex)

        # each pair runs against the original snapshot not the accumulating result
        pair_result = original_series.str.replace(validated_find, replace_val, regex=use_regex)

        # only overwrite cells that this pair actually matched
        changed_mask = pair_result != original_series
        working_series[changed_mask] = pair_result[changed_mask]

        n_changed = int(changed_mask.sum())
        counts.append(n_changed)

    df_result.loc[mask, col] = working_series
    return df_result, counts
