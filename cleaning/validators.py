import numpy as np
import pandas as pd

from .basic import checking_valid_input


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
        df_clean = df_clean[is_valid].reset_index(drop=True)
    return df_clean


def validate_phone_col(df, col):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()

    # vectorized: strip all non-digit characters across the whole column at once
    digits = df_clean[col].astype(str).str.replace(r"\D", "", regex=True)
    length = digits.str.len()

    result = pd.Series(np.nan, index=df_clean.index, dtype=object)
    result = result.where(~(length == 10), "+1" + digits)
    result = result.where(~((length == 11) & digits.str.startswith("1")), "+" + digits)
    result = result.where(~((length >= 7) & ~(length == 10) & ~((length == 11) & digits.str.startswith("1"))), "+" + digits)

    df_clean[col] = result
    return df_clean


def validate_date_col(df, col, output_format="%Y-%m-%d"):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    df_clean = df.copy()

    # vectorized: pd.to_datetime with infer_datetime_format handles the vast
    # majority of formats in one C-level pass; only unparseable values become NaT
    cleaned = df_clean[col].astype(str).str.strip().replace("", np.nan).replace("nan", np.nan)
    parsed = pd.to_datetime(cleaned, infer_datetime_format=True, errors="coerce")

    # second pass: try the explicit format list only for rows that failed the
    # first pass — keeps the fast path fast and still handles edge-case formats
    failed_mask = parsed.isna() & cleaned.notna()
    if failed_mask.any():
        fmts = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
            "%m-%d-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
            "%d/%m/%y", "%m/%d/%y", "%Y.%m.%d", "%d.%m.%Y",
        ]
        for fmt in fmts:
            still_failed = parsed.isna() & cleaned.notna()
            if not still_failed.any():
                break
            parsed[still_failed] = pd.to_datetime(
                cleaned[still_failed], format=fmt, errors="coerce"
            )

    df_clean[col] = parsed.dt.strftime(output_format).where(parsed.notna(), other=None)
    return df_clean


def cap_outliers(df, col, method="iqr", action="cap", threshold=1.5):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    s = df_clean[col].dropna()
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        lower = q1 - threshold * (q3 - q1)
        upper = q3 + threshold * (q3 - q1)
    else:
        lower = s.mean() - threshold * s.std()
        upper = s.mean() + threshold * s.std()
    if action == "cap":
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    else:
        df_clean = df_clean[df_clean[col].isna() | df_clean[col].between(lower, upper)].reset_index(drop=True)
    return df_clean


def validate_range(df, col, min_val, max_val, action="flag"):
    checking_valid_input(df)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric")
    df_clean = df.copy()
    in_range = df_clean[col].between(min_val, max_val, inclusive="both") | df_clean[col].isna()
    if action == "flag":
        df_clean[f"{col}_in_range"] = in_range
    else:
        df_clean = df_clean[in_range].reset_index(drop=True)
    return df_clean

