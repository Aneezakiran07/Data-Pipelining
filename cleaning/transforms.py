import re

import numpy as np
import pandas as pd

# maximum character length allowed for a column name
# names longer than this are almost always a paste accident or injection attempt
_MAX_COL_NAME_LEN = 200

# maximum character length for a split delimiter
# anything longer than this is not a real delimiter
_MAX_DELIMITER_LEN = 20

# characters that are never safe in a column name
# null byte corrupts CSV parsers silently
# newline and carriage return break CSV headers and pipeline JSON label parsing
# pipe is the delimiter used in history labels so it breaks _parse_label_meta
_FORBIDDEN_COL_CHARS = re.compile(r"[\x00\n\r|]")


def _sanitize_col_name(name: str) -> str:
    """
    Cleans and validates a user supplied column name before it is applied.
    Strips leading and trailing whitespace first, then checks:
      - name must not be empty after stripping
      - name must not exceed _MAX_COL_NAME_LEN characters
      - name must not contain null bytes, newlines, carriage returns, or pipe characters
    Raises ValueError with a clear message on any violation so the UI can
    show it to the user before anything is written to the dataframe.
    """
    cleaned = name.strip()

    if not cleaned:
        raise ValueError("Column name cannot be empty or whitespace only")

    if len(cleaned) > _MAX_COL_NAME_LEN:
        raise ValueError(
            f"Column name is too long ({len(cleaned)} chars, max {_MAX_COL_NAME_LEN}): "
            f'"{cleaned[:40]}..."'
        )

    match = _FORBIDDEN_COL_CHARS.search(cleaned)
    if match:
        char = match.group()
        display = {"\x00": "\\0 (null byte)", "\n": "\\n (newline)",
                   "\r": "\\r (carriage return)", "|": "| (pipe)"}
        raise ValueError(
            f'Column name contains a forbidden character {display.get(char, repr(char))}: '
            f'"{cleaned[:40]}"'
        )

    return cleaned


def _validate_delimiter(delimiter: str) -> str:
    """
    Validates a user supplied split delimiter before it is passed to str.split.
    pandas passes the delimiter to Python's re engine when expand=True is used.
    re.compile cannot detect catastrophic backtracking patterns like (a+)+$ at
    compile time so they only blow up at match time, meaning the server hangs
    silently mid-operation with no error shown to the user.
    The safe approach is to only allow plain literal delimiters and reject anything
    that contains regex metacharacters. Split delimiters are always literals in
    practice: commas, spaces, pipes, dashes, tabs. Nobody needs a regex here.
    Every accepted delimiter is passed through re.escape before str.split so the
    re engine always treats it as a fixed string, never as a pattern.
    Raises ValueError with a clear message on any violation.
    Returns the re-escaped delimiter safe to pass to str.split.
    """
    if not delimiter:
        raise ValueError("Delimiter cannot be empty")

    if len(delimiter) > _MAX_DELIMITER_LEN:
        raise ValueError(
            f"Delimiter is too long ({len(delimiter)} chars, max {_MAX_DELIMITER_LEN})"
        )

    # reject any delimiter that contains regex metacharacters
    # re.compile cannot detect ReDoS patterns at compile time so the only safe
    # option is to block regex delimiters entirely and require plain strings
    _REGEX_META = set(r"\.^$*+?{}[]|()")
    bad = [c for c in delimiter if c in _REGEX_META]
    if bad:
        unique_bad = ", ".join(f'"{c}"' for c in sorted(set(bad)))
        raise ValueError(
            f"Delimiter contains regex special characters ({unique_bad}). "
            "Use a plain string like a comma, space, dash, or tab."
        )

    # escape the delimiter so the re engine always treats it as a fixed string
    return re.escape(delimiter)


def split_column(df, col, delimiter, new_col_names, keep_original=False):
    # splits a text column into multiple new columns on a fixed delimiter
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found")
    if not delimiter:
        raise ValueError("Delimiter cannot be empty")
    if not new_col_names:
        raise ValueError("At least one output column name is required")

    # validate and escape the delimiter before it reaches the re engine
    # this prevents ReDoS from patterns like (a+)+$ typed into the delimiter box
    safe_delimiter = _validate_delimiter(delimiter)

    # sanitize every output column name before splitting so bad names
    # are caught before the operation runs, not after
    sanitized_names = []
    for raw in new_col_names:
        sanitized_names.append(_sanitize_col_name(str(raw)))

    tmp = df.copy()
    n_parts = len(sanitized_names)
    split_result = tmp[col].astype(str).str.split(safe_delimiter, expand=True)

    for i in range(n_parts):
        col_name = sanitized_names[i] or f"{col}_part{i + 1}"
        tmp[col_name] = split_result[i] if i < split_result.shape[1] else np.nan

    if not keep_original:
        tmp = tmp.drop(columns=[col])

    return tmp


def merge_columns(df, cols, new_col_name, separator, keep_originals=False):
    # merges two or more columns into one by joining their values with a separator
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if len(cols) < 2:
        raise ValueError("At least two columns are required for merging")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")
    if not new_col_name or not new_col_name.strip():
        raise ValueError("New column name cannot be empty")

    # sanitize the merged column name before creating it
    sanitized_new = _sanitize_col_name(new_col_name)

    tmp = df.copy()
    merged = tmp[cols[0]].astype(str).str.strip()
    for c in cols[1:]:
        merged = merged + separator + tmp[c].astype(str).str.strip()
    tmp[sanitized_new] = merged

    if not keep_originals:
        tmp = tmp.drop(columns=[c for c in cols if c != sanitized_new])

    return tmp


def rename_columns(df, rename_map):
    # renames columns from a dict of old name to new name
    # skips blank entries and raises if a new name collides with an existing column
    # sanitizes every new name to reject null bytes newlines pipes and overlength strings
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    clean_map = {}
    for old, new in rename_map.items():
        if old not in df.columns:
            continue
        stripped = new.strip() if isinstance(new, str) else ""
        if not stripped or stripped == old:
            continue
        # raises ValueError with a clear message if the name is unsafe
        sanitized = _sanitize_col_name(stripped)
        clean_map[old] = sanitized

    if not clean_map:
        return df.copy()

    duplicates = [n for n in clean_map.values() if n in df.columns and n not in clean_map]
    if duplicates:
        raise ValueError(f"New names conflict with existing columns: {', '.join(duplicates)}")

    return df.rename(columns=clean_map)


def apply_type_suggestions(df, selected_suggestions):
    # applies a list of type suggestion dicts to the dataframe
    # selected_suggestions is a filtered list where the user has ticked which ones to apply
    # each suggestion dict must have column and suggested_action keys
    import re as _re
    from .advanced import _convert_duration_val

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    tmp = df.copy()
    applied = []

    for suggestion in selected_suggestions:
        col = suggestion["column"]
        action = suggestion["suggested_action"]

        if col not in tmp.columns:
            continue

        s = tmp[col].astype(str).str.strip()
        non_empty = s.replace("", np.nan).dropna()

        try:
            if action == "convert_currency":
                cleaned = (
                    non_empty
                    .str.replace(r"[^\d.,\-()]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                    .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                    .str.replace(",", "", regex=False)
                )
                tmp[col] = pd.to_numeric(cleaned, errors="coerce").reindex(tmp.index)
                applied.append(col)

            elif action == "convert_percentage":
                cleaned = non_empty.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
                tmp[col] = (pd.to_numeric(cleaned, errors="coerce") / 100).reindex(tmp.index)
                applied.append(col)

            elif action == "convert_units":
                cleaned = non_empty.str.extract(r"([-]?\d+\.?\d*)", expand=False)
                tmp[col] = pd.to_numeric(cleaned, errors="coerce").reindex(tmp.index)
                applied.append(col)

            elif action == "convert_duration":
                tmp[col] = non_empty.apply(_convert_duration_val).reindex(tmp.index)
                applied.append(col)

            elif action == "convert_datetime":
                tmp[col] = pd.to_datetime(tmp[col], errors="coerce")
                applied.append(col)

            elif action == "convert_numeric":
                cleaned = non_empty.str.replace(r"[^\d.\-]", "", regex=True)
                tmp[col] = pd.to_numeric(cleaned, errors="coerce").reindex(tmp.index)
                applied.append(col)

            elif action == "convert_to_boolean":
                tmp[col] = tmp[col].astype(str).str.lower().map(
                    {"true": True, "1": True, "yes": True, "y": True,
                     "false": False, "0": False, "no": False, "n": False}
                )
                applied.append(col)

            elif action == "convert_category":
                tmp[col] = tmp[col].astype("category")
                applied.append(col)

            elif action == "validate_email":
                pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
                tmp[f"{col}_valid_email"] = tmp[col].astype(str).str.strip().str.match(pattern)
                applied.append(col)

        except Exception:
            continue

    return tmp, applied