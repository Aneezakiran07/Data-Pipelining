import json
import streamlit as st


def snapshot():
    return st.session_state.current_df.copy()


def _autosave():
    # saves current session to disk after every mutating operation
    # only runs when a persist key exists which is set during init
    persist_key = st.session_state.get("_persist_key")
    if not persist_key:
        return
    try:
        from session_persist import save_session
        save_session(
            stable_key=persist_key,
            current_df=st.session_state.current_df,
            history=st.session_state.get("history", []),
            original_df=st.session_state.get("original_df", st.session_state.current_df),
            redo_stack=st.session_state.get("redo_stack", [])
        )
    except Exception:
        pass


def commit_history(label, snap):
    if "history" not in st.session_state:
        st.session_state.history = []
    if "redo_stack" not in st.session_state:
        st.session_state.redo_stack = []

    if len(st.session_state.history) >= 20:
        st.session_state.history.pop(0)

    st.session_state.history.append({"label": label, "df": snap})
    st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1

    # clear redo stack because a new action creates a new timeline
    st.session_state.redo_stack.clear()
    _autosave()


def undo_last():
    if st.session_state.get("history"):
        if "redo_stack" not in st.session_state:
            st.session_state.redo_stack = []

        last = st.session_state.history.pop()

        # save current state before overwriting so redo can restore it
        st.session_state.redo_stack.append({
            "label": last["label"],
            "df": st.session_state.current_df.copy()
        })

        st.session_state.current_df = last["df"]
        st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1
        _autosave()
        return last["label"]
    return None


def redo_action():
    # applies last undone action from redo stack and moves it back to history
    if st.session_state.get("redo_stack"):
        next_action = st.session_state.redo_stack.pop()

        # push current state back to history before applying the redone state
        st.session_state.history.append({
            "label": next_action["label"],
            "df": st.session_state.current_df.copy()
        })

        st.session_state.current_df = next_action["df"]
        st.session_state["history_len"] = st.session_state.get("history_len", 0) + 1
        _autosave()
        return next_action["label"]
    return None


def export_pipeline_json(history):
    steps = [{"step": i + 1, "label": entry["label"]} for i, entry in enumerate(history)]
    return json.dumps({"version": 1, "steps": steps}, indent=2)


# moved to module level so both import_pipeline_json and build_pipeline_script can share it
def _parse_label_meta(label, prefix, keys):
    # pulls key=value pairs embedded in a history label string
    # labels are stored as "Step Name|key1=val1|key2=val2"
    # returns a dict with the requested keys, missing keys return empty string
    meta = {k: "" for k in keys}
    if "|" not in label:
        return meta
    parts = label.split("|")
    for part in parts[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            if k in meta:
                meta[k] = v
    return meta


def import_pipeline_json(json_bytes, df, settings):
    from cleaning.basic import (
        clean_string_edges,
        drop_duplicate_columns,
        drop_duplicate_rows,
        stripping_whitespace,
    )
    from cleaning.advanced import missing_value_handler, smart_column_cleaner
    from cleaning.validators import validate_email_col, validate_phone_col, validate_date_col
    from cleaning.validators import cap_outliers, validate_range

    missing_threshold = settings.get("missing_threshold", 0.3)
    numeric_strategy = settings.get("numeric_strategy", "auto")
    conversion_threshold = settings.get("conversion_threshold", 0.6)

    try:
        payload = json.loads(json_bytes)
    except Exception:
        raise ValueError("File is not valid JSON.")

    if not isinstance(payload, dict) or "steps" not in payload:
        raise ValueError("JSON does not look like a saved pipeline file.")

    tmp = df.copy()
    applied = []
    skipped = []

    for entry in payload["steps"]:
        label = entry.get("label", "")

        if label == "Strip Whitespace" or label.startswith("Fix: strip_whitespace"):
            tmp = stripping_whitespace(tmp)
            applied.append(label)

        elif label == "Drop Duplicate Rows" or label.startswith("Fix: drop_duplicates"):
            tmp = drop_duplicate_rows(tmp)
            applied.append(label)

        elif label == "Drop Duplicate Columns" or label.startswith("Fix: drop_dup_cols"):
            tmp = drop_duplicate_columns(tmp)
            applied.append(label)

        elif label == "Clean String Edges" or label.startswith("Fix: clean_edges"):
            tmp = clean_string_edges(tmp, threshold=0.7)
            applied.append(label)

        elif label == "Smart Column Cleaner" or label.startswith("Fix: convert"):
            tmp = smart_column_cleaner(tmp, conversion_threshold=conversion_threshold)
            applied.append(label)

        elif label == "Handle Missing Values" or label.startswith("Fix: handle_missing"):
            tmp = missing_value_handler(tmp, threshold=missing_threshold, numeric_strategy=numeric_strategy)
            applied.append(label)

        elif label == "Auto-Fix All":
            tmp = stripping_whitespace(tmp)
            tmp = drop_duplicate_rows(tmp)
            tmp = drop_duplicate_columns(tmp)
            tmp = clean_string_edges(tmp, threshold=0.7)
            tmp = smart_column_cleaner(tmp, conversion_threshold=conversion_threshold)
            tmp = missing_value_handler(tmp, threshold=missing_threshold, numeric_strategy=numeric_strategy)
            applied.append(label)

        # validation steps - each reads metadata from the label and checks the column exists first

        elif label.startswith("Validate Email"):
            meta = _parse_label_meta(label, "Validate Email", ["col", "action", "pattern"])
            col = meta["col"] or "email"
            action = meta["action"] or "flag"
            pattern = meta["pattern"] or r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
            # skip gracefully if the column no longer exists in this dataset
            if col not in tmp.columns:
                skipped.append(f"{label} (column '{col}' not found)")
            else:
                tmp = validate_email_col(tmp, col=col, action=action, custom_pattern=pattern)
                applied.append(label)

        elif label.startswith("Standardize Phone"):
            meta = _parse_label_meta(label, "Standardize Phone", ["col", "cc"])
            col = meta["col"] or "phone"
            cc = meta["cc"] or "1"
            if col not in tmp.columns:
                skipped.append(f"{label} (column '{col}' not found)")
            else:
                tmp = validate_phone_col(tmp, col=col, default_country_code=cc)
                applied.append(label)

        elif label.startswith("Standardize Dates"):
            meta = _parse_label_meta(label, "Standardize Dates", ["col", "output_fmt", "input_fmt"])
            col = meta["col"] or "date"
            output_fmt = meta["output_fmt"] or "%Y-%m-%d"
            input_fmt = meta["input_fmt"] or ""
            if col not in tmp.columns:
                skipped.append(f"{label} (column '{col}' not found)")
            else:
                tmp = validate_date_col(tmp, col=col, output_format=output_fmt, custom_input_format=input_fmt)
                applied.append(label)

        elif label.startswith("Cap Outliers"):
            meta = _parse_label_meta(label, "Cap Outliers", ["col", "method", "action", "threshold"])
            col = meta["col"] or "value"
            method = meta["method"] or "iqr"
            action = meta["action"] or "cap"
            # cast threshold safely so a bad saved value never crashes replay
            try:
                threshold = float(meta["threshold"]) if meta["threshold"] else 1.5
            except ValueError:
                threshold = 1.5
            if col not in tmp.columns:
                skipped.append(f"{label} (column '{col}' not found)")
            else:
                tmp = cap_outliers(tmp, col=col, method=method, action=action, threshold=threshold)
                applied.append(label)

        elif label.startswith("Validate Range"):
            meta = _parse_label_meta(label, "Validate Range", ["col", "min", "max", "action"])
            col = meta["col"] or "value"
            action = meta["action"] or "flag"
            # cast min and max safely so a bad saved value never crashes replay
            try:
                minval = float(meta["min"]) if meta["min"] else 0.0
            except ValueError:
                minval = 0.0
            try:
                maxval = float(meta["max"]) if meta["max"] else 100.0
            except ValueError:
                maxval = 100.0
            if col not in tmp.columns:
                skipped.append(f"{label} (column '{col}' not found)")
            else:
                tmp = validate_range(tmp, col=col, min_val=minval, max_val=maxval, action=action)
                applied.append(label)

        else:
            skipped.append(label)

    return tmp, applied, skipped


def build_pipeline_script(history):
    lines = [
        "import pandas as pd",
        "import numpy as np",
        "import re",
        "from sklearn.impute import KNNImputer",
        "",
        "df = pd.read_csv('your_file.csv')",
        "",
    ]

    for step in history:
        label = step["label"]
        lines.append(f"# {label}")

        # basic cleaning steps
        if label == "Strip Whitespace" or label.startswith("Fix: strip_whitespace"):
            lines += [
                "df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)",
            ]

        elif label == "Drop Duplicate Rows" or label.startswith("Fix: drop_duplicates"):
            lines += [
                "df = df.drop_duplicates().reset_index(drop=True)",
            ]

        elif label == "Drop Duplicate Columns" or label.startswith("Fix: drop_dup_cols"):
            lines += [
                "df = df.loc[:, ~df.columns.duplicated()]",
                "df = df.T.drop_duplicates().T",
            ]

        elif label == "Clean String Edges" or label.startswith("Fix: clean_edges"):
            lines += [
                "for col in df.select_dtypes(include='object').columns:",
                "    df[col] = df[col].astype(str).str.replace(r'^\\W+', '', regex=True).str.replace(r'\\W+$', '', regex=True)",
            ]

        elif label == "Smart Column Cleaner" or label.startswith("Fix: convert"):
            lines += [
                "for col in df.select_dtypes(include='object').columns:",
                "    _cleaned = df[col].str.replace(r'[^\\d.\\-]', '', regex=True)",
                "    _converted = pd.to_numeric(_cleaned, errors='coerce')",
                "    if _converted.notna().mean() > 0.6:",
                "        df[col] = _converted",
            ]

        elif label == "Handle Missing Values" or label.startswith("Fix: handle_missing"):
            lines += [
                "df = df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan)",
                "_num_cols = df.select_dtypes(include=np.number).columns.tolist()",
                "if _num_cols and df[_num_cols].isna().any().any():",
                "    df[_num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[_num_cols])",
                "for _col in df.select_dtypes(exclude=np.number).columns:",
                "    _mode = df[_col].mode()",
                "    df[_col] = df[_col].fillna(_mode[0] if not _mode.empty else 'Missing')",
            ]

        elif label == "Auto-Fix All":
            lines += [
                "df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)",
                "df = df.drop_duplicates().reset_index(drop=True)",
                "df = df.loc[:, ~df.columns.duplicated()]",
                "df = df.replace(['?', 'NA', 'unknown', 'n/a', 'NaN', 'null', ''], np.nan)",
                "_num_cols = df.select_dtypes(include=np.number).columns.tolist()",
                "if _num_cols and df[_num_cols].isna().any().any():",
                "    df[_num_cols] = KNNImputer(n_neighbors=5).fit_transform(df[_num_cols])",
            ]

        # transform steps
        elif label.startswith("Find and Replace in"):
            col = label.replace("Find and Replace in ", "").strip()
            lines += [
                f"# TODO set FIND and REPLACE values for column '{col}'",
                f"df['{col}'] = df['{col}'].astype(str).str.replace('FIND', 'REPLACE', regex=False)",
            ]

        elif label.startswith("Type Override:"):
            parts = label.replace("Type Override: ", "").split(" -> ")
            if len(parts) == 2:
                col, dtype = parts[0].strip(), parts[1].strip()
                if "int" in dtype:
                    lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').astype('Int64')")
                elif "float" in dtype:
                    lines.append(f"df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')")
                elif "datetime" in dtype:
                    lines.append(f"df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
                elif "bool" in dtype:
                    lines.append(f"df['{col}'] = df['{col}'].astype(str).str.lower().map({{'true':True,'1':True,'yes':True,'false':False,'0':False,'no':False}})")
                elif "category" in dtype:
                    lines.append(f"df['{col}'] = df['{col}'].astype('category')")
                else:
                    lines.append(f"df['{col}'] = df['{col}'].astype(str)")

        elif label.startswith("Split column"):
            lines += [
                "# TODO set col name and delimiter below",
                "_parts = df['col'].str.split('delimiter', expand=True)",
                "df['part1'] = _parts[0]",
                "df['part2'] = _parts[1]",
            ]

        elif label.startswith("Merge columns into"):
            new_col = label.replace("Merge columns into ", "").strip()
            lines += [
                "# TODO set col1 col2 and separator below",
                f"df['{new_col}'] = df['col1'].astype(str) + ' ' + df['col2'].astype(str)",
            ]

        elif label.startswith("Rename"):
            lines += [
                "# TODO set old_name and new_name below",
                "df = df.rename(columns={'old_name': 'new_name'})",
            ]

        # validation steps
        # each label carries embedded metadata as key=value pairs after a pipe
        # e.g. "Validate Email|col=email|action=flag|pattern=^[...]$"
        # _parse_label_meta extracts those values so the script uses exact settings

        elif label.startswith("Validate Email"):
            meta = _parse_label_meta(label, "Validate Email", ["col", "action", "pattern"])
            col = meta["col"] or "email"
            action = meta["action"] or "flag"
            pattern = meta["pattern"] or r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
            lines += [
                f"# col: {col}  action: {action}",
                f"# pattern: {pattern}",
                f"_email_pattern = r'{pattern}'",
                f"_mask = df['{col}'].astype(str).str.strip().str.match(_email_pattern)",
                f"if '{action}' == 'flag':",
                f"    df['{col}_valid_email'] = _mask",
                "else:",
                f"    df = df[_mask].reset_index(drop=True)",
            ]

        elif label.startswith("Standardize Phone"):
            meta = _parse_label_meta(label, "Standardize Phone", ["col", "cc"])
            col = meta["col"] or "phone"
            cc = meta["cc"] or "1"
            lines += [
                f"# col: {col}  default country code: {cc}",
                f"_cc = '{cc}'",
                f"_cc_len = len(_cc)",
                f"_digits = df['{col}'].astype(str).str.replace(r'\\D', '', regex=True)",
                "_length = _digits.str.len()",
                "import numpy as _np",
                f"_result = _np.where(",
                f"    _length == 10, '+' + _cc + _digits,",
                f"    _np.where(",
                f"        (_length == (10 + _cc_len)) & _digits.str.startswith(_cc), '+' + _digits,",
                f"        _np.where(_length >= 7, '+' + _digits, _np.nan)",
                f"    )",
                ")",
                f"df['{col}'] = _result",
            ]

        elif label.startswith("Standardize Dates"):
            meta = _parse_label_meta(label, "Standardize Dates", ["col", "output_fmt", "input_fmt"])
            col = meta["col"] or "date"
            output_fmt = meta["output_fmt"] or "%Y-%m-%d"
            input_fmt = meta["input_fmt"] or ""
            lines += [
                f"# col: {col}  output format: {output_fmt}",
                f"_cleaned = df['{col}'].astype(str).str.strip().replace('', _np.nan).replace('nan', _np.nan)",
                "import pandas as _pd",
            ]
            if input_fmt:
                lines += [
                    f"_parsed = _pd.to_datetime(_cleaned, format='{input_fmt}', errors='coerce')",
                    "# fill any rows the custom format missed using auto detection",
                    "_still_unparsed = _parsed.isna() & _cleaned.notna()",
                    "if _still_unparsed.any():",
                    "    _parsed[_still_unparsed] = _pd.to_datetime(_cleaned[_still_unparsed], infer_datetime_format=True, errors='coerce')",
                ]
            else:
                lines += [
                    "_parsed = _pd.to_datetime(_cleaned, infer_datetime_format=True, errors='coerce')",
                ]
            lines += [
                f"df['{col}'] = _parsed.dt.strftime('{output_fmt}').where(_parsed.notna(), other=None)",
            ]

        elif label.startswith("Cap Outliers"):
            meta = _parse_label_meta(label, "Cap Outliers", ["col", "method", "action", "threshold"])
            col = meta["col"] or "value"
            method = meta["method"] or "iqr"
            action = meta["action"] or "cap"
            threshold = meta["threshold"] or "1.5"
            lines += [
                f"# col: {col}  method: {method}  action: {action}  threshold: {threshold}",
                f"_s = df['{col}'].dropna()",
            ]
            if method == "iqr":
                lines += [
                    f"_q1 = _s.quantile(0.25)",
                    f"_q3 = _s.quantile(0.75)",
                    f"_lower = _q1 - {threshold} * (_q3 - _q1)",
                    f"_upper = _q3 + {threshold} * (_q3 - _q1)",
                ]
            else:
                lines += [
                    f"_lower = _s.mean() - {threshold} * _s.std()",
                    f"_upper = _s.mean() + {threshold} * _s.std()",
                ]
            if action == "cap":
                lines += [
                    f"df['{col}'] = df['{col}'].clip(lower=_lower, upper=_upper)",
                ]
            else:
                lines += [
                    f"df = df[df['{col}'].isna() | df['{col}'].between(_lower, _upper)].reset_index(drop=True)",
                ]

        elif label.startswith("Validate Range"):
            meta = _parse_label_meta(label, "Validate Range", ["col", "min", "max", "action"])
            col = meta["col"] or "value"
            minval = meta["min"] or "0.0"
            maxval = meta["max"] or "100.0"
            action = meta["action"] or "flag"
            lines += [
                f"# col: {col}  range: [{minval}, {maxval}]  action: {action}",
                f"_in_range = df['{col}'].between({minval}, {maxval}, inclusive='both') | df['{col}'].isna()",
            ]
            if action == "flag":
                lines += [
                    f"df['{col}_in_range'] = _in_range",
                ]
            else:
                lines += [
                    f"df = df[_in_range].reset_index(drop=True)",
                ]

        else:
            lines.append("# manual step - no code generated")

        lines.append("")

    lines.append("print('Pipeline complete. Shape:', df.shape)")
    return "\n".join(lines)