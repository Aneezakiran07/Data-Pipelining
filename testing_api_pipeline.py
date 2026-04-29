import ast
import pathlib
import sys

print("TEST 1: API key not in URL")

# testinggg!!
for fname in ["ai_insights.py", "nl_cleaner.py"]:
    src = pathlib.Path(fname).read_text(encoding="utf-8")
    if "?key=" in src:
        print(f"FAIL: {fname} still appends ?key= to the URL")
        sys.exit(1)
    else:
        print(f"PASS: {fname} has no ?key= in URL")
    if "x-goog-api-key" in src:
        print(f"PASS: {fname} sends key in x-goog-api-key header")
    else:
        print(f"FAIL: {fname} missing x-goog-api-key header")
        sys.exit(1)

print()
print("-" * 55)
print("TEST 2: pipeline.py enum validation helpers")

sys.path.insert(0, ".")
from pipeline import _safe_enum, _safe_threshold, _safe_date_fmt
from pipeline import (
    _ALLOWED_OUTLIER_METHODS, _ALLOWED_OUTLIER_ACTIONS,
    _ALLOWED_RANGE_ACTIONS, _ALLOWED_DATE_OUTPUT_FMTS,
)

# _safe_enum: valid value passes through
result = _safe_enum("iqr", _ALLOWED_OUTLIER_METHODS, "iqr")
assert result == "iqr", f"FAIL: expected iqr got {result}"
print("PASS: valid method 'iqr' accepted")

# _safe_enum: crafted value is replaced with default
result = _safe_enum("../../etc", _ALLOWED_OUTLIER_METHODS, "iqr")
assert result == "iqr", f"FAIL: expected iqr got {result}"
print("PASS: crafted method '../../etc' replaced with default")

# _safe_enum: unknown action replaced
result = _safe_enum("execute", _ALLOWED_OUTLIER_ACTIONS, "cap")
assert result == "cap", f"FAIL: expected cap got {result}"
print("PASS: unknown action 'execute' replaced with default")

# _safe_threshold: normal value passes through
result = _safe_threshold("1.5")
assert result == 1.5, f"FAIL: expected 1.5 got {result}"
print("PASS: normal threshold 1.5 accepted")

# _safe_threshold: overflow value clamped
result = _safe_threshold("1e308")
assert result == 100.0, f"FAIL: expected 100.0 got {result}"
print("PASS: overflow threshold 1e308 clamped to 100.0")

# _safe_threshold: negative value clamped to min
result = _safe_threshold("-5")
assert result == 0.1, f"FAIL: expected 0.1 got {result}"
print("PASS: negative threshold -5 clamped to 0.1")

# _safe_threshold: non-numeric falls back to default
result = _safe_threshold("notanumber")
assert result == 1.5, f"FAIL: expected 1.5 got {result}"
print("PASS: non-numeric threshold falls back to default")

# _safe_date_fmt: allowed format passes
result = _safe_date_fmt("%Y-%m-%d", "%Y-%m-%d", allowed=_ALLOWED_DATE_OUTPUT_FMTS)
assert result == "%Y-%m-%d", f"FAIL: expected %Y-%m-%d got {result}"
print("PASS: allowed output format accepted")

# _safe_date_fmt: disallowed format replaced with default
result = _safe_date_fmt("%s", "%Y-%m-%d", allowed=_ALLOWED_DATE_OUTPUT_FMTS)
assert result == "%Y-%m-%d", f"FAIL: expected %Y-%m-%d got {result}"
print("PASS: disallowed format '%s' replaced with default")

# _safe_date_fmt: overlength input_fmt rejected
result = _safe_date_fmt("%" * 41, "", allowed=None)
assert result == "", f"FAIL: expected empty string got {result}"
print("PASS: overlength input_fmt rejected")

print()
print("All tests passed.")