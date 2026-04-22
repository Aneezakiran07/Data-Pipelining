import pandas as pd
from cleaning.transforms import _sanitize_col_name, rename_columns

df = pd.DataFrame({"name": [1, 2], "age": [3, 4]})

# test 1: null byte blocked
try:
    _sanitize_col_name("col\x00name")
    print("FAIL: null byte should have been rejected")
except ValueError as e:
    print(f"PASS: null byte blocked -> {e}")

# test 2: newline blocked
try:
    _sanitize_col_name("col\nname")
    print("FAIL: newline should have been rejected")
except ValueError as e:
    print(f"PASS: newline blocked -> {e}")

# test 3: pipe blocked
try:
    _sanitize_col_name("col|name")
    print("FAIL: pipe should have been rejected")
except ValueError as e:
    print(f"PASS: pipe blocked -> {e}")

# test 4: overlength blocked
try:
    _sanitize_col_name("a" * 201)
    print("FAIL: overlength should have been rejected")
except ValueError as e:
    print(f"PASS: overlength blocked -> {e}")

# test 5: normal rename still works
result = rename_columns(df, {"name": "full_name", "age": "age"})
assert list(result.columns) == ["full_name", "age"], "FAIL: normal rename broken"
print("PASS: normal rename works")

print("\nAll rename tests done.")

