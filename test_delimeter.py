import pandas as pd
from cleaning.transforms import _validate_delimiter, split_column

df = pd.DataFrame({"full_name": ["Alice Smith", "Bob Jones", "Charlie Brown"]})

# test 1: ReDoS pattern blocked
try:
    _validate_delimiter("(a+)+$")
    print("FAIL: ReDoS pattern should have been rejected")
except ValueError as e:
    print(f"PASS: ReDoS blocked -> {e}")

# test 2: overlength delimiter blocked
try:
    _validate_delimiter("," * 21)
    print("FAIL: overlength delimiter should have been rejected")
except ValueError as e:
    print(f"PASS: overlength blocked -> {e}")

# test 3: literal dot treated as literal, not wildcard
result = split_column(df.copy(), "full_name", ".", ["part1", "part2"])
# dot as wildcard would split on every char, giving many columns
# dot as literal finds no match, so part2 should be NaN for all rows
assert result["part2"].isna().all(), "FAIL: dot treated as wildcard, not literal"
print("PASS: literal dot not treated as wildcard")

# test 4: normal space delimiter still works
result = split_column(df.copy(), "full_name", " ", ["first", "last"])
assert result["first"].tolist() == ["Alice", "Bob", "Charlie"], "FAIL: space split broken"
print("PASS: normal space delimiter works")

# test 5: invalid regex syntax blocked
try:
    _validate_delimiter("[unclosed")
    print("FAIL: invalid regex should have been rejected")
except ValueError as e:
    print(f"PASS: invalid regex blocked -> {e}")

print("\nAll delimiter tests done.")

