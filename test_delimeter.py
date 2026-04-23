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

# test 3: dot blocked because it is a regex metacharacter
# previously the bug was that dot silently acted as a wildcard
# now it is blocked entirely with a clear error so users know to use a plain string
try:
    _validate_delimiter(".")
    print("FAIL: dot should be blocked as a regex metacharacter")
except ValueError as e:
    print(f"PASS: dot blocked as metachar -> {e}")

# test 4: normal space delimiter still works
result = split_column(df.copy(), "full_name", " ", ["first", "last"])
assert result["first"].tolist() == ["Alice", "Bob", "Charlie"], "FAIL: space split broken"
print("PASS: normal space delimiter works")

# test 5: invalid regex syntax also blocked (contains metachar)
try:
    _validate_delimiter("[unclosed")
    print("FAIL: invalid pattern should have been rejected")
except ValueError as e:
    print(f"PASS: metachar in bad pattern blocked -> {e}")

# test 6: comma works
try:
    _validate_delimiter(",")
    print("PASS: comma delimiter accepted")
except ValueError as e:
    print(f"FAIL: comma should be accepted -> {e}")

# test 7: dash works
try:
    _validate_delimiter("-")
    print("PASS: dash delimiter accepted")
except ValueError as e:
    print(f"FAIL: dash should be accepted -> {e}")

print("\nAll delimiter tests done.")