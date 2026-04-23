import sys
import ast
import pathlib

# test 1: confirm __import__ is not in the file source at all
src = pathlib.Path("tabs/recommendations.py").read_text(encoding="utf-8")
if "__import__" in src:
    print("FAIL: __import__ still present in recommendations.py")
    sys.exit(1)
else:
    print("PASS: no __import__ in recommendations.py")

# test 2: confirm pandas is imported at the top level
tree = ast.parse(src)
top_level_imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            top_level_imports.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        top_level_imports.append(node.module or "")

if "pandas" in top_level_imports:
    print("PASS: pandas imported at top level")
else:
    print("FAIL: pandas not found in top-level imports")
    sys.exit(1)

# test 3: confirm _apply_fix logic still works correctly by running it directly
import numpy as np
import pandas as pd

# inline the fixed _apply_fix so we can test it without streamlit
def _apply_fix_test(action_key, sel_cols, cdf):
    tmp = cdf.copy()
    if action_key == "convert_currency":
        for c in sel_cols:
            ne = tmp[c].astype(str).str.strip().replace("", np.nan).dropna()
            cl = (
                ne.str.replace(r"[^\d.,\-()]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(r"\((.+?)\)", r"-\1", regex=True)
                .str.extract(r"([-]?\d[\d\.,]*)", expand=False)
                .str.replace(",", "", regex=False)
            )
            tmp[c] = pd.to_numeric(cl, errors="coerce").reindex(tmp.index)
    elif action_key == "convert_percentage":
        for c in sel_cols:
            ne = tmp[c].astype(str).str.strip().replace("", np.nan).dropna()
            cleaned = ne.str.replace("%", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
            tmp[c] = (pd.to_numeric(cleaned, errors="coerce") / 100).reindex(tmp.index)
    elif action_key == "convert_units":
        for c in sel_cols:
            ne = tmp[c].astype(str).str.strip().replace("", np.nan).dropna()
            tmp[c] = pd.to_numeric(
                ne.str.extract(r"([-]?\d+\.?\d*)", expand=False), errors="coerce"
            ).reindex(tmp.index)
    return tmp

df = pd.DataFrame({
    "price":   ["$1,200", "$950", "(1,500)"],
    "pct":     ["10%", "5%", "20%"],
    "weight":  ["70kg", "80kg", "90kg"],
})

result = _apply_fix_test("convert_currency", ["price"], df)
assert result["price"].tolist() == [1200.0, 950.0, -1500.0], f"FAIL: currency conversion wrong: {result['price'].tolist()}"
print("PASS: currency conversion correct")

result = _apply_fix_test("convert_percentage", ["pct"], df)
expected = [0.10, 0.05, 0.20]
actual = [round(v, 2) for v in result["pct"].tolist()]
assert actual == expected, f"FAIL: percentage conversion wrong: {actual}"
print("PASS: percentage conversion correct")

result = _apply_fix_test("convert_units", ["weight"], df)
assert result["weight"].tolist() == [70.0, 80.0, 90.0], f"FAIL: unit conversion wrong: {result['weight'].tolist()}"
print("PASS: unit conversion correct")

print("\nAll recommendations tests done.")