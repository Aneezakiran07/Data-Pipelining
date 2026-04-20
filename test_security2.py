
import os, glob, pickle, sys
sys.path.insert(0, '.')
from session_persist import load_session, save_session, PERSIST_DIR
import pandas as pd

# create a dummy session to work with
df = pd.DataFrame({"a": [1, 2, 3]})
save_session("test_key_123", df, [], df, [])

# find the file we just wrote
files = glob.glob(os.path.join(PERSIST_DIR, "session_*.pkl"))
if not files:
    print("FAIL: no session file found")
    sys.exit(1)

path = files[0]
print(f"Session file: {path}")

# tamper with one byte in the middle
with open(path, "r+b") as f:
    data = f.read()
    mid = len(data) // 2
    tampered = data[:mid] + bytes([data[mid] ^ 0xFF]) + data[mid+1:]
    f.seek(0)
    f.write(tampered)

print("File tampered.")

# now try to load it — should return None, not execute anything
result = load_session("test_key_123")
if result is None:
    print("PASS: tampered file was rejected cleanly")
else:
    print("FAIL: tampered file was loaded, HMAC check is not working")

# cleanup
os.remove(path)
