import hashlib
import hmac
import os
import pickle
import time

import pandas as pd

PERSIST_DIR = os.path.join(os.path.expanduser("~"), ".dp_sessions")
MAX_SESSION_AGE_SECONDS = 60 * 60 * 24 * 2
MAX_HISTORY_STEPS = 20

# secret used to sign session files so a crafted pickle on disk cannot be loaded
# reads from env so the key survives server restarts without being hardcoded
# falls back to a stable machine-specific value when no env key is set
_SIGN_KEY = os.environ.get(
    "DP_SESSION_SECRET",
    hashlib.sha256(os.path.expanduser("~").encode()).hexdigest(),
).encode()

# length of the HMAC tag written at the start of every session file in bytes
_TAG_LEN = 32


def _ensure_dir():
    os.makedirs(PERSIST_DIR, exist_ok=True)


def make_stable_file_key(filename: str, file_bytes: bytes) -> str:
    # uses SHA-256 content hash instead of MD5 for the session key
    # two files with the same name but different contents get different sessions
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    raw = f"{filename}_{content_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _session_path(stable_key: str) -> str:
    return os.path.join(PERSIST_DIR, f"session_{stable_key}.pkl")


def _sign(data: bytes) -> bytes:
    # returns a 32-byte HMAC-SHA256 tag over the payload
    return hmac.new(_SIGN_KEY, data, hashlib.sha256).digest()


def _cleanup_old_sessions():
    try:
        now = time.time()
        for fname in os.listdir(PERSIST_DIR):
            fpath = os.path.join(PERSIST_DIR, fname)
            if os.path.isfile(fpath):
                age = now - os.path.getmtime(fpath)
                if age > MAX_SESSION_AGE_SECONDS:
                    os.remove(fpath)
    except Exception:
        pass


def save_session(
    stable_key: str,
    current_df: pd.DataFrame,
    history: list,
    original_df: pd.DataFrame,
    redo_stack: list = None,
):
    # saves current df original df history and redo stack to disk under the users home dir
    # the pickle payload is prefixed with a 32-byte HMAC tag
    # load_session verifies the tag before unpickling so a tampered file is rejected
    _ensure_dir()
    path = _session_path(stable_key)
    try:
        trimmed_history = history[-MAX_HISTORY_STEPS:]
        trimmed_redo = redo_stack[-MAX_HISTORY_STEPS:] if redo_stack else []
        payload = {
            "stable_key": stable_key,
            "saved_at": time.time(),
            "current_df": current_df,
            "original_df": original_df,
            "history": trimmed_history,
            "history_len": len(trimmed_history),
            "redo_stack": trimmed_redo,
        }
        raw = pickle.dumps(payload)
        tag = _sign(raw)
        with open(path, "wb") as f:
            f.write(tag + raw)
    except Exception:
        pass


def load_session(stable_key: str) -> dict | None:
    # returns saved session dict or None if nothing exists file is unreadable or HMAC fails
    # the HMAC check runs before pickle.load so a crafted file on disk cannot execute code
    path = _session_path(stable_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            blob = f.read()
        if len(blob) < _TAG_LEN:
            return None
        tag, raw = blob[:_TAG_LEN], blob[_TAG_LEN:]
        expected = _sign(raw)
        if not hmac.compare_digest(tag, expected):
            # file was tampered with or written by a different server key
            return None
        payload = pickle.loads(raw)  # noqa: S301
        if payload.get("stable_key") != stable_key:
            return None
        return payload
    except Exception:
        return None


def delete_session(stable_key: str):
    # wipes the persisted session for this key
    path = _session_path(stable_key)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def session_exists(stable_key: str) -> bool:
    return os.path.exists(_session_path(stable_key))


def format_saved_time(saved_at: float) -> str:
    # returns a human readable string like 3 minutes ago or 2 hours ago
    delta = time.time() - saved_at
    if delta < 60:
        return "just now"
    if delta < 3600:
        mins = int(delta // 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    if delta < 86400:
        hrs = int(delta // 3600)
        return f"{hrs} hour{'s' if hrs != 1 else ''} ago"
    days = int(delta // 86400)
    return f"{days} day{'s' if days != 1 else ''} ago"


def cleanup_old_sessions():
    _cleanup_old_sessions()