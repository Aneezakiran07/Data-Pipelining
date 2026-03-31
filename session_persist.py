import hashlib
import os
import pickle
import time

import pandas as pd

PERSIST_DIR = os.path.join(os.path.expanduser("~"), ".dp_sessions")
MAX_SESSION_AGE_SECONDS = 60 * 60 * 24 * 2
MAX_HISTORY_STEPS = 20


def _ensure_dir():
    os.makedirs(PERSIST_DIR, exist_ok=True)


def make_stable_file_key(filename: str, file_bytes: bytes) -> str:
    """
    Builds a session key from filename and file size.
    file_id from Streamlit changes on every reload so it cannot be used.
    filename plus size is stable as long as the user uploads the same file.
    """
    size = len(file_bytes)
    raw = f"{filename}_{size}"
    return hashlib.md5(raw.encode()).hexdigest()


def _session_path(stable_key: str) -> str:
    return os.path.join(PERSIST_DIR, f"session_{stable_key}.pkl")


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


def save_session(stable_key: str, current_df: pd.DataFrame, history: list, original_df: pd.DataFrame):
    """
    Saves current df, original df, and history to disk under the users home dir.
    Home dir works on both Windows and Linux.
    Pickle is used because it handles pandas DataFrames natively and is fast.
    Only the last MAX_HISTORY_STEPS entries are kept to cap file size.
    """
    _ensure_dir()
    path = _session_path(stable_key)
    try:
        trimmed_history = history[-MAX_HISTORY_STEPS:]
        payload = {
            "stable_key": stable_key,
            "saved_at": time.time(),
            "current_df": current_df,
            "original_df": original_df,
            "history": trimmed_history,
            "history_len": len(trimmed_history),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def load_session(stable_key: str) -> dict | None:
    """
    Returns saved session dict or None if nothing exists or file is unreadable.
    Dict keys: current_df, original_df, history, history_len, saved_at
    """
    path = _session_path(stable_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if payload.get("stable_key") != stable_key:
            return None
        return payload
    except Exception:
        return None


def delete_session(stable_key: str):
    """
    Wipes the persisted session for this key.
    Called when user picks start fresh or resets data.
    """
    path = _session_path(stable_key)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def session_exists(stable_key: str) -> bool:
    return os.path.exists(_session_path(stable_key))


def format_saved_time(saved_at: float) -> str:
    """
    Returns a human readable string like 3 minutes ago or 2 hours ago.
    """
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