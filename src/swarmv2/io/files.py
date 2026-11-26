"""File handling that survives multi-hour runs on macOS, Windows and Linux.

On Windows a file that another process holds open (the viewer, an antivirus scanner, a search indexer
or a cloud-sync client) cannot be replaced or renamed for a moment; every replace and rename here waits
that out instead of ending the run. JSON is written atomically and never fails on NaN, and JSON-lines
logs tolerate the partial last line that a hard kill can leave behind.
"""
import errno
import io
import json
import math
import os
import time
from pathlib import Path

RETRY_SECONDS = 60.0
_TRANSIENT_WINERRORS = {5, 32, 33}  # access denied, sharing violation, lock violation
_TRANSIENT_ERRNOS = {errno.EACCES, errno.EBUSY, getattr(errno, "ETXTBSY", errno.EBUSY)}


def _transient(error):
    if isinstance(error, (FileNotFoundError, IsADirectoryError, NotADirectoryError)):
        return False
    return (isinstance(error, PermissionError) or getattr(error, "winerror", None) in _TRANSIENT_WINERRORS
            or error.errno in _TRANSIENT_ERRNOS)


def retry(action, timeout=RETRY_SECONDS):
    """Run `action()` again with backoff while it fails with a transient lock error, for up to `timeout` s."""
    delay, deadline = 0.05, time.monotonic() + timeout
    while True:
        try:
            return action()
        except OSError as error:
            if not _transient(error) or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def replace_with_retry(source, target, timeout=RETRY_SECONDS):
    retry(lambda: os.replace(source, target), timeout)


def rename_with_retry(source, target, timeout=RETRY_SECONDS):
    """Rename a file or folder (same file system), waiting out transient locks on the files inside."""
    retry(lambda: os.rename(source, target), timeout)


def _temporary(path):
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def _write_atomic(path, write, timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary(path)
    try:
        write(temp)
        replace_with_retry(temp, path, timeout)
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def save_torch_atomic(data, path, timeout=RETRY_SECONDS):
    import torch
    _write_atomic(path, lambda temp: torch.save(data, temp), timeout)


def load_torch(path, map_location="cpu"):
    """torch.load of a checkpoint that holds the file open only while its bytes are read."""
    import torch
    data = retry(lambda: Path(path).read_bytes(), timeout=10.0)
    return torch.load(io.BytesIO(data), map_location=map_location, weights_only=True)


def sanitize(value):
    """JSON-safe copy: NaN and infinity become None, NumPy scalars and arrays become Python values."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return sanitize(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dumps(value, indent=None):
    return json.dumps(sanitize(value), indent=indent, allow_nan=False)


def write_text_atomic(path, text, timeout=RETRY_SECONDS):
    _write_atomic(path, lambda temp: temp.write_text(text, encoding="utf-8"), timeout)


def write_json_atomic(path, value, indent=2, timeout=RETRY_SECONDS):
    write_text_atomic(path, dumps(value, indent) + "\n", timeout)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, row):
    """Append one JSON row; a partial last line left by a hard kill is closed off first."""
    path = Path(path)
    prefix = ""
    try:
        if path.stat().st_size:
            with path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                prefix = "" if handle.read(1) == b"\n" else "\n"
    except FileNotFoundError:
        pass
    line = prefix + dumps(row) + "\n"

    def write():
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    retry(write)


def read_jsonl(path):
    """Rows of a JSON-lines file; unreadable lines (a hard kill mid-write) are skipped, not fatal."""
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return rows


def portable_path(path):
    """Path text that reads back the same on every OS (forward slashes), for config and summary files."""
    return None if path is None else Path(path).as_posix()


def local_path(text):
    """Path from text written on any OS; Windows backslashes are accepted on macOS and Linux too."""
    return None if not text else Path(str(text).replace("\\", "/"))
