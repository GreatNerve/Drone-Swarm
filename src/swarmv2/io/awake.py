"""Keep the computer from sleeping while a long training runs (macOS, Windows and Linux)."""
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _start():
    if sys.platform == "win32":
        import ctypes
        if ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED):
            return "windows"
        return None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        command = ["caffeinate", "-i", "-w", str(os.getpid())]
    elif shutil.which("systemd-inhibit") and shutil.which("tail"):
        # `tail --pid` ends with this process, so the inhibitor can never outlive a crashed run.
        command = ["systemd-inhibit", "--what=idle:sleep", "--who=SwarmFlow", "--why=training",
                   "tail", f"--pid={os.getpid()}", "-f", "/dev/null"]
    else:
        return None
    try:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError:
        return None


def _stop(handle):
    if handle == "windows":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    elif handle is not None and handle.poll() is None:
        handle.terminate()


@contextmanager
def keep_awake(enabled=True):
    """While active, system sleep is blocked (the display may still turn off). A no-op when unsupported."""
    handle = None
    if enabled:
        try:
            handle = _start()
        except (OSError, AttributeError):
            handle = None
        if handle is not None:
            print("Keeping the computer awake while this runs (pass --allow-sleep to disable)", flush=True)
    try:
        yield
    finally:
        if handle is not None:
            try:
                _stop(handle)
            except (OSError, AttributeError):
                pass
