# =============================================================================
# Non-blocking shell commands for Streamlit (Stop kills process tree on Windows)
# =============================================================================
# Used by 47.py "Run cmd". Long-lived commands (e.g. streamlit run) block the
# whole app if subprocess.run() is used; this runner keeps the UI responsive.
# =============================================================================

from __future__ import annotations

import os
import subprocess
import threading
from typing import List, Optional, Tuple


class BackgroundShellRunner:
    """Run a shell command in the background; capture stdout/stderr; Stop kills the tree."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.proc: Optional[subprocess.Popen] = None
        self.stdout_parts: List[str] = []
        self.stderr_parts: List[str] = []
        self.returncode: Optional[int] = None
        self.command: str = ""
        self._start_error: Optional[str] = None
        self.run_id: int = 0

    def is_running(self) -> bool:
        with self._lock:
            if self.proc is None:
                return False
            return self.proc.poll() is None

    def start(self, cmd: str, cwd: Optional[str]) -> None:
        cmd = (cmd or "").strip()
        if not cmd:
            with self._lock:
                self._start_error = "Empty command."
            return

        self.stop()

        with self._lock:
            self.stdout_parts = []
            self.stderr_parts = []
            self.returncode = None
            self.command = cmd
            self._start_error = None
            self.proc = None

        try:
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                proc = subprocess.Popen(
                    ["cmd.exe", "/c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or None,
                    creationflags=flags,
                    bufsize=1,
                )
            else:
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or None,
                    start_new_session=True,
                    bufsize=1,
                )
        except Exception as e:
            with self._lock:
                self._start_error = str(e)
            return

        with self._lock:
            self.run_id += 1
            self.proc = proc
            out_parts = self.stdout_parts
            err_parts = self.stderr_parts

        def reader(stream, parts: List[str]) -> None:
            try:
                for line in iter(stream.readline, ""):
                    with self._lock:
                        parts.append(line)
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        threading.Thread(
            target=reader, args=(proc.stdout, out_parts), daemon=True
        ).start()
        threading.Thread(
            target=reader, args=(proc.stderr, err_parts), daemon=True
        ).start()

        def waiter() -> None:
            rc = proc.wait()
            with self._lock:
                self.returncode = rc

        threading.Thread(target=waiter, daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            proc = self.proc
        if not proc or proc.poll() is not None:
            return
        pid = proc.pid
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=25,
                )
            else:
                import signal

                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    try:
                        proc.terminate()
                    except Exception:
                        pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=20)
        except Exception:
            pass
        with self._lock:
            if self.returncode is None:
                self.returncode = -1
            self.stderr_parts.append(
                "\n[Process terminated with **Stop** (and child processes on Windows).]\n"
            )

    def snapshot(self) -> Tuple[str, str, Optional[int], bool, Optional[str], str, int]:
        """stdout, stderr, returncode, is_running, start_error, command, run_id."""
        with self._lock:
            so = "".join(self.stdout_parts)
            se = "".join(self.stderr_parts)
            rc = self.returncode
            err = self._start_error
            cmd = self.command
            run = self.proc is not None and self.proc.poll() is None
            rid = self.run_id
        return so, se, rc, run, err, cmd, rid

    def pop_start_error(self) -> Optional[str]:
        with self._lock:
            e = self._start_error
            self._start_error = None
        return e
