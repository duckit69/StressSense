"""runlog.py - Run record for every script: console copy, manifest row, config snapshot.
Usage in a script:   with RunLog("phase_1_make_windows"): main()
Creates, per run (run_id = <YYYYmmdd-HHMMSS-mmm>_<script>):
  logs/console/<run_id>.txt          full console output (stdout + stderr, incl. tracebacks)
  logs/config_snapshots/<run_id>.json  every UPPERCASE value of config.py at run time
  logs/runs.csv                       one appended row: script, args, start/end, duration, status,
                                      git commit + dirty flag (code only, checked at start),
                                      config hash, versions, CPU, outputs
Outputs = files under logs/, data/{windows,features} and firmware/sshrc/src modified during the run.
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from importlib import metadata

import config as C

# Packages whose versions are written to logs/runs.csv for every run.
_PKGS = ("numpy", "scipy", "scikit-learn", "pandas", "joblib", "pyserial")


# Writes everything printed to two places at once: the screen and the console log file.
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


# Runs a git command in the repository; returns '' if git is missing or fails.
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


# Copies every UPPERCASE setting of config.py, so each run records the exact parameters used.
def _config_snapshot():
    snap = {}
    for k in dir(C):
        if k.isupper():
            v = getattr(C, k)
            snap[k] = json.loads(json.dumps(v, default=str))
    return snap


# Installed version of each package in _PKGS ('n/a' if not installed).
def _versions():
    out = {}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "n/a"
    return out


# Context manager wrapped around every script's main(): starts the console copy on entry,
# and on exit writes the status, duration and output files to logs/runs.csv.
class RunLog:
    def __init__(self, script):
        self.script = script

    def __enter__(self):
        self.t0 = time.time()
        self.start = dt.datetime.now()
        # Unique run id = start time to the millisecond + script name.
        self.run_id = f"{self.start:%Y%m%d-%H%M%S}-{self.start.microsecond // 1000:03d}_{self.script}"
        for d in ("console", "config_snapshots"):
            (C.LOG_DIR / d).mkdir(parents=True, exist_ok=True)
        self.console = open(C.LOG_DIR / "console" / f"{self.run_id}.txt", "w", encoding="utf-8")
        self._out, self._err = sys.stdout, sys.stderr
        # From here on, print() goes to the screen AND to logs/console/<run_id>.txt.
        sys.stdout = _Tee(self._out, self.console)
        sys.stderr = _Tee(self._err, self.console)
        # git state of the CODE at start (logs/ changes during the run must not count as dirty)
        self.commit = _git("rev-parse", "--short", "HEAD")
        st = _git("status", "--porcelain", "--", "python", "firmware", "requirements.txt", ".gitignore")
        self.dirty = bool(st)
        self.dirty_files = " | ".join(l.strip() for l in st.splitlines())
        self.autocrlf = _git("config", "--get", "core.autocrlf") or "unset"
        self.snapshot = _config_snapshot()
        with open(C.LOG_DIR / "config_snapshots" / f"{self.run_id}.json", "w") as f:
            json.dump(self.snapshot, f, indent=1, sort_keys=True)
        print(f"[run {self.run_id}] args: {' '.join(sys.argv[1:]) or '(none)'}")
        return self

    # Files created or modified during this run (by modification time), excluding the run log itself.
    def _outputs(self):
        roots = [C.LOG_DIR, C.WIN_DIR, C.FEAT_DIR, C.ROOT / "firmware" / "sshrc" / "src"]
        found = []
        for r in roots:
            if not r.exists():
                continue
            for dirpath, _, files in os.walk(r):
                for fn in files:
                    p = os.path.join(dirpath, fn)
                    if os.path.getmtime(p) >= self.t0 - 1 and "console" not in p \
                            and "config_snapshots" not in p and not p.endswith("runs.csv"):
                        found.append(os.path.relpath(p, C.ROOT).replace("\\", "/"))
        return sorted(found)

    # Called when the script ends, normally or with an error.
    def __exit__(self, exc_type, exc, tb):
        status = "ok"
        if exc_type is not None:
            status = "error" if exc_type is not KeyboardInterrupt else "interrupted"
            traceback.print_exception(exc_type, exc, tb)
        dur = time.time() - self.t0
        print(f"[run {self.run_id}] status={status} duration={dur:.1f} s")
        sys.stdout, sys.stderr = self._out, self._err
        self.console.close()

        # Short fingerprint of the configuration: two runs with the same hash used identical settings.
        cfg_hash = hashlib.sha256(json.dumps(self.snapshot, sort_keys=True).encode()).hexdigest()[:12]
        row = {"run_id": self.run_id, "script": self.script, "args": " ".join(sys.argv[1:]),
               "start": self.start.isoformat(timespec="seconds"),
               "end": dt.datetime.now().isoformat(timespec="seconds"),
               "duration_s": f"{dur:.1f}", "status": status,
               "error": f"{exc_type.__name__}: {exc}" if exc_type else "",
               "git_commit": self.commit, "git_dirty": self.dirty, "git_dirty_files": self.dirty_files,
               "git_autocrlf": self.autocrlf,
               "config_hash": cfg_hash, "python": platform.python_version(),
               "packages": ";".join(f"{k}={v}" for k, v in _versions().items()),
               "platform": platform.platform(), "cpu_count": os.cpu_count(),
               "outputs": ";".join(self._outputs())}
        path = C.LOG_DIR / "runs.csv"
        if path.exists():                     # columns changed: keep the old manifest as runs_vN.csv
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip()
            if header != ",".join(row.keys()):
                n = 1
                while path.with_name(f"runs_v{n}.csv").exists():
                    n += 1
                path.rename(path.with_name(f"runs_v{n}.csv"))
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        return False          # never swallow the exception
