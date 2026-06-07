"""
Run a command, streaming stdout+stderr to a timestamped log file on Drive.

Why this exists
---------------
Colab's frontend connection is flaky: closing the tab or losing network for a
moment can hide the output of a running cell — but the kernel keeps going.
Writing a per-invocation log file to mounted Drive means the output is
preserved regardless of what the frontend does.

The log file lives under ``$SNC_DRIVE_LOG_DIR/runs/`` and is named
``{YYYYmmdd_HHMMSS}_{name}.log``. A short header records the command,
host and start time; a footer records the exit code and end time.

Foreground usage (notebook cell)
--------------------------------
    !python scripts/run_logged.py eval_thresh_sweep -- \\
        python src/model_training/evaluate_detection.py \\
            --allowlist auto --absolute-floor 0.5 --relative-cap 0.95 --top-k 3

Output is streamed to BOTH the notebook cell AND the Drive log file.

Background usage (survives a disconnected frontend)
---------------------------------------------------
    import subprocess
    proc = subprocess.Popen(
        ['python', 'scripts/run_logged.py', '--background',
         'eval_thresh_sweep', '--',
         'python', 'src/model_training/evaluate_detection.py',
         '--allowlist', 'auto', '--absolute-floor', '0.5',
         '--relative-cap', '0.95', '--top-k', '3'],
    )
    print('PID', proc.pid)

The wrapper detaches from the terminal so the cell returns immediately. The
log file fills up as the command produces output, and a final ``# Exit:``
line marks completion. To follow:
    !tail -f $SNC_DRIVE_LOG_DIR/runs/<filename>
"""
from __future__ import annotations

import argparse
import datetime
import os
import socket
import subprocess
import sys
from pathlib import Path

DRIVE_LOG_ENV = "SNC_DRIVE_LOG_DIR"


def _log_dir() -> Path:
    """Drive log directory, or ``/tmp/snc_run_logs`` as a local fallback."""
    raw = os.environ.get(DRIVE_LOG_ENV, "").strip()
    return Path(raw) if raw else Path("/tmp/snc_run_logs")


def _log_path(name: str) -> Path:
    """Timestamped log file path under ``<log_dir>/runs/``."""
    runs_dir = _log_dir() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return runs_dir / f"{ts}_{safe}.log"


def _write_header(f, name: str, cmd: list) -> None:
    """Banner so the file is self-describing without needing the invocation."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    f.write(f"# name    : {name}\n")
    f.write(f"# command : {' '.join(cmd)}\n")
    f.write(f"# host    : {socket.gethostname()}\n")
    f.write(f"# started : {now}\n")
    f.write(f"# cwd     : {os.getcwd()}\n")
    f.write("# " + "=" * 70 + "\n\n")
    f.flush()


def _write_footer(f, exit_code: int) -> None:
    """Final line so you can spot completion at the bottom of the file."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    f.write("\n# " + "=" * 70 + "\n")
    f.write(f"# ended   : {now}\n")
    f.write(f"# exit    : {exit_code}\n")
    f.flush()


def run_foreground(name: str, cmd: list) -> int:
    """Run ``cmd``, streaming to both the terminal and the Drive log file."""
    log_path = _log_path(name)
    print(f"[run_logged] logging to: {log_path}", flush=True)
    with log_path.open("w", buffering=1) as f:
        _write_header(f, name, cmd)
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        proc.wait()
        _write_footer(f, proc.returncode)
    print(f"[run_logged] exit: {proc.returncode}", flush=True)
    return proc.returncode


def run_background(name: str, cmd: list) -> None:
    """Detach a child that runs ``cmd`` with stdout+stderr piped into the log.

    Returns immediately; prints the child PID and log path. The detached
    runner waits for the command to finish so the log gets a proper
    ``# exit: N`` footer — handy for telling "still running" from "done"
    just by tailing the file.
    """
    log_path = _log_path(name)
    # Write the header now so the file is readable before any output arrives.
    with log_path.open("w") as f:
        _write_header(f, name, cmd)

    # A tiny Python subprocess starts the real command, waits for it, and
    # writes the footer when it exits. start_new_session detaches the
    # runner from the controlling terminal so the cell can return.
    runner = (
        "import json, sys, subprocess, datetime\n"
        "argv = json.loads(sys.argv[1])\n"
        "log_path = sys.argv[2]\n"
        "with open(log_path, 'a', buffering=1) as f:\n"
        "    p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)\n"
        "    p.wait()\n"
        "    now = datetime.datetime.now(datetime.timezone.utc).isoformat()\n"
        "    f.write('\\n# ' + '=' * 70 + '\\n')\n"
        "    f.write(f'# ended   : {now}\\n')\n"
        "    f.write(f'# exit    : {p.returncode}\\n')\n"
    )
    import json as _json
    proc = subprocess.Popen(
        [sys.executable, "-c", runner, _json.dumps(cmd), str(log_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[run_logged] log : {log_path}", flush=True)
    print(f"[run_logged] pid : {proc.pid}", flush=True)
    print("[run_logged] tail with:   !tail -f " + str(log_path), flush=True)
    print("[run_logged] check with:  !ps -p " + str(proc.pid), flush=True)


def _split_argv(argv: list) -> tuple:
    """Split ``[flags..., name, '--', cmd...]`` into ``(opts, name, cmd)``."""
    if "--" not in argv:
        raise SystemExit(
            "missing '--' separator. Usage:\n"
            "  python scripts/run_logged.py [--background] <name> -- <cmd...>"
        )
    sep = argv.index("--")
    head, cmd = argv[:sep], argv[sep + 1:]
    if not cmd:
        raise SystemExit("empty command after '--'")
    return head, cmd


if __name__ == "__main__":
    head, cmd = _split_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Run a command, mirror output to a Drive log file.",
    )
    parser.add_argument(
        "--background", action="store_true",
        help="Detach the child process so the cell returns immediately. The "
             "log file keeps filling up regardless of frontend connection.",
    )
    parser.add_argument(
        "name",
        help="Short label included in the log filename, e.g. 'eval_thresh_sweep'.",
    )
    args = parser.parse_args(head)

    if args.background:
        run_background(args.name, cmd)
        sys.exit(0)
    else:
        sys.exit(run_foreground(args.name, cmd))
