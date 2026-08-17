"""Best-effort local check that a target script's own argparse/click/typer parser
accepts the given script args, before submitting to Hail Batch.

This is a heuristic, not a guarantee -- see hailrun._verify_runner for the mechanism
and its known limitations (e.g. import-time side effects in the target script still
execute locally, since parsing must actually run to be checked).
"""

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_RUNNER = Path(__file__).parent / "_verify_runner.py"
_TIMEOUT_S = 60


@dataclass
class VerifyResult:
    status: str  # "ok" | "bad_args" | "skipped" | "inconclusive"
    message: str = ""


def detect_framework(script_path: Path) -> str | None:
    """Static (no-exec) scan of the script's top-level imports for argparse/click/typer."""
    try:
        tree = ast.parse(script_path.read_text(), filename=str(script_path))
    except (SyntaxError, OSError):
        return None

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    if "typer" in imported:
        return "typer"
    if "click" in imported:
        return "click"
    if "argparse" in imported:
        return "argparse"
    return None


def verify_script_args(
    script_path: Path,
    script_args: list[str],
    extra_pythonpath: list[str] | None = None,
) -> VerifyResult:
    """extra_pythonpath: extra directories to put on sys.path before importing the
    script, mirroring the job's real PYTHONPATH (context dir, src-layout dir) so
    sibling-module imports that work in the real job don't false-fail here."""
    framework = detect_framework(script_path)
    if framework is None:
        return VerifyResult("skipped", "no argparse/click/typer import detected in script")

    pythonpath_arg = os.pathsep.join(extra_pythonpath or [])
    try:
        proc = subprocess.run(
            [sys.executable, str(_RUNNER), framework, str(script_path), pythonpath_arg, *script_args],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            env={**os.environ, "NO_COLOR": "1", "TERM": "dumb"},
        )
    except subprocess.TimeoutExpired:
        return VerifyResult("inconclusive", f"local verify timed out after {_TIMEOUT_S}s")

    if "HAILRUN_VERIFY_OK" in proc.stdout:
        return VerifyResult("ok")
    if "HAILRUN_VERIFY_UNKNOWN" in proc.stderr:
        return VerifyResult(
            "inconclusive", "script never reached its arg parser (exited or completed before parsing)"
        )
    if proc.returncode == 2:
        return VerifyResult("bad_args", proc.stderr.strip())
    return VerifyResult(
        "inconclusive",
        f"couldn't verify locally (exit {proc.returncode}): {proc.stderr.strip()[-2000:]}",
    )
