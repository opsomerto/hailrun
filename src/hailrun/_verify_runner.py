"""Internal helper subprocess for `hailrun dispatch --verify-args`.

Invoked as:
    python -m hailrun._verify_runner <framework> <script> <extra_pythonpath> <script args...>

<extra_pythonpath> is a os.pathsep-joined string of extra directories to prepend to
sys.path (context dir / src-layout dir, mirroring the job's real PYTHONPATH), or "".

Patches the target script's argument-parsing entrypoint (argparse.parse_args or
click.Command.invoke) to abort execution right after parsing succeeds, so the
script's real logic never runs locally -- only its arg parsing does.

This is a heuristic, not a guarantee:
- argparse: only the common top-level `parser.parse_args()` call is patched, not a
  bare `parse_known_args()` call used on its own.
- click/typer: patches `Command.invoke`, which fires only after parsing succeeds and
  right before the script's real command body would run.
- If the script never reaches its parser (e.g. exits early, or parses conditionally),
  this can't confirm anything -- see HAILRUN_VERIFY_UNKNOWN below.
- Any import-time side effects in the script (heavy imports, network calls, etc.)
  still execute locally, since patching happens before runpy.run_path. A script that
  never calls its parser runs to completion locally too, bounded only by the caller's
  subprocess timeout -- this flag runs real (if bounded) script code on the
  submitting machine, not a sandbox.
"""

import os
import runpy
import sys
from pathlib import Path


class _ParseOK(Exception):
    pass


def _patch_argparse() -> None:
    import argparse

    orig_parse_args = argparse.ArgumentParser.parse_args

    def patched(self, args=None, namespace=None):
        orig_parse_args(self, args, namespace)
        raise _ParseOK()

    argparse.ArgumentParser.parse_args = patched


def _patch_click() -> None:
    import click

    def patched(self, ctx):
        raise _ParseOK()

    click.Command.invoke = patched


def main() -> None:
    framework = sys.argv[1]
    script = sys.argv[2]
    extra_pythonpath = sys.argv[3]
    script_args = sys.argv[4:]
    sys.argv = [script] + script_args

    script_dir = str(Path(script).resolve().parent)
    extra_dirs = [p for p in extra_pythonpath.split(os.pathsep) if p]
    for p in reversed([script_dir, *extra_dirs]):
        if p not in sys.path:
            sys.path.insert(0, p)

    if framework == "argparse":
        _patch_argparse()
    elif framework in ("click", "typer"):
        _patch_click()

    try:
        runpy.run_path(script, run_name="__main__")
    except _ParseOK:
        print("HAILRUN_VERIFY_OK")
        sys.exit(0)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code == 0:
            # Exited 0 without ever raising _ParseOK -- e.g. no_args_is_help printed
            # help and exited before parsing. Not proof the args are valid.
            print("HAILRUN_VERIFY_UNKNOWN", file=sys.stderr)
            sys.exit(3)
        sys.exit(code)
    else:
        # Script ran to completion without ever calling parse_args()/invoke() --
        # can't confirm one way or the other.
        print("HAILRUN_VERIFY_UNKNOWN", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
