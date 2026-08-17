"""python -m wandb_wrap --project X --job-type Y [--hail-batch-name NAME] -- {command...}

Auto-wraps an arbitrary subprocess command in a wandb run tied to the enclosing Hail
Batch job, so the wrapped script needs zero wandb-specific code. Invoked directly as a
job command string (not through the `hailrun` CLI), so this uses argparse and its own
"--" split rather than typer. `hailrun dispatch --wandb` ships this module to the job as
a loose file (see dispatch.py's WANDB_WRAP_MOUNT) so it runs as top-level `wandb_wrap`,
not `hailrun.wandb_wrap` -- it also still works installed normally, e.g. for manual use.

Empirically verified (not just assumed): a plain subprocess.run() inheriting stdio does
NOT get captured into the run's Logs tab under wandb 0.28 -- confirmed by inspecting a
real run's files via the API (no output.log at all). wandb's console capture wraps the
*parent* process's own stdout, not an fd a child process writes to directly. So instead
we pipe the subprocess and re-print each line from this (parent) process, which wandb's
capture does pick up -- while still also reaching Hail Batch's own per-job log (which
always captures this process's stdout regardless of wandb).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from hailrun.wandb_utils import init_hail_wandb
except ModuleNotFoundError:
    # Dispatch ships this file to the job container as a loose module (alongside a
    # copy of wandb_utils.py on PYTHONPATH) so jobs don't need hailrun itself
    # installed in the image -- only `wandb`. See dispatch.py's WANDB_WRAP_MOUNT.
    from wandb_utils import init_hail_wandb


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        idx = argv.index("--")
        own_argv, command = argv[:idx], argv[idx + 1 :]
    else:
        own_argv, command = argv, []

    parser = argparse.ArgumentParser(prog="python -m wandb_wrap")
    parser.add_argument("--project", required=True)
    parser.add_argument(
        "--job-type",
        default=None,
        help="Defaults to the name of the wrapped script (the first token after '--').",
    )
    parser.add_argument("--hail-batch-name", default=None)
    args = parser.parse_args(own_argv)

    if not command:
        parser.error("no command given after '--'")
    return args, command


_INTERPRETERS = {"python", "python3", "bash", "sh"}


def _default_job_type(command: list[str]) -> str:
    """Name of the wrapped script, skipping a leading interpreter token (and any flag
    that follows it, e.g. `python -m`) if present."""
    token = command[0]
    if token in _INTERPRETERS:
        rest = [t for t in command[1:] if not t.startswith("-")]
        token = rest[0] if rest else token
    return Path(token).stem


def main(argv: list[str] | None = None) -> int:
    args, command = parse_args(argv if argv is not None else sys.argv[1:])

    if args.hail_batch_name and "HAIL_BATCH_NAME" not in os.environ:
        os.environ["HAIL_BATCH_NAME"] = args.hail_batch_name

    job_type = args.job_type or _default_job_type(command)
    run = init_hail_wandb(enabled=True, project=args.project, job_type=job_type, config={})

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if run is not None:
        run.finish(exit_code=proc.returncode)

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
