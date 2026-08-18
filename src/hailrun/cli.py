import subprocess
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv

from hailrun import __version__, dispatch
from hailrun.verify import TIMEOUT_S

load_dotenv(".env")

app = typer.Typer(add_completion=False, no_args_is_help=True, help="hailrun: run local scripts as Hail Batch jobs.")
app.command(
    name="dispatch",
    help="Dispatch a local script as one or more Hail Batch jobs.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)(dispatch.run)


@app.command()
def version() -> None:
    """Print the hailrun version."""
    typer.echo(__version__)


def resolve_dispatch_script(dispatch_argv: list[str]) -> Path | None:
    """Which local script `hailrun dispatch <dispatch_argv>` would run, without
    requiring hailrun's own required options (--hail-image etc.) -- used to show that
    script's own --help alongside dispatch's. None if unresolvable or the resolved
    script doesn't exist locally (e.g. --baked, where the script lives in the image)."""
    command = typer.main.get_command(app).commands["dispatch"]
    try:
        ctx = command.make_context("dispatch", list(dispatch_argv), resilient_parsing=True)
    except Exception:
        return None
    script = ctx.params.get("script")
    if not script:
        return None
    path = Path(script)
    return path if path.exists() else None


def print_script_help(dispatch_argv: list[str]) -> None:
    script = resolve_dispatch_script(dispatch_argv)
    if script is None:
        return
    typer.echo(f"\n--- {script} --help ---\n")
    try:
        proc = subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=TIMEOUT_S
        )
        typer.echo(proc.stdout or proc.stderr)
    except Exception as e:
        typer.echo(f"(could not run {script} --help: {e})")


def main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["dispatch"] and "--help" in argv[1:]:
        try:
            app()
        finally:
            print_script_help(argv[1:])
        return
    app()


if __name__ == "__main__":
    main()
