import typer
from dotenv import load_dotenv

from hailrun import __version__, dispatch

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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
