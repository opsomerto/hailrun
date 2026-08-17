from typer.testing import CliRunner

from hailrun.cli import app

runner = CliRunner()

BASE_ARGS = [
    "dispatch",
    "--hail-image", "img:latest",
    "--hail-billing-project", "bp",
    "--hail-tmp-bucket", "gs://b/tmp",
    "--baked",
    "--hail-dry-run",
]


def test_script_only():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py"])
    assert result.exit_code == 0, result.output
    assert "python /app/myscript.py" in result.output


def test_script_plus_flags_after():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--foo", "bar", "--verbose"])
    assert result.exit_code == 0, result.output
    assert "python /app/myscript.py --foo bar --verbose" in result.output


def test_unknown_flag_forwarded_untouched():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--custom-flag", "value", "-x"])
    assert result.exit_code == 0, result.output
    assert "python /app/myscript.py --custom-flag value -x" in result.output


def test_explicit_separator_still_works():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--", "--hail-image", "shouldnotparse"])
    assert result.exit_code == 0, result.output
    assert "python /app/myscript.py --hail-image shouldnotparse" in result.output


def test_shard_input_requires_hail_shards_gt_1(tmp_path):
    shard_file = tmp_path / "data.csv"
    shard_file.write_text("id\n1\n2\n3\n")
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-input", str(shard_file)])
    assert result.exit_code != 0
    assert "requires --hail-shards N with N > 1" in result.output


def test_hail_shards_without_shard_input_errors():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--hail-shards", "4"])
    assert result.exit_code != 0
    assert "requires --shard-input" in result.output
