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


def _shard_file(tmp_path, n=3):
    f = tmp_path / "data.txt"
    f.write_text("".join(f"line-{i}\n" for i in range(n)))
    return f


def test_shard_default_prepends_positional(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS + ["myscript.py", "--hail-shards", "3", "--shard-input", str(shard_file), "--name", "foo"],
    )
    assert result.exit_code == 0, result.output
    assert "job 0000: python /app/myscript.py " in result.output
    # shard value comes before --name foo, i.e. leading positional
    for line in result.output.splitlines():
        if "job 0000:" in line:
            assert "--name foo" in line
            cmd = line.split("python /app/myscript.py ", 1)[1]
            assert not cmd.startswith("--name")
            assert cmd.endswith("--name foo")


def test_shard_flag_accepts_dashed_value(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "--arg-a",
            "--other", "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--arg-a" in result.output


def test_shard_flag_inserts_when_absent(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "--arg-a",
            "--other", "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--other x --arg-a " in result.output


def test_shard_flag_no_dashes_normalizes(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "arg-a",
            "--other", "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "--arg-a" in result.output
    assert "--other x --arg-a" in result.output


def test_shard_flag_replaces_when_present(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "--arg-a",
            "--arg-a", "placeholder", "--other", "x",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "placeholder" not in result.output
    for line in result.output.splitlines():
        if "job 0000:" in line:
            assert "--arg-a" in line
            assert "--other x" in line


def test_shard_flag_replaces_inline_equals_form(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "--arg-a",
            "--arg-a=placeholder",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "placeholder" not in result.output
    assert "--arg-a=" in result.output


def test_shard_flag_when_name_is_last_token(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "--arg-a",
            "--other", "x", "--arg-a",
        ],
    )
    assert result.exit_code == 0, result.output
    for line in result.output.splitlines():
        if "job 0000:" in line:
            assert "--other x --arg-a " in line
            tail = line.split("--arg-a ", 1)[1].strip()
            assert tail and "shard0000" in tail


def test_shard_flag_and_literal_shard_conflict_errors(tmp_path):
    shard_file = _shard_file(tmp_path)
    result = runner.invoke(
        app,
        BASE_ARGS
        + [
            "myscript.py",
            "--hail-shards", "3",
            "--shard-input", str(shard_file),
            "--shard-flag", "--arg-a",
            "--arg-a", "{shard}",
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_shard_flag_without_shard_input_errors():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-flag", "--arg-a"])
    assert result.exit_code != 0
    assert "--shard-flag requires --shard-input" in result.output


def test_shard_input_directory_infers_job_count(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(3):
        (shard_dir / f"part{i}.csv").write_text("id\n1\n")
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-input", str(shard_dir)])
    assert result.exit_code == 0, result.output
    assert "Submitting 3 job(s)" in result.output
    assert "part0.csv" in result.output
    assert "part1.csv" in result.output
    assert "part2.csv" in result.output


def test_shard_input_directory_hail_shards_mismatch_errors(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(3):
        (shard_dir / f"part{i}.csv").write_text("id\n1\n")
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-input", str(shard_dir), "--hail-shards", "5"])
    assert result.exit_code != 0
    assert "does not match" in result.output


def test_shard_input_directory_hail_shards_matching_ok(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(3):
        (shard_dir / f"part{i}.csv").write_text("id\n1\n")
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-input", str(shard_dir), "--hail-shards", "3"])
    assert result.exit_code == 0, result.output
    assert "Submitting 3 job(s)" in result.output


def test_shard_input_nonexistent_path_errors():
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-input", "/nonexistent/path/xyz"])
    assert result.exit_code != 0
    assert "Error: --shard-input path not found" in result.output


def test_shard_input_empty_directory_errors(tmp_path):
    shard_dir = tmp_path / "empty"
    shard_dir.mkdir()
    result = runner.invoke(app, BASE_ARGS + ["myscript.py", "--shard-input", str(shard_dir)])
    assert result.exit_code != 0
    assert "contains no files" in result.output


def test_shard_format_ignored_in_directory_mode_warns(tmp_path):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    for i in range(2):
        (shard_dir / f"part{i}.csv").write_text("id\n1\n")
    result = runner.invoke(
        app, BASE_ARGS + ["myscript.py", "--shard-input", str(shard_dir), "--shard-format", "csv"]
    )
    assert result.exit_code == 0, result.output
