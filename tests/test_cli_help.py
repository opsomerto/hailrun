import pytest

from hailrun import cli


def test_resolve_dispatch_script_finds_existing_script(tmp_path):
    script = tmp_path / "s.py"
    script.write_text("print('hi')\n")
    resolved = cli.resolve_dispatch_script([str(script), "--foo", "bar", "--help"])
    assert resolved == script


def test_resolve_dispatch_script_none_for_missing_script():
    assert cli.resolve_dispatch_script(["/nonexistent/script.py", "--help"]) is None


def test_resolve_dispatch_script_none_for_baked_no_script():
    assert cli.resolve_dispatch_script(["--baked", "--help"]) is None


def test_main_help_prints_script_help_without_requiring_hail_options(tmp_path, monkeypatch, capsys):
    script = tmp_path / "s.py"
    script.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser(description='demo')\n"
        "p.add_argument('--foo', required=True)\n"
        "p.parse_args()\n"
    )
    monkeypatch.setattr("sys.argv", ["hailrun", "dispatch", str(script), "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "Dispatch a local script" in out  # dispatch's own help
    assert f"--- {script} --help ---" in out
    assert "demo" in out
    assert "--foo" in out
