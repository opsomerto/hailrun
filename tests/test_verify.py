from hailrun.verify import detect_framework, verify_script_args

ARGPARSE_SCRIPT = """\
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--foo", required=True, type=int)
    args = p.parse_args()
    print("RAN REAL LOGIC", args.foo)

if __name__ == "__main__":
    main()
"""

TYPER_SCRIPT = """\
import typer

app = typer.Typer()

@app.command()
def main(foo: int = typer.Option(...)):
    print("RAN REAL LOGIC", foo)

if __name__ == "__main__":
    app()
"""

PLAIN_SCRIPT = """\
import sys
print("plain args:", sys.argv[1:])
"""


def test_detect_framework_argparse(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(ARGPARSE_SCRIPT)
    assert detect_framework(f) == "argparse"


def test_detect_framework_typer(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(TYPER_SCRIPT)
    assert detect_framework(f) == "typer"


def test_detect_framework_none(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(PLAIN_SCRIPT)
    assert detect_framework(f) is None


def test_verify_argparse_ok_does_not_run_real_logic(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(ARGPARSE_SCRIPT)
    result = verify_script_args(f, ["--foo", "3"])
    assert result.status == "ok"


def test_verify_argparse_missing_required(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(ARGPARSE_SCRIPT)
    result = verify_script_args(f, [])
    assert result.status == "bad_args"
    assert "--foo" in result.message


def test_verify_argparse_bad_type(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(ARGPARSE_SCRIPT)
    result = verify_script_args(f, ["--foo", "notanint"])
    assert result.status == "bad_args"
    assert "invalid int value" in result.message


def test_verify_typer_ok(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(TYPER_SCRIPT)
    result = verify_script_args(f, ["--foo", "3"])
    assert result.status == "ok"


def test_verify_typer_missing_required(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(TYPER_SCRIPT)
    result = verify_script_args(f, [])
    assert result.status == "bad_args"
    assert "--foo" in result.message
    assert "\x1b[" not in result.message  # NO_COLOR should strip ANSI codes


def test_verify_skips_when_no_framework_detected(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(PLAIN_SCRIPT)
    result = verify_script_args(f, ["anything"])
    assert result.status == "skipped"


def test_verify_sibling_import_resolves_via_script_dir(tmp_path):
    (tmp_path / "helper.py").write_text("VALUE = 42\n")
    f = tmp_path / "s.py"
    f.write_text(ARGPARSE_SCRIPT.replace("import argparse", "import argparse\nimport helper"))
    result = verify_script_args(f, ["--foo", "3"])
    assert result.status == "ok", result.message


def test_verify_missing_dependency_is_inconclusive_not_blocking(tmp_path):
    f = tmp_path / "s.py"
    f.write_text("import this_module_does_not_exist_xyz\n" + ARGPARSE_SCRIPT)
    result = verify_script_args(f, ["--foo", "3"])
    assert result.status == "inconclusive"
    assert "this_module_does_not_exist_xyz" in result.message


def test_verify_exit_zero_before_parsing_is_inconclusive_not_ok(tmp_path):
    # A script that can sys.exit(0) before ever calling parse_args() -- exit code 0
    # alone must not be read as "args are valid" (it never checked).
    f = tmp_path / "s.py"
    f.write_text(
        "import argparse\n"
        "import sys\n"
        "if len(sys.argv) == 1:\n"
        "    sys.exit(0)\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--foo', required=True, type=int)\n"
        "p.parse_args()\n"
    )
    result = verify_script_args(f, [])
    assert result.status == "inconclusive"
    assert "never reached its arg parser" in result.message


def test_verify_no_args_is_help_multicommand_is_not_ok(tmp_path):
    f = tmp_path / "s.py"
    f.write_text(
        "import typer\n"
        "app = typer.Typer(no_args_is_help=True)\n"
        "@app.command()\n"
        "def sub(foo: int = typer.Option(...)):\n"
        "    print('RAN', foo)\n"
        "@app.command()\n"
        "def other():\n"
        "    pass\n"
        "if __name__ == '__main__':\n"
        "    app()\n"
    )
    result = verify_script_args(f, [])
    assert result.status != "ok"
