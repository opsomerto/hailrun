from pathlib import Path

import pytest
import typer

from hailrun import context


def _make_tree(root: Path) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "main.py").write_text("print('hi')\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("junk")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\x00\x01")


def test_build_context_tar_excludes_defaults(tmp_path):
    _make_tree(tmp_path)
    info = context.build_context_tar(tmp_path, extra_excludes=[])
    assert info.file_count == 2  # pkg/__init__.py, pkg/main.py
    assert info.path.exists()


def test_build_context_tar_deterministic_hash(tmp_path):
    _make_tree(tmp_path)
    info1 = context.build_context_tar(tmp_path, extra_excludes=[])
    info2 = context.build_context_tar(tmp_path, extra_excludes=[])
    assert info1.sha256 == info2.sha256


def test_build_context_tar_extra_excludes(tmp_path):
    _make_tree(tmp_path)
    (tmp_path / "pkg" / "data.bin").write_bytes(b"secret")
    info = context.build_context_tar(tmp_path, extra_excludes=["*.bin"])
    assert info.file_count == 2


def test_resolve_script_rel_outside_context_dir(tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    script = other / "script.py"
    script.write_text("")
    context_dir = tmp_path / "ctx"
    context_dir.mkdir()
    with pytest.raises(typer.BadParameter):
        context.resolve_script_rel(script, context_dir)


def test_resolve_script_rel_inside_context_dir(tmp_path):
    script = tmp_path / "sub" / "script.py"
    script.parent.mkdir()
    script.write_text("")
    rel = context.resolve_script_rel(script, tmp_path)
    assert rel == "sub/script.py"


def test_load_ignore_patterns_uses_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n# comment\n\ndata/\n")
    patterns = context.load_ignore_patterns(tmp_path, extra_excludes=["*.tmp"])
    assert "*.log" in patterns
    assert "data/" in patterns
    assert "*.tmp" in patterns
    assert "# comment" not in patterns
