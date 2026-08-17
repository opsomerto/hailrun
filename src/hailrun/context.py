"""Package the local code directory into a tarball to sync into a Hail job at runtime.

The Docker image only needs *dependencies* baked in -- the code itself is shipped
as build-style context (like `gcloud builds submit`), tarred locally and localized
into the job via Batch.read_input(), which only accepts single files, not directories.
"""

import fnmatch
import hashlib
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import typer

DEFAULT_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.pyc",
    "*.egg-info",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".DS_Store",
    ".ipynb_checkpoints",
]

_IGNORE_FILE_PRECEDENCE = (".gcloudignore", ".dockerignore", ".gitignore")


@dataclass
class ContextTarInfo:
    path: Path
    file_count: int
    total_bytes: int
    sha256: str


def resolve_script_rel(script: Path, context_dir: Path) -> str:
    """Path of `script` relative to `context_dir`, posix style.

    Raises typer.BadParameter if script isn't inside context_dir.
    """
    script = Path(script).resolve()
    context_dir = Path(context_dir).resolve()
    try:
        rel = script.relative_to(context_dir)
    except ValueError:
        raise typer.BadParameter(
            f"script {script} is not inside context dir {context_dir}; "
            "pass --context-dir to point at a directory that contains it."
        )
    return rel.as_posix()


def load_ignore_patterns(context_dir: Path, extra_excludes: list[str]) -> list[str]:
    """.gcloudignore > .dockerignore > .gitignore > hardcoded DEFAULT_EXCLUDES
    (first ignore file found in that precedence order wins, read once). extra_excludes
    (from repeatable --context-exclude) are always appended regardless of source."""
    context_dir = Path(context_dir)
    patterns = None
    for fname in _IGNORE_FILE_PRECEDENCE:
        ignore_file = context_dir / fname
        if ignore_file.exists():
            patterns = [
                line
                for raw in ignore_file.read_text().splitlines()
                if (line := raw.strip()) and not line.startswith("#")
            ]
            break
    if patterns is None:
        patterns = list(DEFAULT_EXCLUDES)
    return patterns + list(extra_excludes)


def _pattern_matches(rel_path: str, pattern: str) -> bool:
    pattern = pattern.strip().rstrip("/")
    if not pattern:
        return False
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    return any(fnmatch.fnmatch(part, pattern) for part in rel_path.split("/"))


def _excluded(rel_path: str, patterns: list[str]) -> bool:
    return any(_pattern_matches(rel_path, p) for p in patterns)


def _walk_files(context_dir: Path, patterns: list[str]) -> list[str]:
    files = []
    for root, dirs, filenames in os.walk(context_dir):
        rel_root = Path(root).relative_to(context_dir).as_posix()
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = sorted(
            d for d in dirs if not _excluded(f"{rel_root}/{d}" if rel_root else d, patterns)
        )
        for fname in sorted(filenames):
            rel_file = f"{rel_root}/{fname}" if rel_root else fname
            if not _excluded(rel_file, patterns):
                files.append(rel_file)
    return sorted(files)


def _content_hash(context_dir: Path, files: list[str]) -> str:
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode())
        h.update((context_dir / rel).read_bytes())
    return h.hexdigest()


def build_context_tar(
    context_dir: Path, extra_excludes: list[str], cache_dir: Path | None = None
) -> ContextTarInfo:
    """Deterministic tar.gz (sorted entries, zeroed mtime/uid/gid) of context_dir,
    minus excluded patterns. Written to a content-hashed path so repeated dry-runs
    during dev don't rebuild an identical tarball (doesn't skip re-upload though --
    ServiceBackend always re-uploads local read_input paths fresh)."""
    context_dir = Path(context_dir).resolve()
    patterns = load_ignore_patterns(context_dir, extra_excludes)
    files = _walk_files(context_dir, patterns)
    if not files:
        raise typer.BadParameter(f"no files to sync in context dir {context_dir} (check excludes)")
    digest = _content_hash(context_dir, files)

    cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir())
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / f"hailrun-context-{digest[:16]}.tar.gz"

    if not tar_path.exists():
        with tarfile.open(tar_path, "w:gz") as tf:
            for rel in files:
                full = context_dir / rel
                info = tf.gettarinfo(str(full), arcname=rel)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with open(full, "rb") as fh:
                    tf.addfile(info, fh)

    total_bytes = sum((context_dir / rel).stat().st_size for rel in files)
    return ContextTarInfo(path=tar_path, file_count=len(files), total_bytes=total_bytes, sha256=digest)
