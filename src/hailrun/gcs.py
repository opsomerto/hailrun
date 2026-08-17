"""Google Cloud Storage helpers for scripts that read/write local or gs:// paths.

A single storage.Client() is cached via functools.lru_cache, so callers just pass
gs://bucket/path strings. Local paths are handled transparently by the
resolve/exists helpers, letting the same script run unchanged locally or in the cloud.
"""

import functools
import io
import tempfile
from pathlib import Path


def is_gcs(path) -> bool:
    return str(path).startswith("gs://")


@functools.lru_cache(maxsize=1)
def _get_client():
    from google.cloud import storage

    return storage.Client()


def gcs_blob(gcs_path: str):
    """Return a Blob for gs://bucket/name, reusing a single cached client."""
    bucket_name, blob_name = gcs_path[5:].split("/", 1)
    return _get_client().bucket(bucket_name).blob(blob_name)


def gcs_upload(local_path: Path, gcs_path: str) -> None:
    gcs_blob(gcs_path).upload_from_filename(str(local_path))


def gcs_download(gcs_path: str, local_path: Path) -> None:
    gcs_blob(gcs_path).download_to_filename(str(local_path))


def gcs_exists(gcs_path: str) -> bool:
    return gcs_blob(gcs_path).exists()


def list_gcs_files(gcs_dir: str, suffix: str = "") -> list[str]:
    """List files under a gs://bucket/prefix directory, filtered by suffix.

    Returns full gs://bucket/... paths.
    """
    bucket_name, prefix = gcs_dir[5:].split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    blobs = _get_client().bucket(bucket_name).list_blobs(prefix=prefix)
    return [
        f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(suffix)
    ]


def read_table(path: str):
    """Read a csv/csv.gz/parquet table from a local or gs:// path (returns a pandas DataFrame)."""
    import pandas as pd

    src = path
    tmp = None
    if is_gcs(path):
        suffix = ".parquet" if path.endswith(".parquet") else ".csv"
        tmp = Path(tempfile.mktemp(suffix=suffix))
        gcs_download(path, tmp)
        src = str(tmp)
    df = pd.read_parquet(src) if str(src).endswith(".parquet") else pd.read_csv(src)
    if tmp is not None:
        tmp.unlink(missing_ok=True)
    return df


def upload_df_to_gcs(df, gcs_path: str) -> None:
    """Upload a DataFrame directly to GCS (no local temp file).

    Format is chosen by the path suffix: .parquet writes parquet, anything else CSV.
    """
    if str(gcs_path).endswith(".parquet"):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        gcs_blob(gcs_path).upload_from_string(
            buf.getvalue(), content_type="application/octet-stream"
        )
    else:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        gcs_blob(gcs_path).upload_from_string(buf.getvalue(), content_type="text/csv")


def resolve_output(output_dir: str, rel_path: str) -> tuple[Path, str | None]:
    """Resolve a relative output path against a local or gs:// base dir.

    Returns (local_write_path, gcs_dest_or_None). For GCS, local is a tempfile the
    caller writes then uploads to gcs_dest; for local, the parent dir is created.
    """
    if is_gcs(output_dir):
        local = Path(tempfile.mkdtemp()) / Path(rel_path).name
        return local, f"{output_dir.rstrip('/')}/{rel_path}"
    local = Path(output_dir) / rel_path
    local.parent.mkdir(parents=True, exist_ok=True)
    return local, None


def output_exists(output_dir: str, rel_path: str) -> bool:
    if is_gcs(output_dir):
        return gcs_exists(f"{output_dir.rstrip('/')}/{rel_path}")
    return (Path(output_dir) / rel_path).exists()


def download_dir(gcs_dir: str, local_dir: Path) -> None:
    """Download every blob under a gs:// prefix into local_dir, preserving relative paths."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = gcs_dir.rstrip("/")
    for blob_path in list_gcs_files(gcs_dir):
        rel = blob_path[len(prefix) + 1 :]
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        gcs_download(blob_path, dest)


def upload_dir(local_dir: Path, gcs_dir: str) -> None:
    """Upload every file under local_dir to gs://.../<relative path>."""
    local_dir = Path(local_dir)
    gcs_dir = gcs_dir.rstrip("/")
    for path in sorted(local_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_dir).as_posix()
            gcs_upload(path, f"{gcs_dir}/{rel}")
