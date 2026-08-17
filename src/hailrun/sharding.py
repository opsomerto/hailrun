"""Split a big text/csv/parquet file into N shard files, once, at submit time.

Each split function is streaming where the format allows it, and produces
CONTIGUOUS row/line blocks per shard (shard 0 = the first chunk) so a single
shard is easy to reason about when debugging.
"""

import csv
from pathlib import Path


def detect_format(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in (".parquet", ".pq"):
        return "parquet"
    print(
        f"  ! unrecognized extension {suffix!r} for {path}, treating as text. "
        "Pass --shard-format explicitly to override."
    )
    return "text"


def _shard_sizes(total: int, n: int) -> list[int]:
    if total < n:
        raise ValueError(f"cannot split {total} rows/lines into {n} shards (fewer rows than shards)")
    base, remainder = divmod(total, n)
    sizes = [base] * n
    sizes[-1] += remainder
    return sizes


def _shard_path(path: Path, out_dir: Path, i: int) -> Path:
    return out_dir / f"{path.stem}.shard{i:04d}{path.suffix}"


def split_text(path: Path, n: int, out_dir: Path) -> list[Path]:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as f:
        total_lines = sum(1 for _ in f)
    sizes = _shard_sizes(total_lines, n)
    out_paths = [_shard_path(path, out_dir, i) for i in range(n)]
    with open(path) as f:
        for out_path, size in zip(out_paths, sizes):
            with open(out_path, "w") as out:
                for _ in range(size):
                    out.write(f.readline())
    return out_paths


def split_csv(path: Path, n: int, out_dir: Path) -> list[Path]:
    """Uses the stdlib csv module (not naive line-splitting) so multi-line quoted
    fields aren't corrupted. Header row is preserved in every shard."""
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        total_rows = sum(1 for _ in reader)
    sizes = _shard_sizes(total_rows, n)
    out_paths = [_shard_path(path, out_dir, i) for i in range(n)]
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for out_path, size in zip(out_paths, sizes):
            with open(out_path, "w", newline="") as out:
                writer = csv.writer(out)
                writer.writerow(header)
                for _ in range(size):
                    writer.writerow(next(reader))
    return out_paths


def split_parquet(path: Path, n: int, out_dir: Path) -> list[Path]:
    """Lazy pyarrow import -- only needed if this function is actually called
    (hailrun[parquet] extra). Streams via iter_batches(), row-balanced contiguous
    writers, same schema preserved in every shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    sizes = _shard_sizes(total_rows, n)
    boundaries = []
    cum = 0
    for size in sizes:
        boundaries.append(cum + size)
        cum += size

    out_paths = [_shard_path(path, out_dir, i) for i in range(n)]
    writers = [pq.ParquetWriter(str(p), pf.schema_arrow) for p in out_paths]
    try:
        shard_idx = 0
        row_offset = 0
        for batch in pf.iter_batches():
            table = pa.Table.from_batches([batch])
            pos = 0
            while pos < table.num_rows:
                take = min(boundaries[shard_idx] - (row_offset + pos), table.num_rows - pos)
                writers[shard_idx].write_table(table.slice(pos, take))
                pos += take
                if row_offset + pos >= boundaries[shard_idx] and shard_idx < n - 1:
                    shard_idx += 1
            row_offset += table.num_rows
    finally:
        for w in writers:
            w.close()
    return out_paths


def split_file(path: Path, n: int, fmt: str, out_dir: Path) -> list[Path]:
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if n < 1:
        raise ValueError(f"number of shards must be >= 1, got {n}")
    if fmt == "auto":
        fmt = detect_format(path)
    if fmt == "text":
        return split_text(path, n, out_dir)
    if fmt == "csv":
        return split_csv(path, n, out_dir)
    if fmt == "parquet":
        return split_parquet(path, n, out_dir)
    raise ValueError(f"unknown shard format {fmt!r} (expected text, csv, parquet, or auto)")
