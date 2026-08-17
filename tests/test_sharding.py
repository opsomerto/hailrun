import csv

import pytest

from hailrun import sharding


def test_detect_format(tmp_path):
    assert sharding.detect_format(tmp_path / "a.csv") == "csv"
    assert sharding.detect_format(tmp_path / "a.parquet") == "parquet"
    assert sharding.detect_format(tmp_path / "a.pq") == "parquet"
    assert sharding.detect_format(tmp_path / "a.txt") == "text"
    assert sharding.detect_format(tmp_path / "a.unknown") == "text"


def test_split_text_line_counts(tmp_path):
    src = tmp_path / "data.txt"
    lines = [f"line-{i}\n" for i in range(97)]
    src.write_text("".join(lines))
    out_dir = tmp_path / "out"
    shards = sharding.split_text(src, 5, out_dir)
    assert len(shards) == 5
    total = sum(len(p.read_text().splitlines()) for p in shards)
    assert total == 97
    assert shards[0].read_text().splitlines()[0] == "line-0"


def test_split_text_too_few_lines(tmp_path):
    src = tmp_path / "data.txt"
    src.write_text("only one line\n")
    with pytest.raises(ValueError):
        sharding.split_text(src, 5, tmp_path / "out")


def test_split_csv_preserves_header_and_rows(tmp_path):
    src = tmp_path / "data.csv"
    with open(src, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "value"])
        for i in range(23):
            w.writerow([i, i * 10])
    out_dir = tmp_path / "out"
    shards = sharding.split_csv(src, 4, out_dir)
    assert len(shards) == 4
    total_rows = 0
    for p in shards:
        with open(p, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["id", "value"]
            total_rows += sum(1 for _ in reader)
    assert total_rows == 23


def test_split_csv_handles_multiline_quoted_field(tmp_path):
    src = tmp_path / "data.csv"
    with open(src, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "note"])
        w.writerow([1, "line one\nline two"])
        w.writerow([2, "plain"])
        w.writerow([3, "another\nmulti-line value"])
    shards = sharding.split_csv(src, 3, tmp_path / "out")
    all_rows = []
    for p in shards:
        with open(p, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            all_rows.extend(reader)
    assert len(all_rows) == 3
    assert ["1", "line one\nline two"] in all_rows
    assert ["3", "another\nmulti-line value"] in all_rows


def test_split_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = tmp_path / "data.parquet"
    table = pa.table({"id": list(range(50)), "value": [i * 2 for i in range(50)]})
    pq.write_table(table, src)

    shards = sharding.split_parquet(src, 4, tmp_path / "out")
    assert len(shards) == 4
    total_rows = 0
    for p in shards:
        t = pq.read_table(p)
        assert t.schema.equals(table.schema)
        total_rows += t.num_rows
    assert total_rows == 50


def test_split_file_unknown_format(tmp_path):
    src = tmp_path / "data.txt"
    src.write_text("a\nb\nc\n")
    with pytest.raises(ValueError):
        sharding.split_file(src, 2, "bogus", tmp_path / "out")


def test_list_shard_dir_returns_sorted_files(tmp_path):
    for name in ["shard2.txt", "shard10.txt", "shard1.txt"]:
        (tmp_path / name).write_text("x")
    files = sharding.list_shard_dir(tmp_path)
    assert [p.name for p in files] == ["shard1.txt", "shard10.txt", "shard2.txt"]


def test_list_shard_dir_skips_subdirectories(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.txt").write_text("x")
    files = sharding.list_shard_dir(tmp_path)
    assert [p.name for p in files] == ["a.txt", "b.txt"]


def test_list_shard_dir_skips_dotfiles(tmp_path):
    (tmp_path / ".DS_Store").write_text("x")
    (tmp_path / "a.txt").write_text("x")
    files = sharding.list_shard_dir(tmp_path)
    assert [p.name for p in files] == ["a.txt"]


def test_list_shard_dir_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        sharding.list_shard_dir(tmp_path)


def test_list_shard_dir_only_dotfiles_raises(tmp_path):
    (tmp_path / ".DS_Store").write_text("x")
    with pytest.raises(ValueError):
        sharding.list_shard_dir(tmp_path)
