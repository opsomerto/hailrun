from pathlib import Path

import pytest

from hailrun import gcs


class _FakeBlob:
    def __init__(self, store, bucket_name, name):
        self._store = store
        self._bucket_name = bucket_name
        self.name = name

    def upload_from_filename(self, path):
        self._store[(self._bucket_name, self.name)] = Path(path).read_bytes()

    def download_to_filename(self, path):
        Path(path).write_bytes(self._store[(self._bucket_name, self.name)])

    def exists(self):
        return (self._bucket_name, self.name) in self._store

    def upload_from_string(self, data, content_type=None):
        self._store[(self._bucket_name, self.name)] = data if isinstance(data, bytes) else data.encode()


class _FakeBucket:
    def __init__(self, store, name):
        self._store = store
        self.name = name

    def blob(self, name):
        return _FakeBlob(self._store, self.name, name)

    def list_blobs(self, prefix=""):
        return [
            _FakeBlob(self._store, self.name, key)
            for (bucket, key) in self._store
            if bucket == self.name and key.startswith(prefix)
        ]


class _FakeClient:
    def __init__(self):
        self.store: dict = {}

    def bucket(self, name):
        return _FakeBucket(self.store, name)


@pytest.fixture
def fake_gcs(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(gcs, "_get_client", lambda: client)
    return client


def test_is_gcs():
    assert gcs.is_gcs("gs://bucket/path")
    assert not gcs.is_gcs("/local/path")


def test_upload_download_exists_round_trip(tmp_path, fake_gcs):
    src = tmp_path / "in.txt"
    src.write_text("hello")
    gcs.gcs_upload(src, "gs://mybucket/dir/in.txt")
    assert gcs.gcs_exists("gs://mybucket/dir/in.txt")
    assert not gcs.gcs_exists("gs://mybucket/dir/missing.txt")

    dst = tmp_path / "out.txt"
    gcs.gcs_download("gs://mybucket/dir/in.txt", dst)
    assert dst.read_text() == "hello"


def test_list_gcs_files(fake_gcs):
    fake_gcs.store[("mybucket", "dir/a.csv")] = b"a"
    fake_gcs.store[("mybucket", "dir/b.txt")] = b"b"
    fake_gcs.store[("mybucket", "other/c.csv")] = b"c"
    files = gcs.list_gcs_files("gs://mybucket/dir", suffix=".csv")
    assert files == ["gs://mybucket/dir/a.csv"]


def test_download_dir_upload_dir_round_trip(tmp_path, fake_gcs):
    local_src = tmp_path / "src"
    (local_src / "sub").mkdir(parents=True)
    (local_src / "top.txt").write_text("top")
    (local_src / "sub" / "nested.txt").write_text("nested")

    gcs.upload_dir(local_src, "gs://mybucket/tree")

    local_dst = tmp_path / "dst"
    gcs.download_dir("gs://mybucket/tree", local_dst)
    assert (local_dst / "top.txt").read_text() == "top"
    assert (local_dst / "sub" / "nested.txt").read_text() == "nested"


def test_resolve_output_and_output_exists_local(tmp_path):
    local, gcs_dest = gcs.resolve_output(str(tmp_path), "sub/out.txt")
    assert gcs_dest is None
    assert local.parent.exists()
    local.write_text("x")
    assert gcs.output_exists(str(tmp_path), "sub/out.txt")


def test_resolve_output_gcs(fake_gcs):
    local, gcs_dest = gcs.resolve_output("gs://mybucket/base", "sub/out.txt")
    assert gcs_dest == "gs://mybucket/base/sub/out.txt"
    assert local.name == "out.txt"
