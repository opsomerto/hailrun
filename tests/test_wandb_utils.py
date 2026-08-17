import sys
import types

from hailrun import wandb_utils


def _install_fake_wandb(monkeypatch, init_fn):
    """Inject a fake `wandb` module into sys.modules so tests don't need the real
    wandb package installed (it's the hailrun[wandb] optional extra)."""
    fake = types.ModuleType("wandb")
    fake.init = init_fn
    monkeypatch.setitem(sys.modules, "wandb", fake)


def test_disabled_returns_none():
    assert wandb_utils.init_hail_wandb(enabled=False, project="p", job_type="j", config={}) is None


def test_init_failure_returns_none(monkeypatch):
    def raise_init(**kwargs):
        raise RuntimeError("boom")

    _install_fake_wandb(monkeypatch, raise_init)
    result = wandb_utils.init_hail_wandb(enabled=True, project="p", job_type="j", config={})
    assert result is None


def test_init_success_merges_config_and_run_name(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return "fake-run"

    _install_fake_wandb(monkeypatch, fake_init)
    monkeypatch.setenv("HAIL_BATCH_ID", "b1")
    monkeypatch.setenv("HAIL_JOB_ID", "j1")
    monkeypatch.setenv("HAIL_ATTEMPT_ID", "a1")
    monkeypatch.setenv("HAIL_JOB_GROUP_ID", "g1")
    monkeypatch.delenv("HAIL_BATCH_NAME", raising=False)

    result = wandb_utils.init_hail_wandb(enabled=True, project="proj", job_type="jt", config={"lr": 0.1})

    assert result == "fake-run"
    assert captured["project"] == "proj"
    assert captured["group"] == "b1"
    assert captured["name"] == "b1-job-j1-a-a1"
    assert captured["config"]["lr"] == 0.1
    assert captured["config"]["hail_batch_id"] == "b1"
    assert captured["config"]["hail_job_group_id"] == "g1"


def test_run_name_uses_batch_name_when_set(monkeypatch):
    captured = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return "fake-run"

    _install_fake_wandb(monkeypatch, fake_init)
    monkeypatch.setenv("HAIL_BATCH_ID", "b1")
    monkeypatch.setenv("HAIL_JOB_ID", "j1")
    monkeypatch.setenv("HAIL_ATTEMPT_ID", "a1")
    monkeypatch.setenv("HAIL_BATCH_NAME", "my-batch")

    wandb_utils.init_hail_wandb(enabled=True, project="proj", job_type="jt", config={})

    assert captured["name"] == "my-batch-job-j1-a-a1"
