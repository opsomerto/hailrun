from hailrun import wandb_wrap


def test_job_type_not_required():
    args, command = wandb_wrap.parse_args(["--project", "p", "--", "python", "s.py"])
    assert args.job_type is None
    assert command == ["python", "s.py"]


def test_default_job_type_skips_interpreter():
    assert wandb_wrap._default_job_type(["python", "sub/train_model.py", "--x"]) == "train_model"
    assert wandb_wrap._default_job_type(["./run.sh"]) == "run"


def test_default_job_type_skips_interpreter_flags():
    assert wandb_wrap._default_job_type(["python", "-m", "mymod"]) == "mymod"
