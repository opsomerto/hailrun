"""`hailrun dispatch` -- run a local script as one or more Hail Batch jobs.

Usage: hailrun dispatch [options] my_script.py [script args...]

No `--` separator is required: hailrun's own options are declared flags, `script` is
a dedicated positional argument, and everything after it is captured as opaque
passthrough (Click's ignore_unknown_options/allow_extra_args) regardless of whether it
looks like a flag -- as long as hailrun's own options come before the script path. An
explicit literal `--` still works too (Click's standard "stop parsing options" marker),
useful if a script argument happens to collide with a hailrun option name.

Pass --verify-args for a best-effort local check that the script's own argparse/click/
typer parser accepts the given script args before submitting -- see hailrun.verify.
"""

import os
import shlex
import tempfile
from datetime import datetime
from pathlib import Path

import typer

from hailrun import context as context_mod
from hailrun import sharding

CONTEXT_MOUNT = "/tmp/hailrun-context"


def _set_machine_type(j, machine_type: str) -> None:
    """Job._machine_type has no public setter as of hailtop 0.2.139 -- reach in
    directly, but fail loudly if a future hailtop release renames/removes it."""
    if not hasattr(j, "_machine_type"):
        raise RuntimeError(
            "hailtop.batch.Job has no _machine_type attribute; the hailtop API may "
            "have changed -- update hailrun.dispatch._set_machine_type"
        )
    j._machine_type = machine_type


def run(
    ctx: typer.Context,
    script: str = typer.Argument(..., help="Local script to run as a Hail Batch job."),
    hail_image: str = typer.Option(
        ...,
        "--hail-image",
        envvar="HAIL_IMAGE",
        help="Docker image with dependencies installed (code is synced separately unless --baked).",
    ),
    hail_machine_type: str | None = typer.Option(None, "--hail-machine-type", envvar="HAIL_MACHINE_TYPE"),
    hail_preemptible: bool = typer.Option(True, "--hail-preemptible/--hail-no-preemptible"),
    hail_storage_gb: int = typer.Option(50, "--hail-storage-gb"),
    hail_cpu: str | None = typer.Option(None, "--hail-cpu"),
    hail_memory: str | None = typer.Option(None, "--hail-memory"),
    hail_shards: int = typer.Option(0, "--hail-shards", help="Number of shards; requires --shard-input."),
    hail_billing_project: str = typer.Option(..., "--hail-billing-project", envvar="HAIL_BILLING_PROJECT"),
    hail_tmp_bucket: str = typer.Option(..., "--hail-tmp-bucket", envvar="TMP_BUCKET"),
    hail_region: str = typer.Option("us-central1", "--hail-region", envvar="GCP_REGION"),
    hail_job_name: str | None = typer.Option(None, "--hail-job-name"),
    hail_env: list[str] = typer.Option([], "--hail-env", help="KEY=VAL, repeatable, kept off the command line."),
    hail_wait: bool = typer.Option(False, "--hail-wait/--hail-no-wait"),
    hail_dry_run: bool = typer.Option(False, "--hail-dry-run"),
    verify_args: bool = typer.Option(
        False,
        "--verify-args",
        help=(
            "Best-effort local check that the script's own argparse/click/typer parser "
            "accepts the script args before submitting (skipped for --baked scripts or "
            "if no argparse/click/typer import is detected in the script)."
        ),
    ),
    context_dir: Path | None = typer.Option(None, "--context-dir", help="Defaults to the script's own directory."),
    context_exclude: list[str] = typer.Option([], "--context-exclude", help="Extra exclude pattern, repeatable."),
    baked: bool = typer.Option(
        False, "--baked/--no-context-sync", help="Skip code sync; assume script already exists in the image."
    ),
    baked_path: str = typer.Option("/app", "--baked-path", help="Only used with --baked."),
    shard_input: Path | None = typer.Option(
        None,
        "--shard-input",
        help="File to split into --hail-shards shards, or a directory of pre-made shard files to use as-is.",
    ),
    shard_format: str = typer.Option("auto", "--shard-format", help="text, csv, parquet, or auto."),
    shard_flag: str | None = typer.Option(
        None,
        "--shard-flag",
        help=(
            "Flag in the script's own args to receive the shard value, e.g. --arg-a "
            "(leading dashes optional, normalized automatically). If omitted, the shard "
            "value is prepended as a leading positional arg. Ignored in favor of a literal "
            "{shard} token if one is present in the script args -- passing both is an error."
        ),
    ),
    wandb: bool = typer.Option(False, "--wandb/--no-wandb", help="Auto-wrap the job command in a wandb run."),
    wandb_project: str | None = typer.Option(None, "--wandb-project", envvar="WANDB_PROJECT"),
    wandb_job_type: str | None = typer.Option(None, "--wandb-job-type", envvar="WANDB_JOB_TYPE"),
) -> None:
    script_args: list[str] = list(ctx.args)

    if shard_flag is not None and not shard_flag.startswith("-"):
        shard_flag = f"--{shard_flag}"

    shard_mode = None  # "file" | "dir" | None
    dir_shard_files: list[Path] = []
    if shard_input is not None:
        if not shard_input.exists():
            typer.echo(f"Error: --shard-input path not found: {shard_input}", err=True)
            raise typer.Exit(1)
        shard_mode = "dir" if shard_input.is_dir() else "file"

    if shard_flag is not None and shard_input is None:
        typer.echo("Error: --shard-flag requires --shard-input", err=True)
        raise typer.Exit(1)
    if shard_flag is not None and "{shard}" in script_args:
        typer.echo(
            "Error: pass either --shard-flag or a literal {shard} token in the script args, "
            "not both (ambiguous which one should receive the shard value)",
            err=True,
        )
        raise typer.Exit(1)

    if shard_mode == "file":
        if hail_shards <= 1:
            typer.echo("Error: --shard-input FILE requires --hail-shards N with N > 1", err=True)
            raise typer.Exit(1)
    elif shard_mode == "dir":
        if shard_format != "auto":
            print(
                f"  ! --shard-format {shard_format!r} is ignored in directory mode "
                "(files are used as-is, not split)"
            )
        try:
            dir_shard_files = sharding.list_shard_dir(shard_input)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)
        n_dir_files = len(dir_shard_files)
        if hail_shards not in (0, n_dir_files):
            typer.echo(
                f"Error: --hail-shards {hail_shards} does not match the {n_dir_files} file(s) "
                f"found in --shard-input directory {shard_input} "
                "(omit --hail-shards to infer it automatically)",
                err=True,
            )
            raise typer.Exit(1)
    elif hail_shards > 1:
        typer.echo(
            "Error: --hail-shards > 1 requires --shard-input PATH "
            "(the target script no longer needs to slice input itself -- pass the file to split).",
            err=True,
        )
        raise typer.Exit(1)

    if not hail_cpu and not hail_memory and not hail_machine_type:
        hail_machine_type = "n1-standard-4"

    if wandb and (not wandb_project or not wandb_job_type):
        typer.echo("Error: --wandb requires --wandb-project and --wandb-job-type", err=True)
        raise typer.Exit(1)
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb and not wandb_api_key:
        typer.echo("Error: --wandb requires WANDB_API_KEY in the submitting environment", err=True)
        raise typer.Exit(1)

    script_rel = None
    tar_info = None
    has_src_layout = False
    if not baked:
        script_path = Path(script)
        if not script_path.exists():
            typer.echo(f"Error: script not found: {script_path}", err=True)
            raise typer.Exit(1)
        ctx_dir = context_dir or script_path.resolve().parent
        script_rel = context_mod.resolve_script_rel(script_path, ctx_dir)
        tar_info = context_mod.build_context_tar(ctx_dir, context_exclude)
        # src-layout convention (package under context_dir/src/, e.g. a script outside
        # src/ that still needs to `import mypackage`): put context_dir/src on
        # PYTHONPATH too, not just the context root itself.
        has_src_layout = (ctx_dir / "src").is_dir()

    n_jobs = 1
    shard_paths: list[Path] = []
    if shard_mode == "file":
        fmt = shard_format if shard_format != "auto" else sharding.detect_format(shard_input)
        tmp_shard_dir = Path(tempfile.mkdtemp(prefix="hailrun-shards-"))
        shard_paths = sharding.split_file(shard_input, hail_shards, fmt, tmp_shard_dir)
        n_jobs = hail_shards
    elif shard_mode == "dir":
        shard_paths = dir_shard_files
        n_jobs = len(shard_paths)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = hail_job_name or f"{Path(script).stem}-{stamp}"
    env_vars = dict(kv.split("=", 1) for kv in hail_env)

    def _job_tokens(i: int, shard_values: list[str]) -> list[str]:
        """shard_values[i] is either a local-path string (dry run) or a hailtop
        ResourceFile's interpolation token (real submission) -- injected into the
        script's own args by three-tier precedence: a literal `{shard}` token wins
        if present; else --shard-flag NAME finds/replaces/appends that named flag;
        else the value is prepended as a leading positional arg."""
        tokens = list(script_args)
        if not shard_values:
            return tokens
        value = shard_values[i]

        if "{shard}" in tokens:
            tokens[tokens.index("{shard}")] = value
            return tokens

        if shard_flag is not None:
            if shard_flag in tokens:
                idx = tokens.index(shard_flag)
                if idx == len(tokens) - 1:
                    tokens.append(value)
                else:
                    tokens[idx + 1] = value
                return tokens
            inline_idx = next((j for j, t in enumerate(tokens) if t.startswith(f"{shard_flag}=")), None)
            if inline_idx is not None:
                tokens[inline_idx] = f"{shard_flag}={value}"
                return tokens
            return tokens + [shard_flag, value]

        return [value] + tokens

    def _run_cmd(i: int, shard_values: list[str]) -> str:
        target = f"{baked_path.rstrip('/')}/{script}" if baked else script_rel
        tokens = ["python", target] + _job_tokens(i, shard_values)
        if wandb:
            tokens = [
                "python", "-m", "hailrun.wandb_wrap",
                "--project", wandb_project,
                "--job-type", wandb_job_type,
                "--hail-batch-name", job_name,
                "--",
            ] + tokens
        return shlex.join(tokens)

    if verify_args:
        if baked:
            typer.echo("  ! --verify-args: skipped (--baked scripts have no local file to check)")
        else:
            from hailrun import verify as verify_mod

            verify_shard_values = [str(p) for p in shard_paths] if shard_paths else []
            verify_tokens = _job_tokens(0, verify_shard_values)
            verify_pythonpath = [str(ctx_dir)]
            if has_src_layout:
                verify_pythonpath.append(str(ctx_dir / "src"))
            result = verify_mod.verify_script_args(Path(script), verify_tokens, verify_pythonpath)
            if result.status == "bad_args":
                typer.echo(f"Error: script rejected args during local --verify-args check:\n{result.message}", err=True)
                raise typer.Exit(1)
            elif result.status == "skipped":
                typer.echo(f"  ! --verify-args: {result.message}, skipping check")
            elif result.status == "inconclusive":
                typer.echo(f"  ! --verify-args: {result.message}, proceeding without check")

    if hail_dry_run:
        dry_run_shard_values = [str(p) for p in shard_paths]
        typer.echo(f"image: {hail_image}")
        typer.echo(f"machine: cpu={hail_cpu} memory={hail_memory} machine_type={hail_machine_type} preemptible={hail_preemptible}")
        typer.echo(f"storage: {hail_storage_gb}Gi  region: {hail_region}")
        if baked:
            typer.echo(f"baked path: {baked_path.rstrip('/')}/{script}")
        else:
            typer.echo(
                f"context: {tar_info.file_count} files, {tar_info.total_bytes} bytes, "
                f"sha256={tar_info.sha256[:16]}, script_rel={script_rel}, "
                f"src_layout_detected={has_src_layout}"
            )
        if shard_paths:
            typer.echo(f"shards: {dry_run_shard_values}")
        typer.echo(f"Submitting {n_jobs} job(s)...")
        for i in range(n_jobs):
            typer.echo(f"  job {i:04d}: {_run_cmd(i, dry_run_shard_values)}")
        return

    import hailtop.batch as hb

    b = hb.Batch(
        name=job_name,
        backend=hb.ServiceBackend(
            billing_project=hail_billing_project,
            remote_tmpdir=f"{hail_tmp_bucket.rstrip('/')}/hail-tmp/",
        ),
        default_regions=[hail_region],
    )

    context_resource = None if baked else b.read_input(str(tar_info.path))
    shard_resources = [b.read_input(str(p)) for p in shard_paths] if shard_paths else []
    shard_values = [str(r) for r in shard_resources]

    for i in range(n_jobs):
        j = b.new_job(name=f"{Path(script).stem}-{i:04d}")
        j.image(hail_image)
        j.storage(f"{hail_storage_gb}Gi")
        if hail_cpu:
            j.cpu(hail_cpu)
        if hail_memory:
            j.memory(hail_memory)
        j.spot(hail_preemptible)
        if hail_machine_type:
            _set_machine_type(j, hail_machine_type)
        for k, v in env_vars.items():
            j.env(k, v)

        if wandb:
            j.env("WANDB_API_KEY", wandb_api_key)
            j.env("HAIL_BATCH_NAME", job_name)

        run_cmd = _run_cmd(i, shard_values)
        if baked:
            j.command(run_cmd)
        else:
            pythonpath = f"{CONTEXT_MOUNT}/src:{CONTEXT_MOUNT}" if has_src_layout else CONTEXT_MOUNT
            j.command(f"mkdir -p {CONTEXT_MOUNT} && tar -xzf {context_resource} -C {CONTEXT_MOUNT}")
            j.command(f"cd {CONTEXT_MOUNT} && PYTHONPATH={pythonpath}:$PYTHONPATH {run_cmd}")

    typer.echo(f"Submitting {n_jobs} job(s)...")
    b.run(wait=hail_wait, open=True)
    typer.echo("Done.")
