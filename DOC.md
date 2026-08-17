# hailrun internals

## `cli.py`
Typer app with two commands: `dispatch` and `version`. `dispatch` is registered
directly on the root app (`app.command("dispatch", context_settings=...)(dispatch.run)`)
rather than as a nested Typer sub-app -- Typer silently collapses a root app with
exactly one command into a bare command, which ate the `dispatch` name entirely when
tried. The `version` command exists partly to keep that collapse from happening again.
`context_settings={"ignore_unknown_options": True, "allow_extra_args": True}` on the
`dispatch` command is what lets script args pass through without a `--` separator.

## `dispatch.py`
`run()` does, in order: validate options -> build the code-context tar (`context.py`,
skipped if `--baked`) -> split the shard input file if `--shard-input` given
(`sharding.py`) -> if `--hail-dry-run`, print everything and return *before* touching
`hailtop.batch` at all -> build `hb.Batch`/`hb.ServiceBackend` -> `b.read_input()` the
context tarball once and each shard file once (Hail auto-uploads local paths here) ->
per-job loop setting image/cpu/memory/`spot()`/env, then two `Job.command()` calls
(untar, then `cd` + run) -- two calls because they share shell state (env, cwd) on the
same job, so a tar failure still shows on its own log line. `hailtop` has no public
setter for machine type; `_set_machine_type()` reaches into `Job._machine_type`
directly and raises loudly if that attribute ever disappears in a future hailtop
release.

## `context.py`
`build_context_tar()` walks the context dir, drops anything matching
`.gcloudignore`/`.dockerignore`/`.gitignore` (first found, in that order) or the
hardcoded `DEFAULT_EXCLUDES`, and writes a deterministic `tar.gz` (sorted entries,
zeroed mtime/uid/gid) to a content-hashed path so repeated dry-runs don't rebuild an
identical tarball. `dispatch.py` also checks for a `src/` subdirectory here and adds it
to the job's `PYTHONPATH` separately, so src-layout packages import correctly even
when the dispatched script lives outside `src/`.

## `sharding.py`
Three format-specific splitters (`split_text`/`split_csv`/`split_parquet`), all
producing N contiguous shard files (shard 0 = first chunk) rather than round-robin, so
a single shard is easy to reason about standalone. `split_csv` uses the stdlib `csv`
module rather than naive line-splitting so multi-line quoted fields survive.
`split_parquet` streams via `iter_batches()` and lazy-imports `pyarrow` (optional
extra). `dispatch.py` substitutes each job's shard path into a `{shard}` token in the
script's own args, or appends `--input <path>` if no token is present.

## `wandb_wrap.py`
Invoked as the job's actual command when `--wandb` is set:
`python -m hailrun.wandb_wrap --project ... -- <real command>`. Calls
`init_hail_wandb()`, then runs the real command via `subprocess.Popen` with piped
stdout, re-printing each line from this (parent) process. That re-print is required --
a plain `subprocess.run()` with inherited stdio does *not* get picked up by wandb's
console capture (verified empirically: no `output.log` was ever uploaded), because
wandb wraps the parent process's own stdout, not an fd a child writes to directly.

## `wandb_utils.py`
`init_hail_wandb()` reads Hail's auto-injected `HAIL_BATCH_ID`/`HAIL_JOB_ID`/
`HAIL_JOB_GROUP_ID`/`HAIL_ATTEMPT_ID` env vars for the run's group/name, plus
`HAIL_BATCH_NAME` -- the one var Hail does *not* inject itself, so `dispatch.py` sets
it explicitly via `Job.env()` before the job runs. Never raises: a failed `wandb.init`
prints a warning and returns `None`, so callers just check `if run is not None`.

## `gcs.py`
Thin wrapper over `google-cloud-storage`, client cached via
`functools.lru_cache(maxsize=1)` on `_get_client()` (one seam to monkeypatch in
tests). Both `google-cloud-storage` and `pandas` are lazy imports, so importing
`hailrun.gcs` itself never requires the `gcs` extra.
