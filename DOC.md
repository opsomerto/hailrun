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
skipped if `--baked`) -> if `--shard-input` given, branch on `shard_input.is_dir()`:
a file gets split via `sharding.split_file` (`--hail-shards` required, N > 1); a
directory is listed as-is via `sharding.list_shard_dir` (`--hail-shards` optional,
inferred from file count, validated against it if given) with no splitting -> if
`--hail-dry-run`, print everything and return *before* touching `hailtop.batch` at all
-> build `hb.Batch`/`hb.ServiceBackend` -> `b.read_input()` the context tarball once
and each shard file once (Hail auto-uploads local paths here) -> per-job loop setting
image/cpu/memory/`spot()`/env, then two `Job.command()` calls (untar, then `cd` + run)
-- two calls because they share shell state (env, cwd) on the same job, so a tar
failure still shows on its own log line. `hailtop` has no public setter for machine
type; `_set_machine_type()` reaches into `Job._machine_type` directly and raises
loudly if that attribute ever disappears in a future hailtop release.

`_job_tokens()` injects each job's shard value into the script's own args by
three-tier precedence: a literal `{shard}` token wins if present; else `--shard-flag
NAME` (dashes auto-normalized) finds/replaces/appends that named flag (in-place
replace, `NAME=value` inline replace, or append if absent); else the value is
prepended as a leading positional arg. `--shard-flag` and a literal `{shard}` token
together is a guard-clause error (ambiguous). `--shard-flag` without `--shard-input`
is also a guard-clause error.

If `--verify-args`, right after `_job_tokens`/`_run_cmd` are defined (so shard values
are substituted the same way a real job would see them) and before the `--hail-dry-run`
branch: skipped for `--baked` (no local file), else delegates to `verify.py`.

## `verify.py` / `_verify_runner.py`
`verify.py:detect_framework()` statically AST-scans the script's top-level imports for
`argparse`/`click`/`typer` -- no execution. If none found, verification is skipped.
Otherwise `verify_script_args()` shells out to `_verify_runner.py` in a subprocess
(`NO_COLOR`/`TERM=dumb` env to keep click/typer output plain), passing the framework,
script path, an extra-`sys.path` string (context dir + src-layout dir, mirroring the
real job's `PYTHONPATH`), and the script args.

`_verify_runner.py` monkeypatches the parsing entrypoint to abort right after parsing
completes, so the script's real logic never runs -- only import + arg parsing does:
`argparse.ArgumentParser.parse_args` is wrapped to call the original then raise a
sentinel; `click.Command.invoke` (also covers typer, which builds click Commands under
the hood, and `Group`/`MultiCommand` subcommand dispatch eventually calls the same
`Command.invoke`) is replaced to raise the sentinel instead of running the real command
body. The sentinel means "parsing succeeded" -> `HAILRUN_VERIFY_OK`. A `SystemExit`
with nonzero code from the framework's own error path (argparse always exits 2 on a
parse error; click/typer usage errors also exit nonzero) propagates as that exit code,
read back as `bad_args`. Exiting 0 *without* the sentinel having fired, or running to
completion without ever calling the patched method, means the script never reached its
real parser (early exit, a `no_args_is_help` branch, conditional parsing, etc.) --
reported as `HAILRUN_VERIFY_UNKNOWN`/`inconclusive`, not `ok`, since exit-0 alone isn't
proof the args are valid. Any other failure (missing dependency, unrelated exception,
subprocess timeout) is also `inconclusive` -- the check degrades to "couldn't verify",
never blocks submission on a problem with the check itself, only on an actual parse
error caught red-handed.

Known limitation: only the common single top-level `parser.parse_args()` call is
patched for argparse, not a bare `parse_known_args()` used on its own without a
wrapping `parse_args()` call.

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
extra). `list_shard_dir()` is the non-splitting counterpart for directory-mode
sharding: sorted, non-recursive, skips subdirectories and dotfiles, raises
`ValueError` if nothing eligible is found. See `dispatch.py` for how shard values get
injected into the script's own args (`{shard}` token / `--shard-flag` / default
leading positional).

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
