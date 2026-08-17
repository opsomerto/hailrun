# hailrun

Run a local script as one or more Hail Batch jobs, with code-context sync, file
sharding, and optional wandb monitoring -- without changing the script itself.

## Install

```bash
uv sync                      # base install
uv sync --extra wandb        # + wandb auto-wrap / init_hail_wandb
uv sync --extra parquet      # + parquet sharding
uv sync --extra gcs          # + hailrun.gcs helpers
uv sync --extra all          # everything
```

## CLI: `hailrun dispatch`

```bash
hailrun dispatch \
  --hail-image us-central1-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG \
  --hail-billing-project MY_BILLING_PROJECT \
  --hail-tmp-bucket gs://my-bucket/tmp \
  my_script.py --some-script-flag value
```

No `--` separator is required between hailrun's own options and the script's own
args -- put hailrun's flags before the script path, and everything after the script
path is forwarded byte-for-byte. An explicit `--` still works if a script argument
happens to collide with a hailrun option name.

### Code delivery: the image only needs dependencies

By default `hailrun` doesn't require your code to be baked into the Docker image at
all -- only the dependencies. At submit time it tars up the local directory containing
the script (or `--context-dir`, if given) and syncs it into the job, like `gcloud
builds submit` sending build context. No image rebuild/push needed after a code
change.

- The script's own directory is the sync root by default -- pass `--context-dir` to
  sync a wider tree (e.g. a repo root) if the script needs to import sibling packages
  outside its own folder.
- If the sync root contains a `src/` subdirectory (the common src-layout convention),
  it's added to `PYTHONPATH` too, so `import mypackage` works even when the script
  itself lives outside `src/` (e.g. a `playground/` script importing a package at
  `src/mypackage/`).
- An existing `.gcloudignore`/`.dockerignore`/`.gitignore` in the sync root is honored
  automatically (falls back to a hardcoded default-exclude list otherwise); add more
  with repeatable `--context-exclude PATTERN`.
- Use `--baked` if the script is already baked into the image instead (assumed at
  `--baked-path`, default `/app`) -- skips the sync step entirely.

### Sharding a big input file

```bash
hailrun dispatch --hail-shards 8 --shard-input big_file.csv \
  --hail-image ... --hail-billing-project ... --hail-tmp-bucket ... \
  my_script.py --name foo
```

`hailrun` splits `big_file.csv` into 8 shard files once at submit time (text, csv,
or parquet -- auto-detected from the extension, or set `--shard-format`), then runs
8 jobs. By default each job gets its shard path as a **leading positional arg** --
`my_script.py` above receives `python my_script.py <shard_path> --name foo` -- so the
target script only needs to accept a plain positional argument, no hailrun-specific
flag name required.

Two ways to control exactly where the shard value lands in the script's own args:

- `--shard-flag NAME` (e.g. `--shard-flag --arg-a`, dashes optional) -- hailrun finds
  `--arg-a` in the script's args and fills in its value, or appends `--arg-a
  <shard_path>` if the flag isn't already there:
  ```bash
  hailrun dispatch --hail-shards 8 --shard-input big_file.csv --shard-flag --arg-a \
    --hail-image ... --hail-billing-project ... --hail-tmp-bucket ... \
    my_script.py --arg-a placeholder
  ```
- A literal `{shard}` token anywhere in the script's args -- hailrun substitutes that
  exact token in place. Useful for positioning the value somewhere other than the
  front, or inside a flag's value without going through `--shard-flag`:
  ```bash
  my_script.py --config "input={shard}"
  ```
  `--shard-flag` and a literal `{shard}` token can't be combined -- pick one.

### Running on a directory of pre-made shards

If the shards already exist (split upstream, or hand-curated), point `--shard-input`
at the directory instead of a file -- no splitting happens, hailrun just runs one job
per file:

```bash
hailrun dispatch --shard-input pre_split_shards/ \
  --hail-image ... --hail-billing-project ... --hail-tmp-bucket ... \
  my_script.py --name foo
```

`--hail-shards` is optional here -- it's inferred from the file count. Pass it anyway
if you want a sanity check (it must match the actual file count, or hailrun errors).
`--shard-format` has no effect in this mode since nothing gets split.

### Breaking change (0.2.0)

Before 0.2.0, a script that didn't use `{shard}` received its shard via an implicit
`--input <path>` append. That fallback is gone -- the new default is a leading
positional arg instead. Scripts written against the old default need
`--shard-flag --input` added to their `hailrun dispatch` invocation to keep working
unchanged.

### wandb monitoring

```bash
hailrun dispatch --wandb --wandb-project my-project --wandb-job-type my-job \
  --hail-image ... --hail-billing-project ... --hail-tmp-bucket ... \
  my_script.py
```

Requires `WANDB_API_KEY` in the submitting environment (forwarded to the job, never
on the command line) and the image to have `hailrun[wandb]` installed. The script
itself needs zero wandb-specific code -- hailrun wraps the job command in a wandb run
tied to the job's Hail batch/job/attempt IDs.

## Library use

```python
from hailrun import init_hail_wandb

run = init_hail_wandb(enabled=True, project="my-project", job_type="training", config={...})
```

```python
from hailrun.gcs import gcs_upload, gcs_download, read_table, upload_df_to_gcs
```
