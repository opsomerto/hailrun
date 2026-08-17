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
  my_script.py --input {shard}
```

`hailrun` splits `big_file.csv` into 8 shard files once at submit time (text, csv,
or parquet -- auto-detected from the extension, or set `--shard-format`), and
substitutes each job's shard path into the `{shard}` placeholder in the script's own
args (or appends `--input <shard_path>` if no placeholder is present). The target
script never needs to implement its own slicing logic.

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
