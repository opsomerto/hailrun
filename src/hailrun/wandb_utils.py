"""Minimal wandb wiring for Hail Batch jobs.

Reads the env vars Hail Batch injects into every job container automatically
(HAIL_BATCH_ID, HAIL_JOB_ID, HAIL_JOB_GROUP_ID, HAIL_ATTEMPT_ID) to build the group
and run name, plus HAIL_BATCH_NAME (not auto-injected by Hail -- set explicitly by
dispatch/wandb_wrap), and layers the caller's own project/job_type/config on top.
"""

import os


def init_hail_wandb(enabled: bool, project: str, job_type: str, config: dict):
    """wandb.init(), or None if disabled/failed -- callers never need try/except.

    group = HAIL_BATCH_ID. run name = "{batch_name}-job-{HAIL_JOB_ID}-a-{HAIL_ATTEMPT_ID}",
    so retried attempts never collide on the same run.
    """
    if not enabled:
        return None
    import wandb

    batch_id = os.environ.get("HAIL_BATCH_ID")
    job_id = os.environ.get("HAIL_JOB_ID")
    attempt_id = os.environ.get("HAIL_ATTEMPT_ID")
    batch_name = os.environ.get("HAIL_BATCH_NAME")
    group = batch_id
    base_name = batch_name or batch_id
    run_name = f"{base_name}-job-{job_id}-a-{attempt_id}"
    hail_config = {
        "hail_batch_id": batch_id,
        "hail_job_id": job_id,
        "hail_job_group_id": os.environ.get("HAIL_JOB_GROUP_ID"),
        "hail_attempt_id": attempt_id,
        "batch_name": batch_name,
    }
    try:
        print(
            f"  * Initializing wandb run: project={project}, group={group}, name={run_name}, job_type={job_type}"
        )
        return wandb.init(
            project=project,
            group=group,
            job_type=job_type,
            name=run_name,
            config={**hail_config, **config},
        )
    except Exception as e:
        print(f"  ! wandb init failed, continuing without it: {e}")
        return None
