from __future__ import annotations

import os
from itertools import product
from pathlib import Path

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J zarr_create
#SBATCH --output=slurm_out/zarr_create.%j
#SBATCH --error=slurm_out/zarr_create.%j
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 2-00:00:00
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=12


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/arrayloaders.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"

SCRIPT="/dss/dsshome1/04/di93zer/git/arrayloader-benchmarks/src/arrayloader_benchmarks/create_benchmark_store.py"

SCRIPT_ARGS="--store_path={store_path} "
SCRIPT_ARGS+="--gene_space={gene_space} "
SCRIPT_ARGS+="--n_shards_input={n_shards_input} "
SCRIPT_ARGS+="--zarr_chunk_size={zarr_chunk_size} "
SCRIPT_ARGS+="--zarr_shard_size={zarr_shard_size} "
SCRIPT_ARGS+="--anndata_shard_size={anndata_shard_size} "
SCRIPT_ARGS+="--should_densify={should_densify} "


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "pip install -e /dss/dsshome1/04/di93zer/git/arrayloaders && python -u ${{SCRIPT}} ${{SCRIPT_ARGS}}"
"""


def schedule_jobs(save_path, config_dict):
    for (
        should_densify,
        n_shards_input,
        gene_space,
        zarr_chunk_size,
        zarr_shard_size,
        anndata_shard_size,
    ) in product(
        config_dict["should_densify"],
        config_dict["n_shards_input"],
        config_dict["gene_space"],
        config_dict["zarr_chunk_size"],
        config_dict["zarr_shard_size"],
        config_dict["anndata_shard_size"],
    ):
        job_script = JOB_SCRIPT.format(
            store_path=save_path,
            gene_space=gene_space,
            n_shards_input=n_shards_input,
            zarr_chunk_size=zarr_chunk_size,
            zarr_shard_size=zarr_shard_size,
            anndata_shard_size=anndata_shard_size,
            should_densify=should_densify,
        )
        with Path("job_script.sbatch").open("w") as f:
            f.write(job_script)
        os.system("sbatch job_script.sbatch && sleep 0.5 && rm job_script.sbatch")


if __name__ == "__main__":
    STORE_PATH = Path("/dss/mcmlscratch/04/di93zer/arrayloader_benchmarks")
    BENCHMARK_CONFIG_DENSE = {
        "should_densify": [True],
        "n_shards_input": [10],
        "gene_space": ["PROTEIN_CODING"],
        "zarr_chunk_size": [1024, 256, 128],
        "zarr_shard_size": [65536],
        "anndata_shard_size": [2_097_152],
    }
    BENCHMARK_CONFIG_SPARSE = {
        "should_densify": [False],
        "n_shards_input": [48],
        "gene_space": ["PROTEIN_CODING"],
        "zarr_chunk_size": [32768],
        "zarr_shard_size": [134_217_728],
        "anndata_shard_size": [2_097_152],
    }
    print("Creating SPARSE benchmark stores...")
    schedule_jobs(STORE_PATH, BENCHMARK_CONFIG_SPARSE)
    # print("Creating DENSE benchmark stores...")
    # schedule_jobs(STORE_PATH, BENCHMARK_CONFIG_DENSE)
