from __future__ import annotations

import os
from itertools import product
from pathlib import Path

JOB_SCRIPT = r"""#!/bin/bash

#SBATCH -J zarr_benchmark
#SBATCH --output=slurm_out/zarr_benchmark.%j
#SBATCH --error=slurm_out/zarr_benchmark.%j
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 0-03:00:00
#SBATCH --mem=128GB
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
SCRIPT_ARGS+="--chunk_size={chunk_size} "
SCRIPT_ARGS+="--preload_nchunks={preload_nchunks} "
SCRIPT_ARGS+="--use_torch_loader={use_torch_loader} "
SCRIPT_ARGS+="--num_workers={num_workers} "
SCRIPT_ARGS+="--batch_size={batch_size} "
SCRIPT_ARGS+="--n_samples={n_samples} "


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "python -u ${{SCRIPT}} ${{SCRIPT_ARGS}}"
"""


def schedule_jobs(save_path, config_dict):
    for (
        should_densify,
        n_shards_input,
        gene_space,
        zarr_chunk_size,
        zarr_shard_size,
        anndata_shard_size,
        chunk_size,
        buffer_size,
        use_torch_loader,
        num_workers,
        batch_size,
        n_samples,
    ) in product(
        config_dict["should_densify"],
        config_dict["n_shards_input"],
        config_dict["gene_space"],
        config_dict["zarr_chunk_size"],
        config_dict["zarr_shard_size"],
        config_dict["anndata_shard_size"],
        config_dict["chunk_size"],
        config_dict["buffer_size"],
        config_dict["use_torch_loader"],
        config_dict["num_workers"],
        config_dict["batch_size"],
        config_dict["n_samples"],
    ):
        job_script = JOB_SCRIPT.format(
            store_path=save_path,
            gene_space=gene_space,
            n_shards_input=n_shards_input,
            zarr_chunk_size=zarr_chunk_size,
            zarr_shard_size=zarr_shard_size,
            anndata_shard_size=anndata_shard_size,
            should_densify=should_densify,
            chunk_size=chunk_size,
            preload_nchunks=buffer_size // chunk_size,
            use_torch_loader=use_torch_loader,
            num_workers=num_workers,
            batch_size=batch_size,
            n_samples=n_samples,
        )
        with Path("job_script.sbatch").open("w") as f:
            f.write(job_script)
        os.system("sbatch job_script.sbatch && sleep 0.5 && rm job_script.sbatch")


if __name__ == "__main__":
    BASE_PATH = Path("/dss/mcmlscratch/04/di93zer")
    GENE_SPACE = "PROTEIN_CODING"
    N_SAMPLES = 2_000_000
    BATCH_SIZE = 4096
    NUM_WORKERS = 4

    BENCHMARK_CONFIG_DENSE = {
        "should_densify": [True],
        "n_shards_input": [10],
        "gene_space": ["PROTEIN_CODING"],
        "zarr_chunk_size": [128, 256, 1024],
        "zarr_shard_size": [65536],
        "anndata_shard_size": [2_097_152],
        "chunk_size": [1, 16, 64, 128, 512, 1024],
        "buffer_size": [16384],
        "use_torch_loader": [True, False],
        "num_workers": [NUM_WORKERS],
        "batch_size": [BATCH_SIZE],
        "n_samples": [N_SAMPLES],
    }
    BENCHMARK_CONFIG_SPARSE = {
        "should_densify": [False],
        "n_shards_input": [10],
        "gene_space": ["PROTEIN_CODING"],
        "zarr_chunk_size": [1024, 2048, 32768, 1_048_576],
        "zarr_shard_size": [134_217_728],
        "anndata_shard_size": [2_097_152, 10_485_760],
        "chunk_size": [1, 16, 64, 128, 512, 1024],
        "buffer_size": [16384],
        "use_torch_loader": [True, False],
        "num_workers": [NUM_WORKERS],
        "batch_size": [BATCH_SIZE],
        "n_samples": [N_SAMPLES],
    }

    print("Creating DENSE benchmark stores...")
    schedule_jobs(BASE_PATH, BENCHMARK_CONFIG_DENSE)
    print("Creating SPARSE benchmark stores...")
    schedule_jobs(BASE_PATH, BENCHMARK_CONFIG_SPARSE)
