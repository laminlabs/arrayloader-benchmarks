#!/bin/bash

#SBATCH -J hydra
#SBATCH --output=slurm_out/hydra.%j
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos mcml
#SBATCH --gres=gpu:1
#SBATCH --time 1-00:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=6


CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer/enroot-images/arrayloaders.sqsh"
CONTAINER_MOUNTS="/dss:/dss,/dss/dssfs02/lwp-dss-0001/pn36po/pn36po-dss-0001/di93zer:/mnt/dssfs02"


srun --cpu-bind=verbose,socket --accel-bind=g --gres=gpu:1 \
     --container-mounts=$CONTAINER_MOUNTS --container-image=$CONTAINER_IMAGE \
     --no-container-remap-root \
     bash -c "python create_benchmark_store.py --config-name=create_benchmark_store --multirun"
