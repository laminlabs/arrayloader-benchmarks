from __future__ import annotations

from pathlib import Path

import anndata as ad
import lamindb as ln
import pandas as pd
from arrayloaders import create_store_from_h5ads

# Set paths where to store the output h5ad files below
OUT_PATH = Path("/mnt/dssfs02/tahoe100M")
OUT_PATH_SUBSET = Path("/mnt/dssfs02/tahoe100M_protein_coding")


if __name__ == "__main__":
    ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"
    ln.track(project="zjQ6EYzMXif4")
    artifact_uids = [
        "aJIqo7bNyJAs9z0r0000",
        "ZFeVfd0ugAHeWCxm0000",
        "XVSrkq9pyF1OBLgG0000",
        "tKTeff0ugWqAm4P70000",
        "EZATJLC4jE7pmwo40000",
        "aAHQ3zbD7n1asyYr0000",
        "DC5cacdJr1VoEXnl0000",
        "czC19UpUEszVH2bU0000",
        "BDttiuV3Te8VB0dU0000",
        "56uA9lPPmJ4zLUcr0000",
        "omn7JStfJMzy8m6O0000",
        "S2h2rPLCaUhZAM9u0000",
        "9L9HZ55HqUL0aqaR0000",
        "vn5cUJCHbjpPPsZx0000",
    ]
    benchmarking_artifacts = ln.Artifact.using("laminlabs/arrayloader-benchmarks")
    file_paths = [benchmarking_artifacts.get(uid).cache() for uid in artifact_uids]

    # Create shuffled and sharded h5ad files --- FULL GENE SPACE
    print("Creating h5ads with full gene space...")
    try:
        OUT_PATH.mkdir(parents=True)
        create_store_from_h5ads(
            file_paths,
            OUT_PATH,
            chunk_size=2048,
            buffer_size=2**21,
            output_format="h5ad",
            should_denseify=False,
        )
    except FileExistsError:
        print("Preprocessed h5ads already exist, skipping creation...")

    artifacts = [
        ln.Artifact.from_anndata(
            shard, key=f"dataloader_v2/tahoe100M_sharded/{shard.name}"
        ).save()
        for shard in OUT_PATH.iterdir()
        if shard.name.endswith(".h5ad")
    ]
    collection = ln.Collection(
        artifacts,
        key="Tahoe100M_sharded",
        description="Shuffled and shared version of the Tahoe100M dataset",
    )
    collection.save()

    # Create shards subset only to protein coding genes --- ONLY PROTEIN CODING GENES
    print("Creating h5ads with subset to protein coding genes...")
    try:
        OUT_PATH_SUBSET.mkdir(parents=True)
        var_subset = pd.read_parquet("protein_coding_genes.parquet").index.tolist()
        for shard in OUT_PATH.iterdir():
            if shard.name.endswith(".h5ad"):
                shard_number = shard.name.split("_")[-1].removesuffix(".h5ad")
                adata_subset = ad.read_h5ad(shard)[:, var_subset].copy()
                assert len(var_subset) == adata_subset.shape[1]
                adata_subset.write(
                    OUT_PATH_SUBSET / f"shard_{shard_number}.h5ad", compression="gzip"
                )
    except FileExistsError:
        print("Subset h5ads already exist, skipping creation...")

    artifacts = [
        ln.Artifact.from_anndata(
            shard,
            key=f"dataloader_v2/tahoe_100M_protein_coding_sharded/protein_coding_{shard.name}",
        ).save()
        for shard in OUT_PATH_SUBSET.iterdir()
        if shard.name.endswith(".h5ad")
    ]
    collection = ln.Collection(
        artifacts,
        key="Tahoe100M_sharded_protein_coding",
        description="Shuffled and shared version of the Tahoe100M dataset subset to protein coding genes",
    )
    collection.save()
    ln.finish()
