from __future__ import annotations

import hashlib
import json


def hash_store_params(
    gene_space: str = "PROTEIN_CODING",
    zarr_chunk_size: int = 2048,
    zarr_shard_size: int = 65536,
    anndata_shard_size: int = 2**21,
    should_densify: bool = True,  # noqa: FBT001, FBT002
):
    store_params = (
        gene_space,
        zarr_chunk_size,
        zarr_shard_size,
        anndata_shard_size,
        should_densify,
    )
    return hashlib.sha256(json.dumps(store_params).encode()).hexdigest()
