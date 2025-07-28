from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import click
import lamindb as ln
import zarr
import zarrs  # noqa
from arrayloaders.io import create_store_from_h5ads
from zarr.codecs import BloscCodec, BloscShuffle

from arrayloader_benchmarks.create_sqlite_databases import DB_PATH

zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": None}
)

COMPRESSOR = BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle)


@click.command()
@click.option("--store_path", type=str)
@click.option("--gene_space", type=str)
@click.option("--n_shards_input", type=int, default=10)
@click.option("--zarr_chunk_size", type=int, default=2048)
@click.option("--zarr_shard_size", type=int, default=65536)
@click.option("--anndata_shard_size", type=int, default=2**21)
@click.option("--should_densify", type=bool, default=True)
def create_store(  # noqa: PLR0917
    store_path: str,
    gene_space: str = "PROTEIN_CODING",
    n_shards_input: int = 48,
    zarr_chunk_size: int = 2048,
    zarr_shard_size: int = 65536,
    anndata_shard_size: int = 2**21,
    should_densify: bool = True,  # noqa: FBT001, FBT002
):
    store_path = Path(store_path)
    if gene_space == "FULL":
        h5ads = ln.Collection.get("eAgoduHMxuDs5Wem0000")
    elif gene_space == "PROTEIN_CODING":
        h5ads = ln.Collection.get("k9GqakN96EmLjn1L0000")
    else:
        err_msg = (
            f"Invalid gene_space: {gene_space}. Must be 'FULL' or 'PROTEIN_CODING'."
        )
        raise ValueError(err_msg)
    h5ad_cache_paths = h5ads.cache()[0].parent

    store_type = "DENSE" if should_densify else "SPARSE"
    store_hash = str(
        hash(
            (
                store_type,
                gene_space,
                zarr_chunk_size,
                zarr_shard_size,
                anndata_shard_size,
            )
        )
    )

    start_time = time.time()
    create_store_from_h5ads(
        [h5ad_cache_paths / f"shard_{i}.h5ad" for i in range(n_shards_input)],
        store_path / store_hash,
        chunk_size=zarr_chunk_size,
        shard_size=zarr_shard_size,
        buffer_size=anndata_shard_size,
        zarr_compressor=(COMPRESSOR,),
        shuffle=False,
        should_denseify=should_densify,
    )
    creation_time = time.time() - start_time

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO benchmarks (
            store_path,
            store_type,
            gene_space,
            zarr_chunk_size,
            zarr_shard_size,
            anndata_shard_size,
            creation_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(store_path / store_hash),
            store_type,
            gene_space,
            zarr_chunk_size,
            zarr_shard_size,
            anndata_shard_size,
            creation_time,
        ),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # ln.track(project="DataLoader v2")
    create_store()
    # ln.finish()
