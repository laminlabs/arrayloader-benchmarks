from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import lamindb as ln
import zarr
import zarrs  # noqa
from arrayloaders.io import create_store_from_h5ads
from hydra.core.config_store import ConfigStore
from zarr.codecs import BloscCodec, BloscShuffle

from arrayloader_benchmarks.create_sqlite_databases import DB_PATH

zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": None}
)

COMPRESSOR = BloscCodec(cname="lz4", clevel=3, shuffle=BloscShuffle.shuffle)


@dataclass
class BenchmarkConfig:
    store_path: str
    n_shards_input: int = 5
    gene_space: str = "FULL"
    should_densify: bool = True
    zarr_chunk_size: int = 2048
    zarr_shard_size: int = 65536
    anndata_shard_size: int = 2**21


ConfigStore.instance().store(name="create_benchmark_store_config", node=BenchmarkConfig)


@hydra.main(
    version_base=None, config_path="conf", config_name="create_benchmark_store_config"
)
def create_store(cfg: BenchmarkConfig):
    store_path = Path(cfg.store_path)

    if cfg.gene_space == "FULL":
        h5ads = ln.Collection.get("eAgoduHMxuDs5Wem0000")
    elif cfg.gene_space == "PROTEIN_CODING":
        h5ads = ln.Collection.get("k9GqakN96EmLjn1L0000")
    else:
        err_msg = (
            f"Invalid gene_space: {cfg.gene_space}. Must be 'FULL' or 'PROTEIN_CODING'."
        )
        raise ValueError(err_msg)
    h5ad_cache_paths = h5ads.cache()[0].parent

    start_time = time.time()
    create_store_from_h5ads(
        [h5ad_cache_paths / f"shard_{i}.h5ad" for i in range(cfg.n_shards_input)],
        Path(store_path),
        chunk_size=cfg.zarr_chunk_size,
        shard_size=cfg.zarr_shard_size,
        buffer_size=cfg.anndata_shard_size,
        zarr_compressor=(COMPRESSOR,),
        shuffle=False,
        should_denseify=cfg.should_densify,
    )
    creation_time = time.time() - start_time

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    store_type = "DENSE" if cfg.should_densify else "SPARSE"
    store_hash = str(
        hash(
            (
                store_type,
                cfg.gene_space,
                cfg.zarr_chunk_size,
                cfg.zarr_shard_size,
                cfg.anndata_shard_size,
            )
        )
    )
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
            cfg.gene_space,
            cfg.zarr_chunk_size,
            cfg.zarr_shard_size,
            cfg.anndata_shard_size,
            creation_time,
        ),
    )
    conn.commit()
    conn.close()
