from __future__ import annotations

import sqlite3
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

import anndata as ad
import hydra
import scipy.sparse as sp
import zarr
import zarrs  # noqa
from arrayloaders.io import ZarrDenseDataset, ZarrSparseDataset
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
from tqdm import tqdm

from arrayloader_benchmarks.create_sqlite_databases import DB_PATH

zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": None}
)


def benchmark(loader, n_samples, batch_size):
    num_iter = n_samples // batch_size
    loader_iter = loader.__iter__()

    start_time = time.time()
    batch_times = []
    batch_time = time.time()
    for i, _batch in tqdm(enumerate(loader_iter), total=num_iter):
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
        if i == num_iter:
            break

    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (num_iter * batch_size)
    print(f"time per sample: {time_per_sample:.2f} μs")
    samples_per_sec = num_iter * batch_size / execution_time
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")

    return samples_per_sec, time_per_sample, batch_times


@dataclass
class BenchmarkConfig:
    store_path: str
    store_type: Literal["SPARSE", "DENSE"]
    n_samples: int = -1
    batch_size: int = 2048
    use_torch_loader: bool = True
    chunk_size: int = 1024
    preload_nchunks: int = 8
    num_workers: int = 4


ConfigStore.instance().store(name="benchmark_store_config", node=BenchmarkConfig)


@hydra.main(version_base=None, config_path="conf", config_name="benchmark_store_config")
def benchmark_store(cfg: BenchmarkConfig):
    store_path = Path(cfg.store_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg.store_type == "SPARSE":
            ds = ZarrSparseDataset(
                shuffle=True,
                chunk_size=cfg.chunk_size,
                preload_nchunks=cfg.preload_nchunks,
            )
            collate_fn = lambda x: sp.vstack([v[0] for v in x])
        elif cfg.store_type == "DENSE":
            ds = ZarrDenseDataset(
                shuffle=True,
                chunk_size=cfg.chunk_size,
                preload_nchunks=cfg.preload_nchunks,
            )
            collate_fn = None
        else:
            err_msg = (
                f"Invalid store_type {cfg.store_type}. Must be 'SPARSE' or 'DENSE'."
            )
            raise ValueError(err_msg)
        ds.add_datasets(
            [
                ad.io.sparse_dataset(zarr.open(p)["X"])
                for p in Path(store_path).glob("*.zarr")
            ]
        )

    n_samples = cfg.n_samples if cfg.n_samples != -1 else len(ds)
    if cfg.use_torch_loader:
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        samples_per_sec, _, _ = benchmark(loader, n_samples, cfg.batch_size)
    else:
        samples_per_sec, _, _ = benchmark(ds, n_samples, 1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO benchmarks (
            store_path,
            store_type,
            n_samples,
            batch_size,
            use_torch_loader,
            chunk_size,
            preload_nchunks,
            num_workers,
            samples_per_sec
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(store_path),
            cfg.store_type,
            n_samples,
            cfg.batch_size,
            cfg.use_torch_loader,
            cfg.chunk_size,
            cfg.preload_nchunks,
            cfg.num_workers,
            samples_per_sec,
        ),
    )
    conn.commit()
    conn.close()
