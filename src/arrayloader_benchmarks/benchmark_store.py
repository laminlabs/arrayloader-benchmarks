from __future__ import annotations

import sqlite3
import time
import warnings
from pathlib import Path

import anndata as ad
import click
import scipy.sparse as sp
import zarr
import zarrs  # noqa
from arrayloaders.io import ZarrDenseDataset, ZarrSparseDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from arrayloader_benchmarks.create_sqlite_databases import DB_PATH
from arrayloader_benchmarks.utils import hash_store_params

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


@click.command()
@click.option("--store_path", type=str)
@click.option("--gene_space", type=str)
@click.option("--zarr_chunk_size", type=int, default=2048)
@click.option("--zarr_shard_size", type=int, default=65536)
@click.option("--anndata_shard_size", type=int, default=2**21)
@click.option("--should_densify", type=bool, default=True)
@click.option("--chunk_size", type=int, default=512)
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--use_torch_loader", type=bool, default=True)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=2048)
@click.option("--n_samples", type=int, default=-1)
def benchmark_store(  # noqa: PLR0913, PLR0917
    store_path: str,
    gene_space: str = "PROTEIN_CODING",
    zarr_chunk_size: int = 2048,
    zarr_shard_size: int = 65536,
    anndata_shard_size: int = 2**21,
    should_densify: bool = True,  # noqa: FBT001, FBT002
    chunk_size: int = 512,
    preload_nchunks: int = 16,
    use_torch_loader: bool = True,  # noqa: FBT001, FBT002
    num_workers: int = 4,
    batch_size: int = 2048,
    n_samples: int = -1,
):
    store_path = Path(store_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if not should_densify:
            ds = ZarrSparseDataset(
                shuffle=True,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=1 if use_torch_loader else batch_size,
            )
            collate_fn = lambda x: sp.vstack([v[0] for v in x])
        else:
            ds = ZarrDenseDataset(
                shuffle=True,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                batch_size=1 if use_torch_loader else batch_size,
            )
            collate_fn = None

    store_hash = hash_store_params(
        gene_space=gene_space,
        zarr_chunk_size=zarr_chunk_size,
        zarr_shard_size=zarr_shard_size,
        anndata_shard_size=anndata_shard_size,
        should_densify=should_densify,
    )
    store_path = store_path / store_hash
    if not store_path.exists():
        err_msg = (
            f"Store for supplied settings does not exist. "
            f"Please create the store with following parameters: "
            f"gene_space={gene_space}, "
            f"zarr_chunk_size={zarr_chunk_size}, "
            f"zarr_shard_size={zarr_shard_size}, "
            f"anndata_shard_size={anndata_shard_size}, "
            f"should_densify={should_densify}."
        )
        raise FileNotFoundError(err_msg)
    ds.add_datasets(
        [ad.io.sparse_dataset(zarr.open(p)["X"]) for p in store_path.glob("*.zarr")]
    )

    n_samples = n_samples if n_samples != -1 else len(ds)
    if use_torch_loader:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        samples_per_sec, _, _ = benchmark(loader, n_samples, batch_size)
    else:
        samples_per_sec, _, _ = benchmark(ds, n_samples, batch_size)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO results (
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
            "DENSE" if should_densify else "SPARSE",
            n_samples,
            batch_size,
            use_torch_loader,
            chunk_size,
            preload_nchunks,
            num_workers,
            samples_per_sec,
        ),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    benchmark_store()
