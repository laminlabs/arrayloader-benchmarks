from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import click
import scipy.sparse as sp
import zarr
import zarrs  # noqa
from arrayloaders.io import ZarrSparseDataset
from torch.utils.data import DataLoader

from arrayloader_benchmarks.utils import benchmark_loader

zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": None}
)


@click.command()
@click.option("--store_path", type=str)
@click.option("--chunk_size", type=int, default=512)
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--use_torch_loader", type=bool, default=True)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=-1)
def benchmark(  # noqa: PLR0917
    store_path: str,
    chunk_size: int = 256,
    preload_nchunks: int = 4,
    use_torch_loader: bool = True,  # noqa: FBT001, FBT002
    num_workers: int = 8,
    batch_size: int = 4096,
    n_samples: int = -1,
):
    store_path = Path(store_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = ZarrSparseDataset(
            shuffle=True,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=1 if use_torch_loader else batch_size,
        )
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
            collate_fn=lambda x: sp.vstack([v[0] for v in x]),
        )
        samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)
    else:
        samples_per_sec, _, _ = benchmark_loader(ds, n_samples, batch_size)


if __name__ == "__main__":
    benchmark()
