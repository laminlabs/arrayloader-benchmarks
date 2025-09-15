from __future__ import annotations

import warnings

import anndata as ad
import click
import lamindb as ln
import numpy as np
import scipy.sparse as sp
import zarr
import zarrs  # noqa
from arrayloaders import ZarrSparseDataset
from torch.utils.data import DataLoader
from torch.utils.dlpack import from_dlpack

from arrayloader_benchmarks.utils import benchmark_loader

zarr.config.set(
    {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": None}
)

# Suppress zarr vlen-utf8 codec warnings
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    category=UserWarning,
    module="zarr.codecs.vlen_utf8",
)


def collate_fn(elems):
    batch_x = sp.vstack([v[0] for v in elems])
    batch_obs = np.concatenate([v[1] for v in elems])
    return from_dlpack(batch_x), from_dlpack(batch_obs)


@click.command()
@click.option("--store_path", type=str)
@click.option("--chunk_size", type=int, default=512)
@click.option("--preload_nchunks", type=int, default=16)
@click.option("--use_torch_loader", type=bool, default=True)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=-1)
def benchmark(  # noqa: PLR0917
    chunk_size: int = 256,
    preload_nchunks: int = 8,
    use_torch_loader: bool = True,  # noqa: FBT001, FBT002
    num_workers: int = 8,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
):
    # Download store from laminHub
    store_shards = ln.Collection.get("LaJOdLd0xZ3v5ZBw0000").cache()

    ds = ZarrSparseDataset(
        shuffle=True,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=1 if use_torch_loader else batch_size,
    )
    ds.add_datasets(
        datasets=[ad.io.sparse_dataset(zarr.open(p)["X"]) for p in store_shards],
        obs=[
            ad.io.read_elem(zarr.open(p)["obs"])["cell_line"].to_numpy()
            for p in store_shards
        ],
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
        samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)
    else:
        samples_per_sec, _, _ = benchmark_loader(ds, n_samples, batch_size)


if __name__ == "__main__":
    benchmark()
