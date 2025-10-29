from __future__ import annotations

import json
import warnings

import anndata as ad
import click
import lamindb as ln
import numpy as np
import scipy.sparse as sp
import zarr
import zarrs  # noqa
from annbatch import ZarrSparseDataset
from torch.utils.data import DataLoader
from torch.utils.dlpack import from_dlpack

from arrayloader_benchmarks.utils import benchmark_loader

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

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
@click.option("--chunk_size", type=int, default=256)
@click.option("--preload_nchunks", type=int, default=8)
@click.option("--use_torch_loader", type=bool, default=True)
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
@click.option("--include_obs", type=bool, default=True)
def benchmark(  # noqa: PLR0917
    chunk_size: int = 256,
    preload_nchunks: int = 64,
    use_torch_loader: bool = False,  # noqa: FBT001, FBT002
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
    include_obs: bool = True,  # noqa: FBT001, FBT002
):
    benchmarking_collections = ln.Collection.using("laminlabs/arrayloader-benchmarks")
    collection = benchmarking_collections.get("LaJOdLd0xZ3v5ZBw0000")
    store_shards = [
        artifact.cache(batch_size=48) for artifact in collection.ordered_artifacts.all()
    ]

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
        ]
        if include_obs
        else None,
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

    click.echo(json.dumps({"samples/sec": samples_per_sec}))


if __name__ == "__main__":
    ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"
    ln.track(project="zjQ6EYzMXif4")
    benchmark()
    ln.finish()
