from __future__ import annotations

import datetime
import warnings
from typing import TYPE_CHECKING, Literal

import anndata as ad
import click
import lamindb as ln
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from arrayloader_benchmarks import benchmark_loader

if TYPE_CHECKING:
    from pathlib import Path


def get_datasets(
    *, collection_key: str, cache: bool = True, n_shards: int = 1
) -> tuple[list[Path], int]:
    benchmarking_collections = ln.Collection.using("laminlabs/arrayloader-benchmarks")
    collection = benchmarking_collections.get(key=collection_key)
    if n_shards == -1:
        n_shards = collection.artifacts.count()
    local_shards = [
        artifact.cache(
            batch_size=48
        )  # batch_size during download shouldn't be necessary to set
        for artifact in collection.ordered_artifacts.all()[:n_shards]
    ]
    n_samples = [
        artifact.n_observations
        for artifact in collection.ordered_artifacts.all()[:n_shards]
    ]
    return local_shards, n_samples


def run_scdataset(
    local_shards: list[Path],
    block_size: int = 4,
    fetch_factor: int = 16,
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
) -> float:
    # local imports so that it can be run without installing all dependencies
    from scdataset import BlockShuffling, scDataset
    from torch.utils.data import DataLoader

    adata_collection = ad.experimental.AnnCollection(
        [ad.read_h5ad(shard, backed="r") for shard in local_shards]
    )

    def fetch_adata(collection, indices):
        return collection[indices].X

    strategy = BlockShuffling(block_size=block_size)
    dataset = scDataset(
        adata_collection,
        strategy,
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        fetch_callback=fetch_adata,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=fetch_factor + 1,
    )
    samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)
    return samples_per_sec


def run_mappedcollection(
    local_shards: list[Path],
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
) -> float:
    mapped_collection = ln.core.MappedCollection(local_shards)
    loader = DataLoader(
        mapped_collection,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)
    return samples_per_sec


def run_annbatch(
    local_shards: list[Path],
    chunk_size: int = 256,
    preload_nchunks: int = 64,
    use_torch_loader: bool = False,  # noqa: FBT001, FBT002
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
    include_obs: bool = True,  # noqa: FBT001, FBT002
):
    # local imports so that it can be run without installing all dependencies
    import scipy.sparse as sp
    import zarr
    from arrayloaders import ZarrSparseDataset
    from torch.utils.data import DataLoader
    from torch.utils.dlpack import from_dlpack

    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": None,
        }
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

    ds = ZarrSparseDataset(
        shuffle=True,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=1 if use_torch_loader else batch_size,
    )
    ds.add_datasets(
        datasets=[ad.io.sparse_dataset(zarr.open(p)["X"]) for p in local_shards],
        obs=[
            ad.io.read_elem(zarr.open(p)["obs"])["cell_line"].to_numpy()
            for p in local_shards
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

    return samples_per_sec


@click.command()
@click.option(
    "--tool",
    type=Literal["annbatch", "MappedCollection", "scDataset"],
    default="annbatch",
)
@click.option(
    "--collection",
    type=Literal["Tahoe100M_tiny", "Tahoe100M"],
    default="Tahoe100M_tiny",
)
@click.option("--chunk_size", type=int, default=256)
@click.option("--preload_nchunks", type=int, default=8)
@click.option("--use_torch_loader", type=bool, default=True)
@click.option("--block_size", type=int, default=4)
@click.option("--fetch_factor", type=int, default=16)
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
@click.option("--include_obs", type=bool, default=True)
@click.option("--n_shards", type=int, default=1)
def run(
    tool: Literal["annbatch", "MappedCollection", "scDataset"] = "annbatch",
    collection: Literal["Tahoe100M_tiny", "Tahoe100M"] = "Tahoe100M_tiny",
    chunk_size: int = 256,
    preload_nchunks: int = 64,
    use_torch_loader: bool = False,  # noqa: FBT001, FBT002
    block_size: int = 4,
    fetch_factor: int = 16,
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
    include_obs: bool = True,  # noqa: FBT001, FBT002
    n_shards: int = 1,
):
    ln.track(project="zjQ6EYzMXif4")

    if tool in {"MappedCollection", "scDataset"}:
        local_shards, n_samples = get_datasets(
            collection_key=f"{collection}_h5ad", cache=True, n_shards=n_shards
        )
    else:
        local_shards, n_samples = get_datasets(
            collection_key=f"{collection}_zarr", cache=True, n_shards=n_shards
        )

    if tool == "annbatch":
        n_samples_per_sec = run_annbatch(
            local_shards,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            use_torch_loader=use_torch_loader,
            num_workers=num_workers,
            batch_size=batch_size,
            n_samples=n_samples,
            include_obs=include_obs,
        )
    elif tool == "MappedCollection":
        n_samples_per_sec = run_mappedcollection(
            local_shards,
            num_workers=num_workers,
            batch_size=batch_size,
            n_samples=n_samples,
        )
    elif tool == "scDataset":
        n_samples_per_sec = run_scdataset(
            local_shards,
            block_size=block_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
            batch_size=batch_size,
            n_samples=n_samples,
        )

    new_result = {
        "tool": tool,
        "samples_per_sec": n_samples_per_sec,
        "n_observations": n_samples,
        "chunk_size": chunk_size,
        "run_uid": ln.context.run.uid,
        "timestamp": datetime.datetime.now(datetime.UTC),
    }

    results_key = "arrayloader_benchmarks_v2/tahoe100m_benchmark.parquet"
    results_description = "Results of v2 of the arrayloader benchmarks"

    try:
        results_af = ln.Artifact.get(key=results_key)
        results_df = results_af.load()
    except ln.Artifact.DoesNotExist:
        results_df = pd.DataFrame(columns=new_result.keys())

    results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)

    ln.Artifact.from_dataframe(
        results_df,
        key=results_key,
        description=results_description,
    ).save()


if __name__ == "__main__":
    run()
