from __future__ import annotations

import datetime
import warnings
from typing import TYPE_CHECKING

import anndata as ad
import click
import lamindb as ln
from torch.utils.data import DataLoader

from arrayloader_benchmarks import benchmark_loader, compute_spec

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# comment this line out if you don't want to enforce running committed code
ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"

BENCHMARK_FEATURE_DTYPES = {
    "tool": str,
    "collection": ln.Collection,
    "n_datasets": int,
    "n_samples_per_sec": float,
    "n_samples_loaded": int,
    "n_samples_collection": int,
    "num_workers": int,
    "batch_size": int,
    "chunk_size": int,
    "compute_spec": str,
    "run": ln.Run,
    "timestamp": datetime.datetime,
    "user": ln.User,
}


def get_datasets(*, collection_key: str, n_datasets: int = 1) -> tuple[list[Path], int]:
    db = ln.DB("laminlabs/arrayloader-benchmarks")
    collection = db.Collection.get(key=collection_key)
    if ln.setup.settings.instance.slug != "laminlabs/arrayloader-benchmarks":
        # transfer in case the collection is on a different instance
        collection.save()
    if n_datasets == -1:
        n_datasets = collection.artifacts.count()
    local_paths = [
        artifact.cache(
            batch_size=48
        )  # batch_size during download shouldn't be necessary to set
        for artifact in collection.ordered_artifacts.all()[:n_datasets]
    ]
    n_samples_collection = [
        artifact.n_observations
        for artifact in collection.ordered_artifacts.all()[:n_datasets]
    ]
    return local_paths, sum(n_samples_collection)


def get_scdataset_loader(
    local_paths: list[Path],
    block_size: int = 4,
    fetch_factor: int = 16,
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
) -> Iterable:
    # local imports so that it can be run without installing all dependencies
    from scdataset import BlockShuffling, scDataset
    from torch.utils.data import DataLoader

    adata_collection = ad.experimental.AnnCollection(
        [ad.read_h5ad(shard, backed="r") for shard in local_paths]
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
    return loader


def get_mappedcollection_loader(
    local_paths: list[Path],
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
) -> Iterable:
    mapped_collection = ln.core.MappedCollection(local_paths, parallel=True)
    loader = DataLoader(
        mapped_collection,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=mapped_collection.torch_worker_init_fn,
        drop_last=True,
    )
    return loader


def get_annbatch_loader(
    local_paths: list[Path],
    chunk_size: int = 256,
    preload_nchunks: int = 64,
    use_torch_loader: bool = False,  # noqa: FBT001, FBT002
    batch_size: int = 4096,
    include_obs: bool = True,  # noqa: FBT001, FBT002
):
    # local imports so that it can be run without installing all dependencies
    import zarr
    import zarrs  # noqa
    from annbatch import Loader

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

    loader = Loader(
        shuffle=True,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        batch_size=batch_size,
        preload_to_gpu=False,
        to_torch=use_torch_loader,
    )
    loader.add_datasets(
        datasets=[ad.io.sparse_dataset(zarr.open(p)["X"]) for p in local_paths],
        obs=[ad.io.read_elem(zarr.open(p)["obs"])[["cell_line"]] for p in local_paths]
        if include_obs
        else None,
    )
    return loader


def _largest_divisor_at_most(value: int, upper_bound: int) -> int:
    for divisor in range(min(value, upper_bound), 0, -1):
        if value % divisor == 0:
            return divisor
    return 1


def _create_benchmarking_sheet() -> ln.Record:
    benchmark_feature_type = ln.Feature(name="Benchmarks", is_type=True).save()
    features = {
        feature_name: ln.Feature(
            name=feature_name, dtype=dtype, type=benchmark_feature_type
        ).save()
        for feature_name, dtype in BENCHMARK_FEATURE_DTYPES.items()
    }
    schema = ln.Schema(
        list(features.values()), name="loading_benchmark_result_schema"
    ).save()

    benchmarks = ln.Record(name="Benchmarks", is_type=True).save()
    sheet = ln.Record(
        name="run_loading_benchmark_on_collection.py",
        type=benchmarks,
        is_type=True,
        schema=schema,
    ).save()
    return sheet


@click.command()
@click.argument(
    "tool", type=click.Choice(["annbatch", "MappedCollection", "scDataset"])
)
@click.option(
    "--collection",
    type=click.Choice(["Tahoe100M_tiny", "Tahoe100M"]),
    default="Tahoe100M_tiny",
)
@click.option("--chunk_size", type=int, default=256)
@click.option("--preload_nchunks", type=int, default=8)
@click.option("--use_torch_loader", type=bool, default=False)
@click.option("--block_size", type=int, default=4)
@click.option("--fetch_factor", type=int, default=16)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
@click.option("--include_obs", type=bool, default=True)
@click.option("--n_datasets", type=int, default=1)
@click.option("--project", type=str, default="Arrayloader benchmarks v2")
@ln.flow("LDSa3IJYQkbm")
def run(
    tool: str,  # No default value since it's required
    collection: str = "Tahoe100M_tiny",
    chunk_size: int = 256,
    preload_nchunks: int = 64,
    use_torch_loader: bool = False,  # noqa: FBT001, FBT002
    block_size: int = 4,
    fetch_factor: int = 16,
    num_workers: int = 4,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
    include_obs: bool = True,  # noqa: FBT001, FBT002
    n_datasets: int = 1,
    project: str = "Arrayloader benchmarks v2",
):
    collection_key = (
        f"{collection}_h5ad"
        if tool in {"MappedCollection", "scDataset"}
        else f"{collection}_zarr"
    )
    local_paths, n_samples_collection = get_datasets(
        collection_key=collection_key, n_datasets=n_datasets
    )

    if 10 * batch_size > n_samples_collection:
        print(f"reducing batch size from {batch_size} to {n_samples_collection // 10}")
        batch_size = n_samples_collection // 10

    if tool == "annbatch":
        preload_window = chunk_size * preload_nchunks
        if preload_window % batch_size != 0:
            adjusted_batch_size = _largest_divisor_at_most(preload_window, batch_size)
            print(
                "adjusting annbatch batch size from "
                f"{batch_size} to {adjusted_batch_size} so that "
                "chunk_size * preload_nchunks is divisible by batch_size"
            )
            batch_size = adjusted_batch_size

    n_samples = (
        n_samples_collection
        if n_samples == -1
        else min(n_samples, n_samples_collection)
    )

    if tool == "annbatch":
        loader = get_annbatch_loader(
            local_paths,
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            use_torch_loader=use_torch_loader,
            batch_size=batch_size,
            include_obs=include_obs,
        )
    elif tool == "MappedCollection":
        loader = get_mappedcollection_loader(
            local_paths,
            num_workers=num_workers,
            batch_size=batch_size,
            n_samples=n_samples,
        )
    elif tool == "scDataset":
        loader = get_scdataset_loader(
            local_paths,
            block_size=block_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
            batch_size=batch_size,
            n_samples=n_samples,
        )

    n_samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)
    sheet = ln.Record.filter(
        name="run_loading_benchmark_on_collection.py"
    ).one_or_none()
    if sheet is None:
        sheet = _create_benchmarking_sheet()

    ln.Record(
        type=sheet,
        features={
            "tool": tool,
            "collection": collection_key,
            "n_datasets": n_datasets,
            "n_samples_per_sec": n_samples_per_sec,
            "n_samples_loaded": n_samples,
            "n_samples_collection": n_samples_collection,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "compute_spec": compute_spec.get_aws_sagemaker_instance_type(),
            "run": ln.context.run,
            "timestamp": datetime.datetime.now(),
            "user": ln.setup.settings.user.handle,
        },
    ).save()


if __name__ == "__main__":
    run()
