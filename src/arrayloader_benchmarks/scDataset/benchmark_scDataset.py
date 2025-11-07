from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import click
from scdataset import BlockShuffling, scDataset
from torch.utils.data import DataLoader

from arrayloader_benchmarks.utils import benchmark_loader


@click.command()
@click.option("--store_path", type=str, default="")
@click.option("--block_size", type=int, default=4)
@click.option("--fetch_factor", type=int, default=16)
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
def benchmark(  # noqa: PLR0917
    store_path: str = "",
    block_size: int = 4,
    fetch_factor: int = 16,
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
):
    if store_path:
        h5ad_shards = list(Path(store_path).glob("*.h5ad"))
    else:
        import lamindb as ln

        benchmarking_collections = ln.Collection.using(
            "laminlabs/arrayloader-benchmarks"
        )
        h5ad_shards = benchmarking_collections.get("eAgoduHMxuDs5Wem0000").cache()

    adata_collection = ad.experimental.AnnCollection(
        [ad.read_h5ad(shard, backed="r") for shard in h5ad_shards]
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

    samples_per_sec, _, _, total_time = benchmark_loader(loader, n_samples, batch_size)

    click.echo(json.dumps({"samples/sec": samples_per_sec, "total_time": total_time}))


if __name__ == "__main__":
    benchmark()
