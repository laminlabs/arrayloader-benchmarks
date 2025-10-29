from __future__ import annotations

import json

import click
import lamindb as ln
from lamindb.core import MappedCollection
from torch.utils.data import DataLoader

from arrayloader_benchmarks.utils import benchmark_loader


@click.command()
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
def benchmark(
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
):
    benchmarking_collections = ln.Collection.using("laminlabs/arrayloader-benchmarks")
    h5ad_shards = benchmarking_collections.get("eAgoduHMxuDs5Wem0000").cache()

    mapped_collection = MappedCollection(h5ad_shards)
    loader = DataLoader(
        mapped_collection,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)

    click.echo(json.dumps({"samples/sec": samples_per_sec}))


if __name__ == "__main__":
    # ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"
    ln.track(project="zjQ6EYzMXif4")
    benchmark()
    ln.finish()
