from __future__ import annotations

import click
import lamindb as ln
from lamindb.core import MappedCollection
from torch.utils.data import DataLoader

from arrayloader_benchmarks.utils import benchmark_loader

ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"
ln.track(project="zjQ6EYzMXif4")


@click.command()
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
def benchmark(
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
):
    # Download h5ad shards from laminHub
    h5ad_shards = ln.Collection.get("eAgoduHMxuDs5Wem0000").cache()

    mapped_collection = MappedCollection(h5ad_shards)
    loader = DataLoader(
        mapped_collection,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    samples_per_sec, _, _ = benchmark_loader(loader, n_samples, batch_size)


if __name__ == "__main__":
    benchmark()

ln.finish()
