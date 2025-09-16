from __future__ import annotations

import anndata as ad
import click
import lamindb as ln
from scdataset import BlockShuffling, scDataset
from torch.utils.data import DataLoader

from arrayloader_benchmarks.utils import benchmark_loader

ln.settings.sync_git_repo = "https://github.com/laminlabs/arrayloader-benchmarks"
ln.track(project="zjQ6EYzMXif4")


@click.command()
@click.option("--block_size", type=int, default=64)
@click.option("--fetch_factor", type=int, default=8)
@click.option("--num_workers", type=int, default=6)
@click.option("--batch_size", type=int, default=4096)
@click.option("--n_samples", type=int, default=2_000_000)
def benchmark(
    block_size: int = 64,
    fetch_factor: int = 8,
    num_workers: int = 6,
    batch_size: int = 4096,
    n_samples: int = 2_000_000,
):
    # Download h5ad shards from laminHub
    store_shards = ln.Collection.get("LaJOdLd0xZ3v5ZBw0000").cache()
    adata = ad.concat(
        [ad.experimental.read_lazy(shard) for shard in store_shards], axis=0
    )

    def fetch_adata(collection, indices):
        return (
            collection[indices].X.toarray(),
            collection[indices].obs["cell_line"].to_numpy(),
        )

    strategy = BlockShuffling(block_size=block_size)
    dataset = scDataset(
        adata,
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


if __name__ == "__main__":
    benchmark()

ln.finish()
