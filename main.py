# %%
import timeit
from pathlib import Path

import rich
import rich_click as click
from loguru import logger

from benchmarks import benchmark

BATCH_SIZE = 128


@click.command()
@click.argument("tobench", type=str, nargs=-1)
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=Path("."),
)
@click.option("--epochs", type=int, default=4)
@click.option("--output", "-o", type=str, default="results.tsv")
def main(path: Path, tobench: list[str], epochs: int, output: str):
    console = rich.get_console()

    paths = {
        name: path / filename
        for name, filename in {
            "h5py_sp": "adata_benchmark_sparse.h5ad",
            "soma_sp": "adata_benchmark_sparse.soma",
            "h5py_dense": "adata_benchmark_dense.h5ad",
            "zarr_sp": "adata_benchmark_sparse.zrad",
            "zarr_dense": "adata_benchmark_dense.zrad",
            "zarr_dense_chunk": f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
            "parquet": "adata_dense.parquet",
            "polars": "adata_dense.parquet",
            "parquet_chunk": f"adata_dense_chunk_{BATCH_SIZE}.parquet",
            "arrow": "adata_dense.parquet",
            "arrow_chunk": f"adata_dense_chunk_{BATCH_SIZE}.parquet",
        }.items()
    }
    logger.info("Initializing")
    benches = {}
    for name, path in paths.items():
        benches[name] = benchmark(path, name.split("_")[0], False, "sp" in name)
        next(benches[name])
        try:
            benches[name + "_rand"] = benchmark(
                path, name.split("_")[0], True, "sp" in name
            )
            next(benches[name + "_rand"])
        except ValueError:
            ...

    if tobench and "all" not in tobench:
        benches = {k: v for k, v in benches.items() if k in tobench}

    for name, bench in benches.items():
        console.rule(f"[bold]Running '{name}'", align="left")
        with open(output, "a") as f:
            for i in range(epochs):
                time_taken = timeit.Timer(lambda: next(bench)).timeit(1)
                f.write(f"{name}\t{i}\t{time_taken}\n")
                print(f"Loop {i}: {time_taken:01f}s/epoch")
                next(bench)


if __name__ == "__main__":
    main()
