# %%
import timeit

import rich
import rich_click as click
from benchmarks import benchmark
from loguru import logger

BATCH_SIZE = 128


paths = {
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


@click.command()
@click.argument("tobench", type=click.Choice(list(benches.keys()) + ["all"]), nargs=-1)
@click.option("--output", "-o", type=str, default="results.tsv")
def main(tobench: list[str], output: str):
    global benches
    console = rich.get_console()

    if not tobench or tobench != "all":
        benches = {k: v for k, v in benches.items() if k in tobench}

    for name, bench in benches.items():
        console.rule(f"[bold]Running '{name}'", align="left")
        with open(output, "a") as f:
            for i in range(5):
                time_taken = timeit.Timer(lambda: next(bench)).timeit(3)
                f.write(f"{name}\t{i}\t{time_taken/3}\n")
                print(f"Loop {i}: {time_taken/3:01f}s/iter of 3 iterations")
                next(bench)


if __name__ == "__main__":
    main()
