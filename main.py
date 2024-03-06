# %%
import timeit
from pathlib import Path

import rich
import rich_click as click
from loguru import logger
import lamindb as ln
from benchmarks import benchmark

ln.transform.stem_uid = "Mf5gs9ezJCrI"
ln.transform.version = "1"

BATCH_SIZE = 128

logger.info("Initializing")


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
@click.option("--test", "is_test", is_flag=True, type=bool, default=False, help="Tell Lamin that we're testing")
def main(path: Path, tobench: list[str], epochs: int, output: str, is_test: bool):
    console = rich.get_console()

    # Lamin checks
    is_production_instance = (ln.setup.settings.instance.identifier == "laminlabs/arrayloader-benchmarks")
    assert is_test != is_production_instance, "You're trying to run a test on the production instance"
    if not is_test:
        assert ln.setup.settings.user.handle != "anonymous"

    # it'd be nice to track the params of this run in a json now, but we can't yet do this
    ln.track()

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
            "zarrV3tensorstore_dense_chunk": "sharded_dense_chunk.zarr",
            "zarrV2tensorstore_dense_chunk": f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
        }.items()
    }
    logger.info("Initializing")

    benches = {}
    for name, path in paths.items():
        benches[name] = benchmark(
            path, name.split("_")[0], random=False, sparse="sp" in name
        )
        next(benches[name])

        try:
            b = benchmark(path, name.split("_")[0], random=True, sparse="sp" in name)
            next(b)  # Need this to try to initialize the generator to catch errors.
            benches[name + "_rand"] = b
        except ValueError:
            ...
        logger.info("Initialized " + name)

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

    ln.Artifact(output, key=f"cli_runs/{output}").save()


if __name__ == "__main__":
    main()
