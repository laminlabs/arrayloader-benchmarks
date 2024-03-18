import lamindb as ln
import scanpy as sc
import h5py
import tensorstore as ts
import tiledbsoma.io
import rich
import timeit
from benchmarks import benchmark
from anndata import AnnData
from loguru import logger
from pathlib import Path


BATCH_SIZE = 128


def write_data(path: Path, adata: AnnData):
    # %%
    adata.write_h5ad(path / "adata_benchmark_sparse.h5ad")
    adata.write_zarr(path / "adata_benchmark_sparse.zrad")

    # %%
    tiledbsoma.io.from_h5ad(
        (path / "adata_benchmark_sparse.soma").as_posix(),
        input_path=(path / "adata_benchmark_sparse.h5ad").as_posix(),
        measurement_name="RNA",
    )

    # %% Dense onwards
    adata.X = adata.X.toarray()

    adata.write_h5ad(path / "adata_benchmark_dense.h5ad")
    adata.write_zarr(path / "adata_benchmark_dense.zrad")
    adata.write_zarr(
        path / f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
        chunks=(BATCH_SIZE, adata.X.shape[1]),
    )

    # %%
    # save h5 with dense chunked X, no way to do it with adata.write_h5ad
    with h5py.File(path / f"adata_dense_chunk_{BATCH_SIZE}.h5", mode="w") as f:
        f.create_dataset(
            "adata",
            adata.X.shape,
            dtype=adata.X.dtype,
            data=adata.X,
            chunks=(BATCH_SIZE, adata.X.shape[1]),
        )
        labels = adata.obs.cell_states.cat.codes.to_numpy()
        f.create_dataset("labels", labels.shape, data=labels)

    # %%
    df_X_labels = sc.get.obs_df(adata, keys=adata.var_names.to_list() + ["cell_states"])

    # %%
    # default row groups
    df_X_labels.to_parquet(path / "adata_dense.parquet", compression=None)
    df_X_labels.to_parquet(
        f"adata_dense_chunk_{BATCH_SIZE}.parquet",
        compression=None,
        row_group_size=BATCH_SIZE,
    )

    sharded_dense_chunk = ts.open(
        {
            "driver": "zarr3",
            "kvstore": "file://sharded_dense_chunk.zarr",
            "metadata": {
                "shape": adata.shape,
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [BATCH_SIZE, adata.shape[1]]},
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": [32, adata.shape[1]],
                            "codecs": [{"name": "blosc"}],
                        },
                    }
                ],
            },
            "dtype": "float32",
            "create": True,
            "delete_existing": True,
        },
        write=True,
    ).result()

    sharded_labels = ts.open(
        {
            "driver": "zarr3",
            "kvstore": "file://sharded_labels_chunk.zarr",
            "metadata": {
                "shape": (adata.shape[0],),
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": (10000,)},
                },
                "chunk_key_encoding": {"name": "default"},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": (1000,),
                            "codecs": [{"name": "blosc"}],
                        },
                    }
                ],
            },
            "dtype": "int8",
            "create": True,
            "delete_existing": True,
        },
        write=True,
    ).result()

    sharded_dense_chunk[:, :] = adata.X
    sharded_labels[:] = adata.obs["cell_states"].cat.codes.values


def run_benchmarks(path: Path, output: str, epochs: int):
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
            "zarrV3tensorstore_dense_chunk": "sharded_dense_chunk.zarr",
            "zarrV2tensorstore_dense_chunk": f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
        }.items()
    }
    logger.info("Initializing")

    main_path = path

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

    for name, bench in benches.items():
        console.rule(f"[bold]Running '{name}'", align="left")
        with open(main_path / output, "a") as f:
            for i in range(epochs):
                time_taken = timeit.Timer(lambda: next(bench)).timeit(1)
                f.write(f"{name}\t{i}\t{time_taken}\n")
                print(f"Loop {i}: {time_taken:01f}s/epoch")
                next(bench)


if __name__ == "__main__":
    ln.settings.transform.stem_uid = "r9vQub7PWucj"
    ln.settings.transform.version = "1"

    ln.track()

    artifact = ln.Artifact.filter(uid="z3AsAOO39crEioi5kEaG").one()
    logger.info("Artifact: {}", artifact)

    # we will save in different formats, so no need to cache
    logger.info("Loading data from S3")
    with artifact.backed() as store:
        adata = store[:, :5000].to_memory()
        adata.raw = None

    path = Path.cwd()

    write_data(path, adata)
    output = "results.tsv"
    run_benchmarks(path, output, 4)

    ln.Artifact(output, key=f"cli_runs/{output}").save()

    ln.finish()
