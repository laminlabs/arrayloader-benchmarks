from pathlib import Path
from typing import Literal, Protocol, Type
import rich_click as click
import lamindb as ln
import scanpy as sc
import h5py
import tensorstore as ts
import tiledbsoma.io
import rich
import timeit
from anndata import AnnData
from loguru import logger
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset
import pyarrow.parquet
import tiledbsoma as soma
import zarr
import anndata as ad
from dask.distributed import Client, LocalCluster
import os

cluster = LocalCluster(n_workers=os.cpu_count())
client = Client(cluster)

BATCH_SIZE = 128
ln.settings.transform.stem_uid = "r9vQub7PWucj"
ln.settings.transform.version = "1"


def index_iter(n_obs, batch_size, shuffle=True):
    # progress_bar = tqdm(total=round(n_obs / batch_size + 0.5))
    if shuffle:
        indices = np.random.permutation(n_obs)
    else:
        indices = np.arange(n_obs)

    for i in range(0, n_obs, batch_size):
        # progress_bar.update(1)
        yield indices[i : min(i + batch_size, n_obs)]
    # progress_bar.close()


BATCH_SIZE = 128


class Interface(Protocol):
    def __init__(self, path: str, sparse: bool = False): ...

    def iterate(self, random: bool = False): ...


def _iterate(dataset, h5labels, random: bool = False, need_sort: bool = False):
    for batch_idx in index_iter(dataset.shape[0], BATCH_SIZE, shuffle=random):
        if random and need_sort:
            batch_idx.sort()
        batch_X = dataset[batch_idx, :]
        batch_labels = h5labels[batch_idx]


class Soma:
    def __init__(self, path, sparse: bool = True):
        if not sparse:
            raise ValueError("Soma only supports sparse data")

        self.file = soma.open(path)
        self.dataset = self.file["ms"]["RNA"]["X"]["data"]
        self.labels = self.file["obs"]

    def iterate(self, random: bool = False):
        n_obs, n_vars = len(self.labels), len(self.file["ms"]["RNA"]["var"])
        for batch_idx in index_iter(n_obs, BATCH_SIZE, shuffle=random):
            batch_X = (
                self.dataset.read([batch_idx])
                .coos((n_obs, n_vars))
                .concat()
                .to_scipy()
                .tocsr()[batch_idx]
            )
            batch_conds = (
                self.labels.read([batch_idx], column_names=["cell_states"])
                .concat()
                .to_pandas()
            )


class H5py:
    def __init__(self, path, sparse: bool = False):
        self.file = h5py.File(path, mode="r")
        self.dataset = ad.experimental.read_elem_as_dask(self.file["X"]) if sparse else self.file["X"]
        self.labels = self.file["obs"]["cell_states"]["codes"]

    def iterate(self, random: bool = False):
        _iterate(self.dataset, self.labels, random, need_sort=True)


class Zarr:
    def __init__(self, path, sparse: bool = False):
        self.file = zarr.open(path)
        self.dataset = ad.experimental.read_elem_as_dask(self.file["X"]) if sparse else self.file["X"]
        self.labels = self.file["obs"]["cell_states"]["codes"]

    def iterate(self, random: bool = False):
        _iterate(self.dataset, self.labels, random, need_sort=False)


class ZarrV3TensorstoreSharded:
    def __init__(self, path, sparse: bool = False):
        if sparse:
            raise ValueError(
                "Tensorstore not working inside AnnData sparse container yet due to lack of Group support."
            )
        self.dataset = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "file",
                    "path": path,
                },
            },
            read=True,
        ).result()
        self.labels = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "file",
                    "path": path.replace("dense", "labels"),
                },
            },
            read=True,
        ).result()

    def iterate(self, random: bool = False):
        for batch_idx in index_iter(self.dataset.shape[0], BATCH_SIZE, shuffle=random):
            batch_X = self.dataset[batch_idx, :].read().result()
            batch_labels = self.labels[batch_idx].read().result()


class ZarrV2Tensorstore:
    def __init__(self, path, sparse: bool = False):
        if sparse:
            raise ValueError(
                "Tensorstore not working inside AnnData sparse container yet due to lack of Group support."
            )
        self.dataset = ts.open(
            {
                "driver": "zarr",
                "kvstore": {
                    "driver": "file",
                    "path": f"{path}/X",
                },
            },
            read=True,
        ).result()
        self.labels = ts.open(
            {
                "driver": "zarr",
                "kvstore": {"driver": "file", "path": f"{path}/obs/cell_states/codes"},
            },
            read=True,
        ).result()

    def iterate(self, random: bool = False):
        for batch_idx in index_iter(self.dataset.shape[0], BATCH_SIZE, shuffle=random):
            batch_X = self.dataset[batch_idx, :].read().result()
            batch_labels = self.labels[batch_idx].read().result()


class Arrow:
    def __init__(self, path, sparse: bool = False):
        if sparse:
            raise ValueError("Arrow does not support sparse data")

        self.dataset = pyarrow.dataset.dataset(path, format="parquet")

    def iterate(self, random: bool = False):
        if random:
            raise ValueError("Arrow does not support random access")
        for batch in self.dataset.to_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            batch_X = df.iloc[:, :-1].to_numpy()
            batch_labels = df.iloc[:, -1].to_numpy()


class Parquet:
    def __init__(self, path, sparse: bool = False):
        if sparse:
            raise ValueError("Parquet does not support sparse data")

        self.file = pyarrow.parquet.ParquetFile(path)

    def iterate(self, random: bool = False):
        n_batches = self.file.num_row_groups
        iterator = np.random.permutation(n_batches) if random else range(n_batches)
        for i in iterator:
            df: pd.DataFrame = self.file.read_row_group(i).to_pandas()
            batch_X = df.iloc[:, :-1].to_numpy()
            batch_labels = df.iloc[:, -1].to_numpy()


class Polars:
    def __init__(self, path, sparse: bool = False):
        if sparse:
            raise ValueError("Polars does not support sparse data")
        self.lazy = pl.scan_parquet(path)

    @staticmethod
    def _callback(df: pl.DataFrame):
        df.to_pandas(use_pyarrow_extension_array=True)
        return df

    def iterate(self, random: bool = False):
        if random:
            raise ValueError("Polars does not support random access")
        self.lazy.map(Polars._callback).collect(streaming=True)


def run_benchmark(
    path: Path | str,
    type: Literal[
        "h5py",
        "zarr",
        "soma",
        "arrow",
        "parquet",
        "polars",
        "zarrV3tensorstore",
        "zarrV2tensorstore",
    ],
    random: bool,
    sparse: bool,
):
    if sparse and type in ["arrow", "parquet", "polars"]:
        raise ValueError(f"{type} does not support sparse data")
    if random and type in ["arrow", "polars"]:
        raise ValueError(f"{type} does not support random access")

    matches: dict[str, Type[Interface]] = {
        "h5py": H5py,
        "zarr": Zarr,
        "soma": Soma,
        "arrow": Arrow,
        "parquet": Parquet,
        "polars": Polars,
        "zarrV3tensorstore": ZarrV3TensorstoreSharded,
        "zarrV2tensorstore": ZarrV2Tensorstore,
    }

    try:
        cl = matches[type](str(path), sparse)
        while True:
            yield
            cl.iterate(random)
            yield
            if type == "polars":
                cl = matches[type](str(path), sparse)
    finally:
        try:
            cl.file.close()  # type: ignore
        except (AttributeError, NameError):
            ...


def convert_adata_to_different_formats(adata: AnnData) -> None:
    path: Path = Path.cwd()

    # Sparse formats

    # HDF5 and ZARR
    adata.write_h5ad(path / "adata_benchmark_sparse.h5ad")
    adata.write_zarr(path / "adata_benchmark_sparse.zrad")

    # tiledbsoma
    tiledbsoma.io.from_h5ad(
        (path / "adata_benchmark_sparse.soma").as_posix(),
        input_path=(path / "adata_benchmark_sparse.h5ad").as_posix(),
        measurement_name="RNA",
    )

    # Dense formats

    adata.X = adata.X.toarray()
    adata.write_h5ad(path / "adata_benchmark_dense.h5ad")
    adata.write_zarr(path / "adata_benchmark_dense.zrad")
    adata.write_zarr(
        path / f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
        chunks=(BATCH_SIZE, adata.X.shape[1]),
    )

    # Save h5 with dense chunked X, no way to do it with adata.write_h5ad
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

    df_X_labels = sc.get.obs_df(adata, keys=adata.var_names.to_list() + ["cell_states"])

    # Parquet

    # default row groups
    df_X_labels.to_parquet(path / "adata_dense.parquet", compression=None)
    df_X_labels.to_parquet(
        f"adata_dense_chunk_{BATCH_SIZE}.parquet",
        compression=None,
        row_group_size=BATCH_SIZE,
    )

    # tensorstore

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


def run_benchmarks(*, epochs: int) -> None:
    main_path: Path = Path.cwd()

    paths = {
        name: main_path / filename
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

    benches = {}
    for name, path in paths.items():
        benches[name] = run_benchmark(
            path, name.split("_")[0], random=False, sparse="sp" in name
        )
        next(benches[name])

        try:
            b = run_benchmark(path, name.split("_")[0], random=True, sparse="sp" in name)
            next(b)  # Need this to try to initialize the generator to catch errors.
            benches[name + "_rand"] = b
        except ValueError:
            ...
        logger.info("Initialized " + name)

    results_filename = "results.tsv"
    console = rich.get_console()
    for name, bench in benches.items():
        console.rule(f"[bold]Running '{name}'", align="left")
        with open(main_path / results_filename, "a") as f:
            for i in range(epochs):
                time_taken = timeit.Timer(lambda: next(bench)).timeit(1)
                f.write(f"{name}\t{i}\t{time_taken}\n")
                print(f"Loop {i}: {time_taken:01f}s/epoch")
                next(bench)



@click.command()
@click.option("--test", "is_test", is_flag=True, type=bool, default=False, help="Tell Lamin that we're testing")
def main(is_test: bool = True):

    is_production_db = (ln.setup.settings.instance.slug == "laminlabs/arrayloader-benchmarks")
    assert is_test != is_production_db, "You're trying to run a test on the production database"
    if not is_test:
        assert ln.setup.settings.user.handle != "anonymous"

    # track script
    ln.track()

    # load input data
    artifact = ln.Artifact.using("laminlabs/arrayloader-benchmarks").filter(uid="z3AsAOO39crEioi5kEaG").one()

    # subset to 5k genes and less for test runs
    nrows = 256 if is_test else None
    ncols = 500 if is_test else 5000

    with artifact.backed() as adata:
        adata_subset = adata[:nrows, :ncols].to_memory()
        adata_subset.raw = None

    # convert data
    convert_adata_to_different_formats(adata_subset)

    # run benchmarks
    run_benchmarks(epochs=4)

    # finish run
    ln.finish()


if __name__ == "__main__":
    main()
