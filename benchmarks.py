from typing import Literal

import h5py
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset
import pyarrow.parquet
import tiledbsoma as soma
import zarr
from anndata._core.sparse_dataset import sparse_dataset


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
        n_obs, n_vars = len(self.labels), len(self.dataset["ms"]["RNA"]["var"])
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
        self.dataset = sparse_dataset(self.file["X"]) if sparse else self.file["X"]
        self.labels = self.file["obs"]["cell_states"]["codes"]

    def iterate(self, random: bool = False):
        _iterate(self.dataset, self.labels, random, need_sort=True)


class Zarr:
    def __init__(self, path, sparse: bool = False):
        self.file = zarr.open(path)
        self.dataset = sparse_dataset(self.file["X"]) if sparse else self.file["X"]
        self.labels = self.file["obs"]["cell_states"]["codes"]

    def iterate(self, random: bool = False):
        _iterate(self.dataset, self.labels, random, need_sort=False)


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


def benchmark(
    path: str,
    type: Literal["h5py", "zarr", "soma", "arrow", "parquet", "polars"],
    random: bool,
    sparse: bool,
):
    if sparse and type in ["arrow", "parquet", "polars"]:
        raise ValueError(f"{type} does not support sparse data")
    if random and type in ["arrow", "polars"]:
        raise ValueError(f"{type} does not support random access")

    matches = {
        "h5py": H5py,
        "zarr": Zarr,
        "soma": Soma,
        "arrow": Arrow,
        "parquet": Parquet,
        "polars": Polars,
    }

    try:
        cl = matches[type](path, sparse)
        while True:
            yield
            cl.iterate(random)
            yield
            if type == "polars":
                cl = matches[type](path, sparse)
    finally:
        try:
            cl.file.close()
        except (AttributeError, NameError):
            ...


# @click.command()
# @click.argument("path", type=str)
# @click.argument("type", type=click.Choice(["h5py", "zarr", "soma", "arrow", "parquet"]))
# @click.option("--random", type=bool, default=False, is_flag=True)
# @click.option("--sparse", type=bool, default=False, is_flag=True)
# def run(
#     path: str,
#     type: Literal["h5py", "zarr", "soma", "arrow", "parquet"],
#     random: bool,
#     sparse: bool,
# ):
#     benchmark(path, type, random, sparse, True)


# if __name__ == "__main__":
#     main()

# # Z
