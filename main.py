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
    @staticmethod
    def read(path):
        soma_file = soma.open(path)
        soma_sparse_dataset = soma_file["ms"]["RNA"]["X"]["data"]
        soma_labels = soma_file["obs"]
        return soma_file, (soma_sparse_dataset, soma_labels)

    @staticmethod
    def iterate(dataset, labels, random: bool = False):
        n_obs, n_vars = len(labels), len(dataset["ms"]["RNA"]["var"])
        for batch_idx in index_iter(len(labels), BATCH_SIZE, shuffle=random):
            batch_X = (
                dataset.read([batch_idx])
                .coos((n_obs, n_vars))
                .concat()
                .to_scipy()
                .tocsr()[batch_idx]
            )
            batch_conds = (
                labels.read([batch_idx], column_names=["cell_states"])
                .concat()
                .to_pandas()
            )


class H5py:
    @staticmethod
    def read(path, is_sparse: bool = False):
        h5_file = h5py.File(path, mode="r")
        dataset = sparse_dataset(h5_file["X"]) if is_sparse else h5_file["X"]
        return h5_file, (dataset, h5_file["obs"]["cell_states"]["codes"])

    @staticmethod
    def iterate(dataset, h5labels, random: bool = False):
        _iterate(dataset, h5labels, random, need_sort=True)


class Zarr:
    @staticmethod
    def read(path, is_sparse: bool = False):
        zarr_file = zarr.open(path)
        dataset = sparse_dataset(zarr_file["X"]) if is_sparse else zarr_file["X"]
        return zarr_file, (dataset, zarr_file["obs"]["cell_states"]["codes"])

    @staticmethod
    def iterate(dataset, h5labels, random: bool = False):
        _iterate(dataset, h5labels, random, need_sort=False)


class Arrow:
    @staticmethod
    def read(path):
        dataset = pyarrow.dataset.dataset(path, format="parquet")
        return dataset

    @staticmethod
    def iterate(dataset):
        for batch in dataset.to_batches(batch_size=BATCH_SIZE):
            df = batch.to_pandas()
            batch_X = df.iloc[:, :-1].to_numpy()
            batch_labels = df.iloc[:, -1].to_numpy()


class Parquet:
    @staticmethod
    def read(path):
        pq_file = pyarrow.parquet.ParquetFile(path)
        return pq_file

    @staticmethod
    def iterate(pq_file: pyarrow.parquet.ParquetFile, random: bool = False):
        n_batches = pq_file.num_row_groups
        iterator = np.random.permutation(n_batches) if random else range(n_batches)
        for i in iterator:
            df: pd.DataFrame = pq_file.read_row_group(i).to_pandas()
            batch_X = df.iloc[:, :-1].to_numpy()
            batch_labels = df.iloc[:, -1].to_numpy()


class Polars:
    @staticmethod
    def read(path):
        return pl.scan_parquet(path)

    @staticmethod
    def _callback(df: pl.DataFrame):
        df.to_pandas(use_pyarrow_extension_array=True)
        return df

    @staticmethod
    def iterate(df: pl.LazyFrame):
        df.map(Polars._callback).collect(streaming=True)


def benchmark(
    path: str,
    type: Literal["h5py", "zarr", "soma", "arrow", "parquet", "polars"],
    random: bool,
    sparse: bool,
):
    try:
        match type:
            case "h5py":
                file, (dataset, labels) = H5py.read(path, sparse)
                while True:
                    yield
                    H5py.iterate(dataset, labels, random)
            case "zarr":
                file, (dataset, labels) = Zarr.read(path, sparse)
                while True:
                    yield
                    Zarr.iterate(dataset, labels, random)
            case "soma":
                if not sparse:
                    raise ValueError("Soma only supports sparse data")
                (dataset, labels) = Soma.read(path)
                while True:
                    yield
                    Soma.iterate(dataset, labels, random)
            case "arrow":
                if random:
                    raise ValueError("Arrow does not support random access")
                if sparse:
                    raise ValueError("Arrow does not support sparse data")
                dataset = Arrow.read(path)
                while True:
                    yield
                    Arrow.iterate(dataset)
            case "parquet":
                if sparse:
                    raise ValueError("Parquet does not support sparse data")
                pq_file = Parquet.read(path)
                while True:
                    yield
                    Parquet.iterate(pq_file, random)
            case "polars":
                if random:
                    raise ValueError("Polars does not support random access")
                if sparse:
                    raise ValueError("Polars does not support sparse data")
                df = Polars.read(path)
                while True:
                    yield
                    Polars.iterate(df)
            case _:
                raise NotImplementedError(f"Type {type} not implemented")

    finally:
        try:
            file.close()
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
