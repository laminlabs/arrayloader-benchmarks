# %%
from pathlib import Path

import h5py
import lamindb as ln
import rich_click as click
import scanpy as sc
import tiledbsoma.io
from loguru import logger

BATCH_SIZE = 128


@click.command()
@click.option(
    "--path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=Path("."),
)
@click.option("--nrows", type=int, default=-1)
@click.option("--ncols", type=int, default=5000)
def main(path: Path, nrows: int | None, ncols: int | None):
    if nrows == -1:
        nrows = None
    if ncols == -1:
        ncols = None

    ln.setup.close()
    ln.setup.load("laminlabs/arrayloader-benchmarks")

    artifact = ln.Artifact.filter(uid="z3AsAOO39crEioi5kEaG").one()
    logger.info("Artifact: {}", artifact)
    artifact.view_lineage()
    artifact.describe()

    # %%
    # we will save in different formats, so no need to cache
    logger.info("Loading data from S3")
    with artifact.backed() as store:
        # ~2GB sparse and ~4.6GB dense stored as h5ad
        adata = store[:nrows, :ncols].to_memory()
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

    # %%
    df_X_labels.to_parquet(
        path / f"adata_dense_chunk_{BATCH_SIZE}.parquet",
        compression=None,
        row_group_size=BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
