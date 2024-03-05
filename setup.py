# %%
import subprocess

import h5py
import lamindb as ln
import scanpy as sc
import tiledbsoma.io

# %%
BATCH_SIZE = 128
subprocess.run("lamin load laminlabs/arrayloader-benchmarks", shell=True)
artifact = ln.Artifact.filter(uid="z3AsAOO39crEioi5kEaG").one()

# %%
artifact.view_lineage()

# %%
artifact.describe()

# %%
# we will save in different formats, so no need to cache
with artifact.backed() as store:
    adata = store[:, :5000].to_memory()  # ~2GB sparse and ~4.6GB dense stored as h5ad
# %%
adata.write_h5ad("adata_benchmark_sparse.h5ad")
adata.write_zarr("adata_benchmark_sparse.zrad")

# %%
tiledbsoma.io.from_h5ad(
    "adata_benchmark_sparse.soma",
    input_path="adata_benchmark_sparse.h5ad",
    measurement_name="RNA",
)

# %% Dense onwards
adata.X = adata.X.toarray()

adata.write_h5ad("adata_benchmark_dense.h5ad")
adata.write_zarr("adata_benchmark_dense.zrad")
adata.write_zarr(
    f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
    chunks=(BATCH_SIZE, adata.X.shape[1]),
)

# %%
# save h5 with dense chunked X, no way to do it with adata.write_h5ad
with h5py.File(f"adata_dense_chunk_{BATCH_SIZE}.h5", mode="w") as f:
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
df_X_labels.to_parquet("adata_dense.parquet", compression=None)

# %%
df_X_labels.to_parquet(
    f"adata_dense_chunk_{BATCH_SIZE}.parquet",
    compression=None,
    row_group_size=BATCH_SIZE,
)
