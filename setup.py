# %%
import h5py
import lamindb as ln
import scanpy as sc
import tiledbsoma.io
import tensorstore as ts
# %%
BATCH_SIZE = 128
ln.setup.close()
ln.setup.load("laminlabs/arrayloader-benchmarks")
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

# %%
# save to tensorstore
sharded_dense_chunk = ts.open(
    {
        'driver': 'zarr3',
        'kvstore': 'file://sharded_dense_chunk.zarr',
        'metadata': {
            "shape": adata.shape,
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [BATCH_SIZE, adata.shape[1]]}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{
            "name": "sharding_indexed",
                "configuration": {
                "chunk_shape": [32, adata.shape[1]],
                "codecs": [{"name":"blosc"}]
            }}],
        },
        "dtype": 'float32',
        'create': True,
        'delete_existing': True,
    }, write=True).result()

sharded_labels = ts.open(
    {
        'driver': 'zarr3',
        'kvstore': 'file://sharded_labels_chunk.zarr',
        'metadata': {
            "shape": (adata.shape[0], ),
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": (10000,)}},
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{
            "name": "sharding_indexed",
                "configuration": {
                "chunk_shape": (1000,),
                "codecs": [{"name":"blosc"}]
            }}],
        },
        "dtype": 'int8',
        'create': True,
        'delete_existing': True,
    }, write=True).result()

sharded_dense_chunk[:, :] = adata.X
sharded_labels[:] = adata.obs['cell_states'].cat.codes.values
