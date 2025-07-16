import lamindb as ln
import pandas as pd
from pathlib import Path

ln.settings.transform.stem_uid = "vwOUDSyYo9HN"
ln.settings.transform.version = "1"

ln.track()

main_path = Path.cwd()
BATCH_SIZE = 128

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

GB = 1024 ** 3

df_info = pd.DataFrame(columns=("storage", "size", "n_objects"))
for i, (name, path) in enumerate(paths.items()):
    if path.is_file():
        row = dict(storage=name, size=path.stat().st_size / GB, n_objects=1)
    else:
        sizes = [file.stat().st_size for file in path.rglob("*") if file.is_file()]
        row = dict(storage=name, size=sum(sizes) / GB, n_objects=len(sizes))
    df_info.loc[i] = row

ln.Artifact(df_info, description="Objects sizes for the figure 2").save()

ln.finish()
    
