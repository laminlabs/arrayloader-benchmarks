# %%
import timeit

from main import benchmark

BATCH_SIZE = 128


paths = {
    "h5py_sp": "adata_benchmark_sparse.h5ad",
    "soma_sp": "adata_benchmark_sparse.soma",
    "h5py_dense": "adata_benchmark_dense.h5ad",
    # "zarr_sp": "adata_benchmark_sparse.zrad",
    # "zarr_dense": "adata_benchmark_dense.zrad",
    # "zarr_dense_chunk": f"adata_benchmark_dense_chunk_{BATCH_SIZE}.zrad",
    "polars": "X_dense.parquet",
}

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


for name, bench in benches.items():
    print(name)
    with open("results.tsv", "a") as f:
        for i in range(5):
            time_taken = timeit.Timer(lambda: next(bench)).timeit(3)
            f.write(f"{name}\t{i}\t{time_taken/3}\n")
            print(f"{i}\t{time_taken/3}")
