# `arrayloader-benchmarks`: Data loader benchmarks for scRNA-seq counts et al.

_A collaboration between scverse, Lamin, and anyone interested in contributing!_

This repository contains benchmarking scripts & utilities for scRNA-seq data loaders and allows to collaboratively contribute new benchmarking results.

A user can choose between different benchmarking dataset collections:

https://lamin.ai/laminlabs/arrayloader-benchmarks/collections

<img width="500" height="481" alt="image" src="https://github.com/user-attachments/assets/b539b13a-9b50-4f66-9b51-16d32fd8566b" />

Typical calls are:

```
python scripts/run_data_loading_benchmark_on_tahoe100m.py annbatch   # run with collection Tahoe100M_tiny, n_datasets = 1
python scripts/run_data_loading_benchmark_on_tahoe100m.py MappedCollection   # run MappedCollection
python scripts/run_data_loading_benchmark_on_tahoe100m.py scDataset   # run scDataset
python scripts/run_data_loading_benchmark_on_tahoe100m.py annbatch --n_datasets -1  # run against all datasets in the collection
python scripts/run_data_loading_benchmark_on_tahoe100m.py annbatch --collection Tahoe100M --n_datasets -1  # run against the full 100M cells
python scripts/run_data_loading_benchmark_on_tahoe100m.py annbatch --collection Tahoe100M --n_datasets 1  # run against the the first dataset, 2M cells
python scripts/run_data_loading_benchmark_on_tahoe100m.py annbatch --collection Tahoe100M --n_datasets 5  # run against the the first dataset, 10M cells
```

Parameters and results for each run are automatically tracked in a parquet file. Source code and datasets are tracked via data lineage.

<img width="1298" height="904" alt="image" src="https://github.com/user-attachments/assets/60c3262f-1bdc-44a4-a488-4784918a6905" />

Results can be downloaded and reproduced from here: https://lamin.ai/laminlabs/arrayloader-benchmarks/artifact/0EiozNVjberZTFHa

Note: A previous version of this repo contained the benchmarking scripts accompanying the 2024 blog post: [lamin.ai/blog/arrayloader-benchmarks](https://lamin.ai/blog/arrayloader-benchmarks).
