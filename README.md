# `arrayloader-benchmarks`: Data loader benchmarks for scRNA-seq counts et al.

_A collaboration between scverse, Lamin, and anyone interested in contributing!_

This repository contains benchmarking scripts & utilities for scRNA-seq data loaders and allows to collaboratively contribute new benchmarking results.

## Quickstart

Typical calls of the main benchmarking script are:

```bash
git clone https://github.com/laminlabs/arrayloader-benchmarks
cd arrayloader-benchmarks
uv pip install --system -e ".[scdataset,annbatch]"  # provide tools you'd like to install
cd scripts
python run_loading_benchmark_on_collection.py annbatch   # run annbatch on collection Tahoe100M_tiny, n_datasets = 1
python run_loading_benchmark_on_collection.py MappedCollection   # run MappedCollection
python run_loading_benchmark_on_collection.py scDataset   # run scDataset
python run_loading_benchmark_on_collection.py annbatch --n_datasets -1  # run against all datasets in the collection
python run_loading_benchmark_on_collection.py annbatch --collection Tahoe100M --n_datasets -1  # run against the full 100M cells
python run_loading_benchmark_on_collection.py annbatch --collection Tahoe100M --n_datasets 1  # run against the the first dataset, 2M cells
python run_loading_benchmark_on_collection.py annbatch --collection Tahoe100M --n_datasets 5  # run against the the first dataset, 10M cells
```

You can choose between different benchmarking [dataset collections](https://lamin.ai/laminlabs/arrayloader-benchmarks/collections).

<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/b539b13a-9b50-4f66-9b51-16d32fd8566b" />
<br>
<br>

When running the script, [parameters and results](https://lamin.ai/laminlabs/arrayloader-benchmarks/artifact/0EiozNVjberZTFHa) are automatically tracked in a parquet file, along with source code, run environment, and input and output datasets.

<img width="1000" height="904" alt="image" src="https://github.com/user-attachments/assets/60c3262f-1bdc-44a4-a488-4784918a6905" />
<br>
<br>

Note: A previous version of this repo contained the benchmarking scripts accompanying the 2024 blog post: [lamin.ai/blog/arrayloader-benchmarks](https://lamin.ai/blog/arrayloader-benchmarks).
