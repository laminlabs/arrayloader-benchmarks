import lamindb as ln

ln.transform.stem_uid = "Md9ea0bLFozt"
ln.transform.version = "1"

import merlin.io
from merlin.dataloader.torch import Loader
from merlin.schema import ColumnSchema, Schema
from merlin.dtypes import float32
import cellxgene_census
import cellxgene_census.experimental.ml as census_ml
import tiledbsoma as soma
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import psutil
import time
import gc

ln.track()

artifact_parquet = ln.Artifact.filter(uid="CQzAZX2kAfVVehbOB6QK").one()

merlin_benchmark = Path("./merlin_benchmark")
# artifact_parquet.stage() or artifact_parquet.load() don't work because it is a folder
if not merlin_benchmark.exists():
    # not tracked
    artifact_parquet.path.download_to(merlin_benchmark, print_progress=True, recursive=True)

collection_h5ads = ln.Collection.filter(uid="VwxM0HNDtEcNjJEKwYqO").one()
# We need collection.stage()
for artifact in collection_h5ads.artifacts:
    artifact.stage()

BATCH_SIZE = 1024

def benchmark(loader, n_samples = None):    
    loader_iter = loader.__iter__()
    # exclude first batch from benchmark as this includes the setup time
    batch = next(loader_iter)
    
    num_iter = n_samples // BATCH_SIZE if n_samples is not None else None
    
    start_time = time.time()
    
    batch_times = []
    batch_time = time.time()
    
    total = num_iter if num_iter is not None else len(loader_iter)
    for i, batch in tqdm(enumerate(loader_iter), total=total):
        X = batch["x"] if isinstance(batch, dict) else batch[0] 
        # for pytorch DataLoader
        # Merlin sends to cuda by default
        if hasattr(X, "is_cuda") and not X.is_cuda:
            X = X.cuda()
        
        if num_iter is not None and i == num_iter:
            break
        if i % 10 == 0:
            gc.collect()
        
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
    
    execution_time = time.time() - start_time
    gc.collect()
    
    time_per_sample = (1e6 * execution_time) / (total * BATCH_SIZE)
    print(f'time per sample: {time_per_sample:.2f} μs')
    samples_per_sec = total * BATCH_SIZE / execution_time
    print(f'samples per sec: {samples_per_sec:.2f} samples/sec')
    
    return samples_per_sec, time_per_sample, batch_times

results = {"Merlin": {}, "MappedCollection": {}, "cellxgene_census": {}}
for epoch in range(3):
    for k in results:
        results[k][f"epoch_{epoch}"] = {}

torch.ones(2).cuda()

# Merlin

dataset = merlin.io.Dataset(
    merlin_benchmark, 
    engine="parquet", 
    part_size="100MB",  # use 100MB for a chunk size of 1024
    schema=Schema([
        ColumnSchema(
            "X", dtype=float32, 
            is_list=True, is_ragged=False, 
            properties={"value_count": {"max": 20000}}
        )
    ])
)

loader = Loader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    parts_per_chunk=1,
    drop_last=True,
).epochs(1)

print("Merlin")
for epoch in range(3):
    samples_per_sec, time_per_sample, batch_times = benchmark(loader)
    results["Merlin"][f"epoch_{epoch}"]["time_per_sample"] = time_per_sample
    results["Merlin"][f"epoch_{epoch}"]["samples_per_sec"] = samples_per_sec
    results["Merlin"][f"epoch_{epoch}"]["batch_times"] = batch_times

# MappedCollection        

dataset = collection_h5ads.mapped(join=None, parallel=True)

num_workers = psutil.cpu_count() - 1 # 7

loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=num_workers,
    worker_init_fn=dataset.torch_worker_init_fn,
    drop_last=True
)

print("MappedCollection")
for epoch in range(3):
    samples_per_sec, time_per_sample, batch_times = benchmark(loader)
    results["MappedCollection"][f"epoch_{epoch}"]["time_per_sample"] = time_per_sample
    results["MappedCollection"][f"epoch_{epoch}"]["samples_per_sec"] = samples_per_sec
    results["MappedCollection"][f"epoch_{epoch}"]["batch_times"] = batch_times

# cellxgene census
    
census = cellxgene_census.open_soma()

reference = ln.Collection.filter(uid="1gsdckxvOvIjQgeDVS1F").one().reference
query_collection_id = f"collection_id == '{reference}'"
datasets =(census["census_info"]["datasets"]
           .read(column_names=["dataset_id"], value_filter=query_collection_id)
           .concat().to_pandas())["dataset_id"].tolist()
query_datasets = "dataset_id in " + str(datasets)

experiment = census["census_data"]["homo_sapiens"]
experiment_datapipe = census_ml.ExperimentDataPipe(
    experiment,
    measurement_name="RNA",
    X_name="raw",
    obs_query=soma.AxisQuery(value_filter=query_datasets),
    var_query=soma.AxisQuery(coords=(slice(20000-1),)),
    batch_size=BATCH_SIZE,
    shuffle=True,
    soma_chunk_size=10000,
)

loader = census_ml.experiment_dataloader(experiment_datapipe)

print("cellxgene_census")
for epoch in range(3):
    samples_per_sec, time_per_sample, batch_times = benchmark(loader, n_samples=experiment_datapipe.shape[0])
    results["cellxgene_census"][f"epoch_{epoch}"]["time_per_sample"] = time_per_sample
    results["cellxgene_census"][f"epoch_{epoch}"]["samples_per_sec"] = samples_per_sec
    results["cellxgene_census"][f"epoch_{epoch}"]["batch_times"] = batch_times

census.close()

# Save results
    
with open("merlin_mapped_epochs.json", mode="w") as file:
    file.write(json.dumps(results))

ln.Artifact(
    "merlin_mapped_epochs.json", 
    description="Results of Merlin and MappedCollection benchmarking for 3 epochs"
).save()