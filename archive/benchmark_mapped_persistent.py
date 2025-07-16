import lamindb as ln

ln.transform.stem_uid = "JaQuNeQiKd9P"
ln.transform.version = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
import psutil
import time
import gc

ln.track()

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

results = {"MappedCollection": {}}
for epoch in range(5):
    for k in results:
        results[k][f"epoch_{epoch}"] = {}

torch.ones(2).cuda()

# MappedCollection        

dataset = collection_h5ads.mapped(join=None, parallel=True)

num_workers = psutil.cpu_count() - 1 # 7

loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=num_workers,
    persistent_workers=True,
    pin_memory=True,
    worker_init_fn=dataset.torch_worker_init_fn,
    drop_last=True
)

print("MappedCollection")
for epoch in range(5):
    samples_per_sec, time_per_sample, batch_times = benchmark(loader)
    results["MappedCollection"][f"epoch_{epoch}"]["time_per_sample"] = time_per_sample
    results["MappedCollection"][f"epoch_{epoch}"]["samples_per_sec"] = samples_per_sec
    results["MappedCollection"][f"epoch_{epoch}"]["batch_times"] = batch_times

# Save results
    
with open("mapped_epochs_persistent.json", mode="w") as file:
    file.write(json.dumps(results))

ln.Artifact(
    "mapped_epochs_persistent.json", 
    description="Results of MappedCollection persistent workers for 5 epochs"
).save()