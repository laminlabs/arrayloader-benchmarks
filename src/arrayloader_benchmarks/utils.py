from __future__ import annotations

import time

from tqdm import tqdm


def benchmark_loader(loader, n_samples, batch_size):
    num_iter = n_samples // batch_size
    loader_iter = loader.__iter__()

    start_time = time.time()
    batch_times = []
    batch_time = time.time()
    total_time = time.time()
    for i, _batch in tqdm(enumerate(loader_iter), total=num_iter):
        batch_times.append(time.time() - batch_time)
        batch_time = time.time()
        if i == num_iter:
            break

    total_time = time.time() - total_time

    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (num_iter * batch_size)
    print(f"time per sample: {time_per_sample:.2f} μs")
    samples_per_sec = num_iter * batch_size / execution_time
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")

    return samples_per_sec, time_per_sample, batch_times, total_time
