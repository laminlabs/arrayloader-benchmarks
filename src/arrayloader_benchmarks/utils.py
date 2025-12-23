from __future__ import annotations

import time

from tqdm import tqdm


def benchmark_loader(loader, n_samples, batch_size) -> tuple[float, float]:
    num_iter = n_samples // batch_size
    loader_iter = loader.__iter__()

    start_time = time.time()
    actual_iter = 0
    for i, _batch in tqdm(enumerate(loader_iter), total=num_iter):
        actual_iter += 1
        if i == num_iter:
            break

    execution_time = time.time() - start_time
    time_per_sample = (1e6 * execution_time) / (actual_iter * batch_size)
    print(f"time per sample: {time_per_sample:.2f} μs")
    samples_per_sec = actual_iter * batch_size / execution_time
    print(f"samples per sec: {samples_per_sec:.2f} samples/sec")

    return samples_per_sec, execution_time
