import asyncio
import time

import numpy as np


async def benchmark(operation, *, requests: int, concurrency: int = 1) -> dict:
    """Measure successful operation service time with a bounded asynchronous worker pool."""
    if type(requests) is not int or requests < 1 or type(concurrency) is not int or concurrency < 1:
        raise ValueError('requests and concurrency must be positive integers')
    indices = iter(range(requests))
    latencies = []
    failures = []

    async def worker():
        while (index := next(indices, None)) is not None:
            started = time.perf_counter()
            try:
                await operation(index)
            except Exception as error:
                failures.append({'index': index, 'error': str(error)})
            else:
                latencies.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(requests, concurrency))))
    elapsed = time.perf_counter() - started
    percentiles = np.percentile(latencies, [50, 95, 99]) if latencies else (None, None, None)
    return {'requests': requests, 'concurrency': min(requests, concurrency),
            'succeeded': len(latencies), 'failed': len(failures), 'failures': failures,
            'elapsed_seconds': elapsed, 'throughput_per_second': len(latencies) / elapsed if elapsed else 0.0,
            'latency_p50_ms': float(percentiles[0]) if latencies else None,
            'latency_p95_ms': float(percentiles[1]) if latencies else None,
            'latency_p99_ms': float(percentiles[2]) if latencies else None}
