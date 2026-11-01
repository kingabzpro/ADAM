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


async def benchmark_d3(model, *, requests: int, concurrency: int = 1) -> dict:
    """Benchmark warm D3 requests; memory values are Ollama's model allocation, not process peaks."""
    if type(requests) is not int or requests < 1 or type(concurrency) is not int or concurrency < 1:
        raise ValueError('requests and concurrency must be positive integers')

    async def operation(index):
        await model.assess('BENCHMARK', 'market', {'example_return': 0.01})

    started = time.perf_counter()
    await operation(0)
    warmup_ms = (time.perf_counter() - started) * 1000
    result = await benchmark(operation, requests=requests, concurrency=concurrency)
    result.update(model=model.model, warmup_ms=warmup_ms, model_memory_bytes=None, model_vram_bytes=None)
    try:
        response = await model.http.get(f'{model.endpoint}/api/ps', follow_redirects=False)
        response.raise_for_status()
        loaded = next((item for item in response.json().get('models', [])
                       if item.get('name') == model.model or item.get('model') == model.model), None)
        if loaded is not None:
            result['model_memory_bytes'] = loaded.get('size')
            result['model_vram_bytes'] = loaded.get('size_vram')
    except Exception as error:
        result['memory_error'] = str(error)
    return result


def main():
    import argparse
    import json
    import sys

    import httpx
    from adam.d3 import D3Client

    parser = argparse.ArgumentParser(description='Measure local D3 inference latency and model memory.')
    parser.add_argument('--requests', type=int, default=10)
    parser.add_argument('--concurrency', type=int, default=1)
    args = parser.parse_args()

    async def run():
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            return await benchmark_d3(D3Client(client), requests=args.requests, concurrency=args.concurrency)

    try:
        result = asyncio.run(run())
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as error:
        print(f'D3 benchmark failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
    print(json.dumps(result, indent=2))
    if result['failed']:
        raise SystemExit(1)
