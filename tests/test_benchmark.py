import asyncio
import unittest

from adam.benchmark import benchmark


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_ticker_benchmark_limits_workers_and_reports_partial_results(self):
        from types import SimpleNamespace
        from adam.benchmark import benchmark_tickers
        active = maximum = 0
        class Coordinator:
            async def analyze(self, ticker, evidence):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                await asyncio.sleep(.01)
                active -= 1
                return SimpleNamespace(agents={'market': 'failed' if ticker == 'BAD' else 'completed',
                                               'technical': 'completed', 'fundamentals': 'completed'},
                                       errors={'market': 'unavailable'} if ticker == 'BAD' else {})
        result = await benchmark_tickers(Coordinator(), ['AAPL', 'BAD', 'MSFT'], concurrency=2)
        self.assertEqual(maximum, 2)
        self.assertEqual((result['succeeded'], result['failed']), (2, 1))
        self.assertEqual(result['failures'][0]['ticker'], 'BAD')
        self.assertEqual(result['requests'], 3)
        with self.assertRaises(ValueError):
            await benchmark_tickers(Coordinator(), ['AAPL', 'aapl'], concurrency=2)

    async def test_ticker_benchmark_handles_dozens_of_requests(self):
        from types import SimpleNamespace
        from adam.benchmark import benchmark_tickers
        class Coordinator:
            async def analyze(self, ticker, evidence):
                await asyncio.sleep(0)
                return SimpleNamespace(agents={'market': 'completed', 'technical': 'completed', 'fundamentals': 'completed'}, errors={})
        result = await benchmark_tickers(Coordinator(), [f'T{i}' for i in range(30)], concurrency=4)
        self.assertEqual(result['succeeded'], 30)
        self.assertEqual(len(result['agent_statuses']), 30)

    async def test_limits_concurrency_and_reports_failures(self):
        active = maximum = 0
        async def operation(index):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
                if index == 2:
                    raise RuntimeError('unavailable')
            finally:
                active -= 1
        result = await benchmark(operation, requests=5, concurrency=2)
        self.assertEqual(maximum, 2)
        self.assertEqual(result['succeeded'], 4)
        self.assertEqual(result['failed'], 1)
        self.assertLessEqual(result['latency_p50_ms'], result['latency_p95_ms'])
        self.assertLessEqual(result['latency_p95_ms'], result['latency_p99_ms'])
        self.assertGreater(result['throughput_per_second'], 0)

    async def test_rejects_invalid_options(self):
        async def operation(index):
            return None
        for requests, concurrency in [(0, 1), (1, 0), (True, 1)]:
            with self.subTest(requests=requests, concurrency=concurrency), self.assertRaises(ValueError):
                await benchmark(operation, requests=requests, concurrency=concurrency)

    async def test_all_failed_does_not_report_success_latency(self):
        async def fail(index):
            raise RuntimeError('unavailable')
        result = await benchmark(fail, requests=2)
        self.assertEqual(result['succeeded'], 0)
        self.assertEqual(result['failed'], 2)
        self.assertIsNone(result['latency_p50_ms'])
        self.assertEqual(result['throughput_per_second'], 0)
