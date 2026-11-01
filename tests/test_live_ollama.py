import os
import unittest

import httpx

from adam.benchmark import benchmark_d3
from adam.d3 import D3Client


@unittest.skipUnless(os.getenv('ADAM_TEST_LIVE_OLLAMA') == '1', 'set ADAM_TEST_LIVE_OLLAMA=1 to test the local D3 model')
class LiveOllamaTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_model_and_concurrent_benchmark(self):
        async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
            model = D3Client(client)
            decision = await model.assess('AAPL', 'market', {'example_return': 0.01})
            self.assertIn(decision.signal, {'BUY', 'HOLD', 'SELL'})
            self.assertAlmostEqual(sum(decision.probabilities.values()), 1.0, places=5)
            report = await benchmark_d3(model, requests=3, concurrency=2)
            self.assertEqual(report['succeeded'], 3)
            self.assertEqual(report['failed'], 0)
            self.assertIsNotNone(report['latency_p95_ms'])
