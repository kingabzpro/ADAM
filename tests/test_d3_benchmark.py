import json
import unittest

import httpx

from adam.benchmark import benchmark_d3
from adam.d3 import D3Client


class D3BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_and_memory_are_reported_separately(self):
        calls = []
        def handler(request):
            calls.append(request.url.path)
            if request.url.path == '/api/ps':
                return httpx.Response(200, json={'models': [{'name': 'd3:2b', 'size': 3000, 'size_vram': 2000}]})
            payload = json.loads(request.content)
            self.assertEqual(payload['model'], 'd3:2b')
            return httpx.Response(200, json={'answers': {'signal': {'choice': 'HOLD', 'probabilities': {'BUY': 0.2, 'HOLD': 0.6, 'SELL': 0.2}}}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await benchmark_d3(D3Client(client, endpoint='http://127.0.0.1:11434', model='d3:2b'), requests=3, concurrency=2)
        self.assertEqual(calls.count('/v1/systemone'), 4)
        self.assertEqual(result['succeeded'], 3)
        self.assertEqual(result['model_memory_bytes'], 3000)
        self.assertEqual(result['model_vram_bytes'], 2000)
        self.assertGreaterEqual(result['warmup_ms'], 0)

    async def test_warmup_failure_is_not_reported_as_success(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await benchmark_d3(D3Client(client, endpoint='http://127.0.0.1:11434', model='d3:2b'), requests=1)

    def test_cli_emits_json_and_reports_failures(self):
        import io
        from unittest.mock import AsyncMock, patch
        from adam.benchmark import main
        report = {'succeeded': 2, 'failed': 0, 'latency_p50_ms': 1.0}
        with patch('sys.argv', ['adam-benchmark', '--requests', '2', '--concurrency', '1']), patch('adam.benchmark.benchmark_d3', new_callable=AsyncMock, return_value=report), patch('sys.stdout', new_callable=io.StringIO) as output:
            main()
        self.assertEqual(json.loads(output.getvalue())['succeeded'], 2)

    def test_cli_returns_failure_if_any_request_fails(self):
        import io
        from unittest.mock import AsyncMock, patch
        from adam.benchmark import main
        report = {'succeeded': 1, 'failed': 1}
        with patch('sys.argv', ['adam-benchmark']), patch('adam.benchmark.benchmark_d3', new_callable=AsyncMock, return_value=report), patch('sys.stdout', new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as error:
                main()
        self.assertEqual(error.exception.code, 1)
