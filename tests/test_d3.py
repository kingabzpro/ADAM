import json
import unittest

import httpx

from adam.d3 import D3Client


class D3Tests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_request_and_response(self):
        def handler(request):
            self.assertEqual(request.url.path, "/v1/systemone")
            payload = json.loads(request.content)
            self.assertEqual(payload["questions"]["signal"]["type"], "choice")
            self.assertEqual(set(payload["questions"]["signal"]["criteria"]), {"BUY", "HOLD", "SELL"})
            return httpx.Response(200, json={"answers": {"signal": {"choice": "BUY", "probabilities": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1}}}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            result = await D3Client(http).assess("AAPL", "market", {"momentum": 0.05})
        self.assertEqual(result.signal, "BUY")
        self.assertEqual(result.probabilities["BUY"], 0.7)

    async def test_invalid_response_and_http_errors_propagate(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"answers": {}}))) as http:
            with self.assertRaises(ValueError):
                await D3Client(http).assess("AAPL", "market", {"price": 100})
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as http:
            with self.assertRaises(httpx.HTTPStatusError):
                await D3Client(http).assess("AAPL", "market", {"price": 100})

    def test_rejects_remote_endpoint(self):
        with self.assertRaises(ValueError):
            D3Client(httpx.AsyncClient(), endpoint="https://example.com")
