import unittest

import httpx

from adam.health import ollama_health


class HealthTests(unittest.IsolatedAsyncioTestCase):
    def test_dashboard_messages_include_next_step(self):
        from adam.health import health_message
        self.assertIn('adam-setup', health_message({'status': 'missing_model', 'version': '0.40.13', 'model': 'd3:2b'}))
        self.assertIn('restart', health_message({'status': 'outdated', 'version': '0.39.0', 'model': 'd3:2b'}).lower())
        self.assertIn('unavailable', health_message({'status': 'unavailable', 'version': None, 'model': 'd3:2b'}).lower())

    async def test_ready_requires_recent_server_and_local_model(self):
        def handler(request):
            if request.url.path == '/api/version':
                return httpx.Response(200, json={'version': '0.40.13'})
            return httpx.Response(200, json={'models': [{'name': 'd3:2b'}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await ollama_health(client)
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['version'], '0.40.13')
        self.assertEqual(result['model'], 'd3:2b')

    async def test_missing_model_old_server_and_network_failures_are_clear(self):
        for version, models, status in [('0.40.13', [], 'missing_model'), ('0.39.0', [], 'outdated')]:
            def handler(request):
                return httpx.Response(200, json={'version': version} if request.url.path == '/api/version' else {'models': models})
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                self.assertEqual((await ollama_health(client))['status'], status)
        async def offline(request):
            raise httpx.ConnectError('unavailable', request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(offline)) as client:
            self.assertEqual((await ollama_health(client))['status'], 'unavailable')
