import unittest

import httpx

from adam.news import discover_news, parse_feed

FEED = b'''<rss><channel>
<item><title>Company reports growth</title><link>https://news.google.com/articles/one</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate><source>Example</source></item>
<item><title>Duplicate</title><link>https://news.google.com/articles/one</link></item>
<item><title>Missing link</title></item>
</channel></rss>'''


class NewsTests(unittest.IsolatedAsyncioTestCase):
    def test_parses_and_deduplicates_feed(self):
        items = parse_feed(FEED)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["title"], "Company reports growth")
        self.assertEqual(items[0]["published"], "2026-10-01T12:00:00+00:00")
        self.assertEqual(items[0]["source"], "Example")

    def test_rejects_invalid_xml(self):
        with self.assertRaises(ValueError):
            parse_feed(b"not XML")

    async def test_searches_by_ticker_and_propagates_failure(self):
        def handler(request):
            self.assertEqual(request.url.host, "news.google.com")
            self.assertEqual(request.url.params["q"], "AAPL stock")
            return httpx.Response(200, content=FEED)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            articles = await discover_news(" aapl ", client)
        self.assertEqual(len(articles), 1)
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503))) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                await discover_news("AAPL", client)
        async with httpx.AsyncClient() as client:
            with self.assertRaises(ValueError):
                await discover_news(" ", client)
