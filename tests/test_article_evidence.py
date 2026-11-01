import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock
from urllib.parse import quote

import httpx

from adam.news import article_evidence, compact_news_evidence, news_evidence


class ArticleEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_to_headlines_and_compacts_article_evidence(self):
        from unittest.mock import patch
        headlines = {'basis': 'headlines', 'headlines': [], 'mean_scores': None}
        with patch('adam.news.article_evidence', side_effect=httpx.ConnectError('offline')), \
             patch('adam.news.headline_evidence', return_value=headlines):
            result = await news_evidence('AAPL', object(), object())
        self.assertIs(result, headlines)
        article = {'title': 'Results', 'summary': 'Revenue rose', 'published': '2026-10-01',
                   'sentiment': {'label': 'positive'}, 'url': 'https://coincentral.com/a'}
        compact = compact_news_evidence({'basis': 'articles', 'articles': [article],
                                         'mean_scores': {'positive': .8, 'neutral': .1, 'negative': .1}})
        self.assertEqual(compact['basis'], 'articles')
        self.assertNotIn('url', compact['articles'][0])
        self.assertEqual(compact['articles'][0]['summary'], 'Revenue rose')

    async def test_scores_relevant_article_from_allowed_publisher_only(self):
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        good = 'https://coincentral.com/markets/aapl-results'
        evil = 'http://127.0.0.1:8080/private'
        feed = f'''<rss><channel>
        <item><title>AAPL results</title><link>http://www.bing.com/news/apiclick.aspx?url={quote(good, safe='')}</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item>
        <item><title>internal</title><link>http://www.bing.com/news/apiclick.aspx?url={quote(evil, safe='')}</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item>
        </channel></rss>'''.encode()
        calls = []
        def respond(request):
            calls.append(str(request.url))
            if request.url.host == 'www.bing.com':
                return httpx.Response(200, content=feed)
            if request.url.path == '/robots.txt':
                return httpx.Response(200, text='User-agent: *\nDisallow: /out/\n')
            return httpx.Response(200, text='<article><p>AAPL revenue rose during the quarter amid strong product demand.</p><p>The company reported significant growth in its main markets.</p></article>', headers={'content-type': 'text/html'})
        classifier = MagicMock(return_value=[[{'label': 'positive', 'score': .8}, {'label': 'negative', 'score': .1}, {'label': 'neutral', 'score': .1}]])
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await article_evidence('AAPL', client, classifier, now=now)
        self.assertEqual(result['basis'], 'articles')
        self.assertEqual(len(result['articles']), 1)
        self.assertEqual(result['articles'][0]['url'], good)
        self.assertEqual(result['articles'][0]['sentiment']['label'], 'positive')
        self.assertFalse(any('127.0.0.1' in url for url in calls))
        self.assertEqual(len(calls), 3)

    async def test_robots_disallow_prevents_article_download(self):
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        url = quote('https://blockonomi.com/out/private', safe='')
        feed = f'<rss><channel><item><title>AAPL</title><link>http://www.bing.com/news/apiclick.aspx?url={url}</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>'.encode()
        calls = []
        def respond(request):
            calls.append(request.url.path)
            if request.url.host == 'www.bing.com':
                return httpx.Response(200, content=feed)
            if request.url.path == '/robots.txt':
                return httpx.Response(200, text='User-agent: *\nDisallow: /out/\n')
            self.fail('Disallowed article was requested')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await article_evidence('AAPL', client, MagicMock(), now=now)
        self.assertEqual(result['articles'], [])
        self.assertEqual(calls, ['/news/search', '/robots.txt'])

    async def test_robots_longer_disallow_overrides_general_allow(self):
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        link = quote('https://coincentral.com/private/article', safe='')
        feed = f'<rss><channel><item><title>AAPL</title><link>http://www.bing.com/news/apiclick.aspx?url={link}</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>'.encode()
        def respond(request):
            if request.url.host == 'www.bing.com':
                return httpx.Response(200, content=feed)
            if request.url.path == '/robots.txt':
                return httpx.Response(200, text='User-agent: *\nAllow: /\nDisallow: /private/\n')
            self.fail('More specific disallow was ignored')
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await article_evidence('AAPL', client, MagicMock(), now=now)
        self.assertEqual(result['articles'], [])

    async def test_new_publisher_requires_readable_robots_and_allows_article(self):
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        link = quote('https://blockonomi.com/aapl-results', safe='')
        feed = f'<rss><channel><item><title>AAPL</title><link>http://www.bing.com/news/apiclick.aspx?url={link}</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>'.encode()
        scores = [[{'label': 'positive', 'score': .8}, {'label': 'negative', 'score': .1}, {'label': 'neutral', 'score': .1}]]
        for status, rules, expected in [(200, 'User-agent: *\nDisallow: /out/\n', 1),
                                        (503, '', 0), (200, 'User-agent: *\nDisallow: /*.php\n', 0)]:
            def respond(request):
                if request.url.host == 'www.bing.com':
                    return httpx.Response(200, content=feed)
                if request.url.path == '/robots.txt':
                    return httpx.Response(status, text=rules)
                return httpx.Response(200, text='<article><p>AAPL revenue rose during the latest fiscal quarter amid growing demand.</p><p>Analysts reviewed the company results after the announcement.</p></article>', headers={'content-type': 'text/html'})
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
                result = await article_evidence('AAPL', client, MagicMock(return_value=scores), now=now)
            self.assertEqual(len(result['articles']), expected)

    async def test_rejects_redirects_and_unrelated_articles(self):
        now = datetime(2026, 10, 2, tzinfo=timezone.utc)
        link = quote('https://coincentral.com/a', safe='')
        feed = f'<rss><channel><item><title>AAPL</title><link>http://www.bing.com/news/apiclick.aspx?url={link}</link><pubDate>Thu, 01 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>'.encode()
        def respond(request):
            if request.url.host == 'www.bing.com':
                return httpx.Response(200, content=feed)
            if request.url.path == '/robots.txt':
                return httpx.Response(200, text='User-agent: *\nAllow: /\n')
            return httpx.Response(302, headers={'location': 'http://127.0.0.1/'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            result = await article_evidence('AAPL', client, MagicMock(), now=now)
        self.assertEqual(result['articles'], [])
        self.assertIsNone(result['mean_scores'])

        def unrelated(request):
            if request.url.host == 'www.bing.com':
                return httpx.Response(200, content=feed)
            if request.url.path == '/robots.txt':
                return httpx.Response(200, text='User-agent: *\nAllow: /\n')
            return httpx.Response(200, text='<article><p>Microsoft reported strong growth in cloud revenue during the quarter.</p><p>Analysts discussed its outlook for the next fiscal year.</p></article>', headers={'content-type': 'text/html'})
        classifier = MagicMock()
        async with httpx.AsyncClient(transport=httpx.MockTransport(unrelated)) as client:
            result = await article_evidence('AAPL', client, classifier, now=now)
        self.assertEqual(result['articles'], [])
        classifier.assert_not_called()
