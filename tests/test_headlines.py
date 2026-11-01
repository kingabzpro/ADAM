import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from adam.news import headline_evidence

NOW = datetime(2026, 10, 2, tzinfo=timezone.utc)


class HeadlineEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_filters_stale_and_duplicate_headlines(self):
        items = [
            {'title': 'AAPL profits rise', 'url': 'https://example.com/1', 'published': '2026-10-01T12:00:00+00:00', 'source': 'Example'},
            {'title': ' AAPL profits rise ', 'url': 'https://example.com/2', 'published': '2026-10-01T12:00:00+00:00', 'source': 'Example'},
            {'title': 'Old news', 'url': 'https://example.com/3', 'published': '2026-01-01T00:00:00+00:00', 'source': 'Example'},
            {'title': 'No date', 'url': 'https://example.com/4', 'published': None, 'source': 'Example'},
        ]
        classifier = MagicMock(return_value=[[{'label': 'positive', 'score': .8}, {'label': 'negative', 'score': .1}, {'label': 'neutral', 'score': .1}]])
        with patch('adam.news.discover_news', return_value=items):
            result = await headline_evidence('AAPL', object(), classifier, now=NOW)
        self.assertEqual(result['basis'], 'headlines')
        self.assertEqual(len(result['headlines']), 1)
        self.assertEqual(result['headlines'][0]['sentiment']['label'], 'positive')
        self.assertEqual(result['excluded_count'], 3)
        classifier.assert_called_once()

    async def test_no_recent_headlines_abstains(self):
        classifier = MagicMock()
        with patch('adam.news.discover_news', return_value=[]):
            result = await headline_evidence('AAPL', object(), classifier, now=NOW)
        self.assertEqual(result['headlines'], [])
        self.assertIsNone(result['mean_scores'])
        classifier.assert_not_called()
