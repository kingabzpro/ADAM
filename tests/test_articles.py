import unittest
from unittest.mock import MagicMock

from adam.news import extract_article, analyze_articles


class ArticleTests(unittest.TestCase):
    def test_scores_multiple_chunks_and_gives_each_article_equal_weight(self):
        classifier = MagicMock(return_value=[
            [{'label': 'positive', 'score': .8}, {'label': 'neutral', 'score': .1}, {'label': 'negative', 'score': .1}],
            [{'label': 'positive', 'score': .2}, {'label': 'neutral', 'score': .1}, {'label': 'negative', 'score': .7}],
            [{'label': 'positive', 'score': .2}, {'label': 'neutral', 'score': .7}, {'label': 'negative', 'score': .1}],
        ])
        result = analyze_articles([{'text': 'A' * 900}, {'text': 'Small article.'}], classifier)
        self.assertEqual(result['articles'][0]['sentiment_chunks'], 2)
        self.assertAlmostEqual(result['articles'][0]['sentiment']['scores']['positive'], .5)
        self.assertAlmostEqual(result['mean_scores']['positive'], .35)
        self.assertEqual(len(classifier.call_args.args[0]), 3)

    def test_extracts_bounded_financial_highlights_from_article(self):
        from adam.news import financial_highlights
        text = 'AAPL introduced a new phone. Revenue grew 20% to $40 billion. Analysts praised the results. Cash flow reached $8 billion.'
        self.assertEqual(financial_highlights(text), ['Revenue grew 20% to $40 billion.', 'Cash flow reached $8 billion.'])

    def test_article_text_is_capped_and_marked(self):
        scores = [{'label': 'positive', 'score': .8}, {'label': 'neutral', 'score': .1}, {'label': 'negative', 'score': .1}]
        classifier = MagicMock(return_value=[scores] * 25)
        result = analyze_articles([{'text': 'A' * 25000}], classifier)
        self.assertTrue(result['articles'][0]['text_truncated'])
        self.assertEqual(result['articles'][0]['sentiment_chunks'], 25)

    def test_extracts_main_text_and_marks_empty(self):
        html = "<html><body><article><h1>Results</h1><p>Revenue rose by twenty percent during the quarter.</p><p>Management said demand remained strong in its largest markets.</p></article></body></html>"
        result = extract_article(html)
        self.assertIn("Revenue rose", result)
        self.assertIsNone(extract_article("<html><body></body></html>"))

    def test_only_readable_articles_are_scored_and_duplicates_excluded(self):
        classifier = MagicMock(return_value=[[{"label": "positive", "score": 0.8}, {"label": "neutral", "score": 0.1}, {"label": "negative", "score": 0.1}]])
        records = [{"url": "https://example.com/a", "text": "Revenue rose."},
                   {"url": "https://example.com/b", "text": None},
                   {"url": "https://example.com/c", "text": "Revenue rose."}]
        result = analyze_articles(records, classifier)
        self.assertEqual(result["articles"][0]["sentiment"]["label"], "positive")
        self.assertEqual(result["articles"][1]["status"], "unavailable")
        self.assertEqual(result["articles"][2]["status"], "duplicate")
        self.assertEqual(result["scored_count"], 1)
        classifier.assert_called_once()

    def test_no_readable_articles_abstains(self):
        classifier = MagicMock()
        result = analyze_articles([{"url": "https://example.com", "text": None}], classifier)
        self.assertIsNone(result["mean_scores"])
        classifier.assert_not_called()
