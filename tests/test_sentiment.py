import unittest
from unittest.mock import MagicMock

from adam.sentiment import sentiment_evidence


class SentimentTests(unittest.TestCase):
    def test_scores_articles_and_aggregates_probabilities(self):
        classifier = MagicMock(return_value=[
            [{"label": "positive", "score": 0.8}, {"label": "negative", "score": 0.1}, {"label": "neutral", "score": 0.1}],
            [{"label": "negative", "score": 0.6}, {"label": "positive", "score": 0.1}, {"label": "neutral", "score": 0.3}],
        ])
        result = sentiment_evidence(["Profits rose.", "Losses increased."], classifier)
        classifier.assert_called_once_with(["Profits rose.", "Losses increased."], top_k=None, truncation=True)
        self.assertEqual([item["label"] for item in result["articles"]], ["positive", "negative"])
        self.assertAlmostEqual(result["mean_scores"]["positive"], 0.45)
        self.assertAlmostEqual(result["mean_scores"]["negative"], 0.35)

    def test_empty_text_and_malformed_model_output_are_rejected(self):
        classifier = MagicMock()
        for articles in ([], ["  "]):
            with self.assertRaises(ValueError):
                sentiment_evidence(articles, classifier)
        classifier.return_value = [[{"label": "positive", "score": 1.0}]]
        with self.assertRaises(ValueError):
            sentiment_evidence(["headline"], classifier)
