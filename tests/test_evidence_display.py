import unittest

from adam.presentation import evidence_summary, forecast_chart, vote_summary
from test_decisions import decision


class EvidenceDisplayTests(unittest.TestCase):
    def test_vote_summary_shows_counts_disagreement_and_abstentions(self):
        result = vote_summary([decision('market'), decision('forecast'), decision('news', 'SELL'),
                               decision('technical', stale=True)])
        self.assertEqual(result['signal'], 'BUY')
        self.assertEqual(result['counts'], {'BUY': 2, 'HOLD': 0, 'SELL': 1})
        self.assertTrue(result['disagreement'])
        self.assertEqual(result['abstained'], ['technical'])

    def test_forecast_chart_shows_three_bands_in_percent(self):
        evidence = {'horizons': [1, 2], 'lower_returns': [-.1, -.2],
                    'median_returns': [.01, .02], 'upper_returns': [.1, .2]}
        chart = forecast_chart(evidence)
        self.assertEqual(chart['xAxis']['data'], [1, 2])
        self.assertEqual(chart['series'][0]['data'], [-10., -20.])
        self.assertEqual(chart['series'][1]['data'], [1., 2.])
        self.assertEqual(chart['series'][2]['data'], [10., 20.])
        self.assertIsNone(forecast_chart(evidence | {'lower_returns': []}))
        self.assertIsNone(forecast_chart(evidence | {'upper_returns': [float('nan'), .2]}))
        self.assertIsNone(forecast_chart(evidence | {'upper_returns': [1e308, 1e308]}))

    def test_evidence_summary_pairs_facts_with_a_plain_reading(self):
        summary = evidence_summary({'market': {'as_of': '2026-01-01', 'last_close': 100., 'annualized_volatility': .2},
                                    'technical': {'rsi_14': 40., 'sma_20': 98.},
                                    'fundamentals': {'metrics': {'revenue_growth': .1}, 'freshness': 'current'},
                                    'news': {'basis': 'headlines', 'headlines': [{'title': 'A'}]}})
        self.assertIn('100.00', summary['market']['facts'])
        self.assertIn('unremarkable', summary['market']['meaning'])
        self.assertIn('40.0', summary['technical']['facts'])
        self.assertIn('above', summary['technical']['meaning'])
        self.assertIn('10.0%', summary['fundamentals']['facts'])
        self.assertIn('headline', summary['news']['facts'])
        self.assertNotIn('forecast', summary)

    def test_news_summary_reports_counts_and_sentiment(self):
        summary = evidence_summary({'news': {'basis': 'articles', 'articles': [
            {'title': 'A', 'sentiment': {'label': 'positive'}},
            {'title': 'B', 'sentiment': {'label': 'positive'}},
            {'title': 'C', 'sentiment': {'label': 'negative'}}]}})
        self.assertEqual(summary['news']['facts'], '3 articles scored · 2 positive, 1 negative')
        self.assertIn('lean positive', summary['news']['meaning'])
