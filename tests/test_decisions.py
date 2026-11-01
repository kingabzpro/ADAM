import unittest

from adam.decisions import Decision, consensus


def decision(agent, signal="BUY", **kwargs):
    probabilities = {"BUY": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1},
                     "HOLD": {"BUY": 0.1, "HOLD": 0.7, "SELL": 0.2},
                     "SELL": {"BUY": 0.1, "HOLD": 0.2, "SELL": 0.7}}[signal]
    return Decision(ticker="AAPL", agent=agent, signal=signal,
                    probabilities=probabilities,
                    summary="Evidence", status="completed", **kwargs)


class DecisionTests(unittest.TestCase):
    def test_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            Decision(ticker="AAPL", agent="market", signal="BUY",
                     probabilities={"BUY": 1.2, "HOLD": 0, "SELL": 0},
                     summary="", status="completed")

    def test_majority_and_disagreement(self):
        result = consensus([decision("market"), decision("forecast"), decision("news", "SELL")])
        self.assertEqual(result.signal, "BUY")
        self.assertEqual(result.counts, {"BUY": 2, "HOLD": 0, "SELL": 1})
        self.assertTrue(result.disagreement)

    def test_tie_and_no_votes(self):
        self.assertEqual(consensus([decision("market"), decision("news", "SELL")]).signal, "HOLD")
        self.assertIsNone(consensus([]).signal)

    def test_missing_and_unreliable_evidence_abstain(self):
        result = consensus([decision("market", reliable=False), decision("news", stale=True)])
        self.assertIsNone(result.signal)
        self.assertEqual(result.abstained, ("market", "news"))

    def test_hold_majority_resolves_directionally(self):
        hold = {'BUY': .2, 'HOLD': .6, 'SELL': .2}
        result = consensus([decision('market', 'HOLD').model_copy(update={'probabilities': hold}),
                            decision('forecast', 'HOLD').model_copy(update={'probabilities': hold}),
                            decision('technical', 'HOLD').model_copy(update={'probabilities': hold}),
                            decision('fundamentals', 'BUY')])
        self.assertEqual(result.signal, 'BUY')
        self.assertEqual(result.counts, {'BUY': 1, 'HOLD': 3, 'SELL': 0})
        self.assertTrue(result.disagreement)

    def test_margin_requires_a_directional_lead(self):
        lean = {'BUY': .4, 'HOLD': .3, 'SELL': .3}
        voters = lambda: [decision('market', 'HOLD').model_copy(update={'probabilities': lean}),
                          decision('technical', 'HOLD').model_copy(update={'probabilities': lean})]
        self.assertEqual(consensus(voters()).signal, 'BUY')
        self.assertEqual(consensus(voters(), margin=.2).signal, 'HOLD')

    def test_rejects_invalid_margin(self):
        with self.assertRaises(ValueError):
            consensus([decision('market')], margin=1.0)
        with self.assertRaises(ValueError):
            consensus([decision('market')], margin=-.1)

    def test_duplicate_agents_and_mixed_tickers_rejected(self):
        with self.assertRaises(ValueError):
            consensus([decision("market"), decision("market")])
        other = decision("news").model_copy(update={"ticker": "MSFT"})
        with self.assertRaises(ValueError):
            consensus([decision("market"), other])

    def test_weighted_vote(self):
        result = consensus([decision("market"), decision("news", "SELL")], weights={"market": 1, "news": 2})
        self.assertEqual(result.signal, "SELL")
        with self.assertRaises(ValueError):
            consensus([decision("market")], weights={})
