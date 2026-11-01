import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from adam.calibration import evaluate_calibration, main
from test_decisions import decision


class CalibrationTests(unittest.TestCase):
    def test_multiclass_scores_and_top_choice_reliability(self):
        first = decision('market').model_copy(update={'probabilities': {'BUY': .8, 'HOLD': .1, 'SELL': .1}})
        second = decision('market').model_copy(update={'probabilities': {'BUY': .6, 'HOLD': .2, 'SELL': .2}})
        result = evaluate_calibration([{'decision': first.model_dump(), 'label': 'BUY'},
                                       {'decision': second.model_dump(), 'label': 'SELL'}], bins=2)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['brier_score'], .55)
        self.assertAlmostEqual(result['log_loss'], -(math.log(.8) + math.log(.2)) / 2)
        self.assertAlmostEqual(result['expected_calibration_error'], .2)
        self.assertEqual(result['bins'][1]['count'], 2)
        self.assertAlmostEqual(result['bins'][1]['mean_confidence'], .7)

    def test_filters_stale_and_rejects_mixed_agents_invalid_labels_and_bad_bins(self):
        valid = {'decision': decision('market').model_dump(), 'label': 'BUY'}
        stale = {'decision': decision('market', stale=True).model_dump(), 'label': 'SELL'}
        result = evaluate_calibration([valid, stale])
        self.assertEqual((result['samples'], result['excluded']), (1, 1))
        for entries, bins in [([valid, {'decision': decision('news').model_dump(), 'label': 'BUY'}], 10),
                              ([valid | {'label': 'MAYBE'}], 10), ([valid], True), ([stale], 10)]:
            with self.subTest(entries=entries, bins=bins), self.assertRaises(ValueError):
                evaluate_calibration(entries, bins=bins)

    def test_zero_probability_is_finite_and_unit_confidence_uses_last_bin(self):
        certain = decision('market').model_copy(update={'probabilities': {'BUY': 1., 'HOLD': 0., 'SELL': 0.}})
        result = evaluate_calibration([{'decision': certain.model_dump(), 'label': 'SELL'}], bins=2)
        self.assertEqual(result['bins'][-1]['count'], 1)
        self.assertEqual(result['brier_score'], 2.)
        self.assertAlmostEqual(result['log_loss'], -math.log(1e-15))
        self.assertEqual(result['expected_calibration_error'], 1.)

    def test_rejects_signal_that_disagrees_with_scores(self):
        incorrect = decision('market', 'SELL')
        with self.assertRaises(ValueError):
            evaluate_calibration([{'decision': incorrect.model_dump(), 'label': 'BUY'}])

    def test_file_command_reads_labeled_json_lines(self):
        example = {'decision': decision('market').model_dump(), 'label': 'BUY'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'examples.jsonl'
            path.write_text(json.dumps(example) + '\n', encoding='utf-8')
            with patch('sys.argv', ['adam-calibrate', str(path)]), patch('sys.stdout', new_callable=__import__('io').StringIO) as output:
                main()
        self.assertEqual(json.loads(output.getvalue())['samples'], 1)
