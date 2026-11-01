import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from adam.snapshots import read_prices, read_fundamentals, save_prices, save_fundamentals
from test_technical import history


class SnapshotTests(unittest.TestCase):
    def test_prices_and_fundamentals_round_trip_with_timezone(self):
        frame = history([100.] * 40)
        frame['Open'] = 100.
        frame.index = frame.index.tz_convert('America/New_York')
        with tempfile.TemporaryDirectory() as directory:
            save_prices(directory, 'aapl', frame)
            save_fundamentals(directory, 'AAPL', {'revenueGrowth': .12, 'mostRecentQuarter': datetime.now(timezone.utc).timestamp(), 'extra': object()})
            loaded, saved_at = read_prices(directory, 'AAPL')
            info, info_saved_at = read_fundamentals(directory, 'AAPL')
        pd.testing.assert_frame_equal(loaded, frame[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float), check_freq=False)
        self.assertEqual(info['revenueGrowth'], .12)
        self.assertNotIn('extra', info)
        self.assertIsNotNone(saved_at.tzinfo)
        self.assertIsNotNone(info_saved_at.tzinfo)

    def test_missing_invalid_and_path_like_tickers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                read_prices(directory, 'AAPL')
            with self.assertRaises(ValueError):
                save_fundamentals(directory, '../escape', {'revenueGrowth': .2})
            frame = history([100.] * 40)
            frame['Open'] = 0
            with self.assertRaises(ValueError):
                save_prices(directory, 'AAPL', frame)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_fundamentals_do_not_replace_a_valid_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            save_fundamentals(directory, 'AAPL', {'revenueGrowth': .1, 'mostRecentQuarter': datetime.now(timezone.utc).timestamp()})
            with self.assertRaises(ValueError):
                save_fundamentals(directory, 'AAPL', {'mostRecentQuarter': 1})
            self.assertEqual(read_fundamentals(directory, 'AAPL')[0]['revenueGrowth'], .1)
