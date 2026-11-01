import threading
import time
import unittest
from unittest.mock import patch

from adam.market import fetch_many


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_limits_simultaneous_fetches(self):
        lock = threading.Lock()
        active = 0
        maximum = 0

        def fetch(ticker):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return ticker

        with patch("adam.market.fetch_history", side_effect=fetch):
            result = await fetch_many(["A", "B", "C", "D"], concurrency=2)
        self.assertEqual(result, ["A", "B", "C", "D"])
        self.assertEqual(maximum, 2)
