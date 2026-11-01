import time
import unittest
from unittest.mock import patch

from adam import cache


class CacheTests(unittest.TestCase):
    def setUp(self):
        cache.clear()
        self.addCleanup(cache.clear)

    def test_round_trip_and_version_bump(self):
        first = cache.store('prices', 'AAPL', 'one', 60)
        self.assertEqual(cache.cached('prices', 'AAPL', 60), ('one', first))
        second = cache.store('prices', 'AAPL', 'two', 60)
        self.assertEqual(second, first + 1)
        self.assertEqual(cache.cached('prices', 'AAPL', 60), ('two', second))
        self.assertIsNone(cache.cached('prices', 'MSFT', 60))
        self.assertIsNone(cache.cached('fundamentals', 'AAPL', 60))

    def test_expired_entries_are_not_served(self):
        version = cache.store('prices', 'AAPL', 'one', 60)
        monotonic = time.monotonic
        with patch('adam.cache.time.monotonic', lambda: monotonic() + 61):
            self.assertIsNone(cache.cached('prices', 'AAPL', 60))
        self.assertEqual(cache.cached('prices', 'AAPL', 60), ('one', version))

    def test_non_positive_ttl_disables_the_cache(self):
        self.assertIsNone(cache.store('prices', 'AAPL', 'one', 0))
        self.assertIsNone(cache.cached('prices', 'AAPL', 0))
        cache.store('prices', 'AAPL', 'one', 60)
        self.assertIsNone(cache.cached('prices', 'AAPL', -1))

    def test_clear_removes_entries_but_keeps_versions_monotonic(self):
        first = cache.store('prices', 'AAPL', 'one', 60)
        cache.clear()
        self.assertIsNone(cache.cached('prices', 'AAPL', 60))
        self.assertEqual(cache.store('prices', 'AAPL', 'two', 60), first + 1)


if __name__ == '__main__':
    unittest.main()
