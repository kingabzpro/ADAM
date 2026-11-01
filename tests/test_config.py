import tempfile
import unittest
from pathlib import Path

from adam.config import Config, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_and_partial_configuration(self):
        self.assertEqual(load_config(None), Config(inference_concurrency=1, data_concurrency=4, ticker_concurrency=4))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('data_concurrency: 2\nticker_concurrency: 3\n', encoding='utf-8')
            self.assertEqual(load_config(path), Config(inference_concurrency=1, data_concurrency=2, ticker_concurrency=3))

    def test_rejects_unknown_and_invalid_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            for content in ('other: 1', 'data_concurrency: 0', 'ticker_concurrency: true'):
                path.write_text(content, encoding='utf-8')
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_config(path)

    def test_loads_inference_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("inference_concurrency: 2\n", encoding="utf-8")
            self.assertEqual(load_config(path).inference_concurrency, 2)

    def test_rejects_non_positive_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("inference_concurrency: 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)
