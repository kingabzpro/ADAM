import tempfile
import unittest
from pathlib import Path

from adam.config import load_config


class ConfigTests(unittest.TestCase):
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
