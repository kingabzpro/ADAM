import unittest
from unittest.mock import patch

from adam import main


class StartupTests(unittest.TestCase):
    @patch("adam.supervisor.run_dashboard")
    @patch("adam.supervisor.load_config")
    def test_startup_loads_configuration_before_dashboard(self, load_config, run_dashboard):
        main()
        load_config.assert_called_once()
        run_dashboard.assert_called_once()

    @patch("adam.supervisor.run_dashboard")
    @patch("adam.supervisor.load_config", side_effect=ValueError("invalid config"))
    def test_invalid_configuration_prevents_startup(self, load_config, run_dashboard):
        with self.assertRaises(ValueError):
            main()
        run_dashboard.assert_not_called()
