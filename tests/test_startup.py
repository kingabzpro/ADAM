import unittest
from unittest.mock import patch

from adam import main


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.ollama_patch = patch('adam.supervisor.ollama_executable', return_value=None)
        self.ollama_patch.start()
        self.addCleanup(self.ollama_patch.stop)

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

    @patch('adam.supervisor.ensure_server')
    @patch('adam.supervisor.ollama_executable', return_value='ollama')
    @patch('adam.supervisor.run_dashboard')
    @patch('adam.supervisor.load_config')
    def test_starts_installed_server_automatically(self, load_config, run_dashboard, executable, ensure):
        main()
        ensure.assert_called_once_with('ollama')
        run_dashboard.assert_called_once_with(load_config.return_value)

    @patch('adam.supervisor.ensure_server', side_effect=RuntimeError('server error'))
    @patch('adam.supervisor.ollama_executable', return_value='ollama')
    @patch('adam.supervisor.run_dashboard')
    @patch('adam.supervisor.load_config')
    def test_dashboard_remains_available_if_server_cannot_start(self, load_config, run_dashboard, executable, ensure):
        main()
        run_dashboard.assert_called_once()
