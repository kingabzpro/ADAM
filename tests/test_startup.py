import unittest
from unittest.mock import patch

from adam import main


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.ollama_patch = patch('adam.supervisor.ollama_executable', return_value=None)
        self.ollama_patch.start()
        self.addCleanup(self.ollama_patch.stop)

    def test_starts_without_a_config_file(self):
        import os
        import tempfile
        from adam.config import Config
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch('adam.supervisor.run_dashboard') as run_dashboard:
            old_directory = os.getcwd()
            try:
                os.chdir(directory)
                main()
            finally:
                os.chdir(old_directory)
        run_dashboard.assert_called_once_with(Config(inference_concurrency=1, data_concurrency=4, ticker_concurrency=4))

    def test_explicit_missing_config_is_an_error(self):
        import os
        with patch.dict(os.environ, {'ADAM_CONFIG': 'missing-config-test.yaml'}), \
             patch('adam.supervisor.run_dashboard') as run_dashboard:
            with self.assertRaises(FileNotFoundError):
                main()
        run_dashboard.assert_not_called()

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
