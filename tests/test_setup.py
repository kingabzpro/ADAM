import subprocess
import unittest
from unittest.mock import MagicMock, patch

from adam.setup import setup_ollama


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.model_patch = patch('adam.setup.model_installed', return_value=False)
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)
        self.version_patch = patch('adam.setup.server_version', return_value='0.40.13')
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.ollama_executable', return_value='C:/Ollama/ollama.exe')
    @patch('adam.setup.ensure_server')
    def test_existing_install_starts_server_and_pulls_model(self, server, executable, run):
        run.return_value.stdout = 'ollama version is 0.40.13'
        setup_ollama()
        server.assert_called_once_with('C:/Ollama/ollama.exe')
        self.assertEqual(run.call_args_list[-1].args[0], ['C:/Ollama/ollama.exe', 'pull', 'd3:2b'])

    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.ollama_executable', side_effect=[None, 'C:/Ollama/ollama.exe'])
    @patch('adam.setup.ensure_server')
    @patch('adam.setup.shutil.which', return_value='winget.exe')
    @patch('adam.setup.sys.platform', 'win32')
    def test_missing_install_uses_winget(self, which, server, executable, run):
        run.return_value.stdout = 'ollama version is 0.40.13'
        setup_ollama()
        self.assertIn('Ollama.Ollama', run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[-1].args[0][-2:], ['pull', 'd3:2b'])

    @patch('adam.setup.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'winget'))
    @patch('adam.setup.ollama_executable', return_value=None)
    @patch('adam.setup.shutil.which', return_value='winget.exe')
    @patch('adam.setup.sys.platform', 'win32')
    def test_install_failure_stops_setup(self, which, executable, run):
        with self.assertRaises(subprocess.CalledProcessError):
            setup_ollama()

    @patch('adam.setup.run_official_installer')
    @patch('adam.setup.sys.platform', 'darwin')
    def test_macos_uses_official_installer(self, installer):
        from adam.setup import install_ollama
        install_ollama()
        installer.assert_called_once_with('https://ollama.com/install.sh', 'sh')

    @patch('adam.setup.run_official_installer')
    @patch('adam.setup.sys.platform', 'linux')
    def test_linux_uses_official_installer(self, installer):
        from adam.setup import install_ollama
        install_ollama()
        installer.assert_called_once_with('https://ollama.com/install.sh', 'sh')

    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.ollama_executable', return_value='C:/Ollama/ollama.exe')
    @patch('adam.setup.ensure_server')
    @patch('adam.setup.model_installed', return_value=True)
    def test_reuses_installed_model(self, installed, server, executable, run):
        run.return_value.stdout = 'ollama version is 0.40.13'
        setup_ollama()
        self.assertEqual(run.call_count, 1)

    @patch('adam.setup.subprocess.Popen')
    @patch('adam.setup.server_ready', side_effect=[False, True])
    def test_starts_server_when_not_running(self, ready, popen):
        from adam.setup import ensure_server
        ensure_server('ollama')
        self.assertEqual(popen.call_args.args[0], ['ollama', 'serve'])

    @patch('adam.setup.install_ollama')
    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.ollama_executable', return_value='ollama')
    @patch('adam.setup.ensure_server')
    def test_rejects_old_version_before_start(self, server, executable, run, installer):
        run.return_value.stdout = 'ollama version is 0.30.0'
        with self.assertRaisesRegex(RuntimeError, '0.40.9'):
            setup_ollama()
        server.assert_not_called()

    @patch('adam.setup.sys.platform', 'unknown')
    def test_unsupported_platform_has_clear_error(self):
        from adam.setup import install_ollama
        with self.assertRaisesRegex(RuntimeError, 'Unsupported operating system'):
            install_ollama()

    @patch('adam.setup.install_ollama')
    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.ollama_executable', side_effect=['ollama', 'ollama'])
    @patch('adam.setup.ensure_server')
    def test_old_version_is_updated_automatically(self, server, executable, run, installer):
        run.side_effect = [MagicMock(stdout='ollama version is 0.30.0'), MagicMock(stdout='ollama version is 0.40.13'), MagicMock()]
        setup_ollama()
        installer.assert_called_once_with(upgrade=True)
        server.assert_called_once_with('ollama')

    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.httpx.Client')
    def test_official_installer_is_downloaded_and_executed_as_file(self, client, run):
        from pathlib import Path
        from adam.setup import run_official_installer
        client.return_value.__enter__.return_value.get.return_value.content = b'#!/bin/sh\necho ready\n'
        def verify(command, check):
            self.assertEqual(command[0], 'sh')
            self.assertTrue(check)
            self.assertEqual(Path(command[1]).read_bytes(), b'#!/bin/sh\necho ready\n')
        run.side_effect = verify
        run_official_installer('https://ollama.com/install.sh', 'sh')

    @patch('adam.setup.httpx.Client')
    def test_detects_model_by_exact_tag(self, client):
        self.model_patch.stop()
        from adam.setup import model_installed
        response = client.return_value.__enter__.return_value.get.return_value
        response.json.return_value = {'models': [{'name': 'd3:2b'}]}
        self.assertTrue(model_installed())
        response.json.return_value = {'models': [{'name': 'd3:4b'}]}
        self.assertFalse(model_installed())

    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.ollama_executable', return_value='ollama')
    @patch('adam.setup.ensure_server')
    @patch('adam.setup.server_version', return_value='0.30.0')
    def test_outdated_running_server_is_reported(self, version, server, executable, run):
        run.return_value.stdout = 'ollama version is 0.40.13'
        with self.assertRaisesRegex(RuntimeError, 'restart'):
            setup_ollama()

    @patch('adam.setup.run_official_installer')
    @patch('adam.setup.shutil.which', return_value=None)
    @patch('adam.setup.sys.platform', 'win32')
    def test_windows_without_winget_uses_official_installer(self, which, installer):
        from adam.setup import install_ollama
        install_ollama()
        installer.assert_called_once_with('https://ollama.com/install.ps1', 'powershell')

    @patch('adam.setup.subprocess.Popen')
    @patch('adam.setup.server_ready', return_value=False)
    def test_server_failure_does_not_continue(self, ready, popen):
        from adam.setup import ensure_server
        popen.return_value.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            ensure_server('ollama')

    @patch('adam.setup.subprocess.run')
    @patch('adam.setup.httpx.Client')
    def test_installer_download_error_does_not_execute(self, client, run):
        from adam.setup import run_official_installer
        client.return_value.__enter__.return_value.get.return_value.raise_for_status.side_effect = RuntimeError('download failed')
        with self.assertRaisesRegex(RuntimeError, 'download failed'):
            run_official_installer('https://ollama.com/install.sh', 'sh')
        run.assert_not_called()

    @patch('adam.setup.time.sleep')
    @patch('adam.setup.time.monotonic', side_effect=[0, 31])
    @patch('adam.setup.subprocess.Popen')
    @patch('adam.setup.server_ready', return_value=False)
    def test_startup_timeout_terminates_owned_server(self, ready, popen, clock, sleep):
        from adam.setup import ensure_server
        popen.return_value.poll.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'within 30 seconds'):
            ensure_server('ollama')
        popen.return_value.terminate.assert_called_once()

    @patch('adam.setup.setup_ollama', side_effect=RuntimeError('network unavailable'))
    @patch('sys.stderr', new_callable=__import__('io').StringIO)
    def test_cli_reports_actionable_error(self, stderr, setup):
        from adam.setup import main
        with self.assertRaises(SystemExit) as result:
            main()
        self.assertEqual(result.exception.code, 1)
        self.assertIn('network unavailable', stderr.getvalue())
