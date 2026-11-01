import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from adam.d3 import DEFAULT_MODEL

MODEL = DEFAULT_MODEL
SERVER = 'http://127.0.0.1:11434'


def ollama_executable():
    executable = shutil.which('ollama')
    if executable:
        return executable
    local = os.environ.get('LOCALAPPDATA')
    candidate = Path(local) / 'Programs' / 'Ollama' / 'ollama.exe' if local else None
    if candidate and candidate.is_file():
        return str(candidate)
    for path in ('/Applications/Ollama.app/Contents/Resources/ollama', '/usr/local/bin/ollama', '/usr/bin/ollama'):
        if Path(path).is_file():
            return path
    return None


def run_official_installer(url, shell):
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    with tempfile.TemporaryDirectory(prefix='adam-ollama-') as directory:
        path = Path(directory) / ('install.ps1' if shell == 'powershell' else 'install.sh')
        path.write_bytes(response.content)
        command = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(path)] if shell == 'powershell' else ['sh', str(path)]
        subprocess.run(command, check=True)


def install_ollama(*, upgrade=False):
    if sys.platform == 'win32':
        if shutil.which('winget'):
            subprocess.run(['winget', 'upgrade' if upgrade else 'install', '--id', 'Ollama.Ollama', '--exact', '--silent',
                            '--accept-package-agreements', '--accept-source-agreements', '--disable-interactivity'], check=True)
        else:
            run_official_installer('https://ollama.com/install.ps1', 'powershell')
    elif sys.platform == 'darwin' or sys.platform.startswith('linux'):
        run_official_installer('https://ollama.com/install.sh', 'sh')
    else:
        raise RuntimeError(f'Unsupported operating system: {sys.platform}')


def model_installed():
    with httpx.Client(timeout=5, trust_env=False) as client:
        response = client.get(f'{SERVER}/api/tags')
        response.raise_for_status()
        return any(model.get('name') == MODEL for model in response.json().get('models', []))


def server_version():
    with httpx.Client(timeout=2, trust_env=False) as client:
        response = client.get(f'{SERVER}/api/version')
        response.raise_for_status()
        return response.json()['version']


def server_ready():
    try:
        return isinstance(server_version(), str)
    except (httpx.HTTPError, ValueError, AttributeError, KeyError, TypeError):
        return False


def ensure_server(executable):
    if server_ready():
        return
    process = subprocess.Popen([executable, 'serve'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server_ready():
            return
        if process.poll() is not None:
            raise RuntimeError('Ollama server exited before becoming ready')
        time.sleep(0.5)
    if process.poll() is None:
        process.terminate()
    raise RuntimeError('Ollama server did not become ready within 30 seconds')


def setup_ollama():
    executable = ollama_executable()
    if not executable:
        install_ollama()
        executable = ollama_executable()
        if not executable:
            raise RuntimeError('Ollama was installed but its executable could not be found')
    version = subprocess.run([executable, '--version'], check=True, capture_output=True, text=True).stdout
    match = re.search(r'(\d+)\.(\d+)\.(\d+)', version)
    if not match:
        raise RuntimeError('Unable to determine the installed Ollama version')
    if tuple(map(int, match.groups())) < (0, 40, 9):
        install_ollama(upgrade=True)
        executable = ollama_executable()
        if not executable:
            raise RuntimeError('Ollama executable not found after update')
        version = subprocess.run([executable, '--version'], check=True, capture_output=True, text=True).stdout
        match = re.search(r'(\d+)\.(\d+)\.(\d+)', version)
        if not match or tuple(map(int, match.groups())) < (0, 40, 9):
            raise RuntimeError('Ollama update did not provide version 0.40.9 or newer')
    ensure_server(executable)
    running = re.search(r'(\d+)\.(\d+)\.(\d+)', server_version())
    if not running or tuple(map(int, running.groups())) < (0, 40, 9):
        raise RuntimeError('Running Ollama server is too old; restart Ollama after updating')
    if not model_installed():
        subprocess.run([executable, 'pull', MODEL], check=True)
    print('Ollama is ready with', MODEL)


def main():
    try:
        setup_ollama()
    except (RuntimeError, OSError, subprocess.CalledProcessError, httpx.HTTPError) as error:
        print(f'Ollama setup failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
