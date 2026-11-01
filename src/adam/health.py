import re

import httpx

from adam.setup import MIN_OLLAMA_VERSION, MODEL, SERVER


async def ollama_health(client):
    """Check the local server and installed model without downloading anything."""
    try:
        response = await client.get(f'{SERVER}/api/version', follow_redirects=False)
        response.raise_for_status()
        version = response.json()['version']
        match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:[-+].*)?', version)
        if not match:
            raise ValueError('Invalid version')
        if tuple(map(int, match.groups())) < MIN_OLLAMA_VERSION:
            return {'status': 'outdated', 'version': version, 'model': MODEL}
        response = await client.get(f'{SERVER}/api/tags', follow_redirects=False)
        response.raise_for_status()
        models = response.json()['models']
        if not isinstance(models, list):
            raise ValueError('Invalid model list')
        installed = any(isinstance(item, dict) and item.get('name') == MODEL for item in models)
        return {'status': 'ready' if installed else 'missing_model', 'version': version, 'model': MODEL}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
        return {'status': 'unavailable', 'version': None, 'model': MODEL}


def health_message(health):
    status = health['status']
    if status == 'ready':
        return f"Ollama {health['version']} is reachable; {health['model']} is installed."
    if status == 'missing_model':
        return f"Ollama is reachable, but {health['model']} is missing. Run uv run adam-setup."
    if status == 'outdated':
        return f"Ollama {health['version']} is too old. Run uv run adam-setup, then restart Ollama."
    return 'Ollama is unavailable. Run uv run adam-setup and check the local server.'
