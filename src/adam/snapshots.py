"""Local, validated snapshots for explicitly requested offline research."""
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from adam.fundamentals import FIELDS
from adam.technical import technical_evidence


def _path(directory, ticker, kind):
    symbol = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9.^=\-]{1,20}', symbol):
        raise ValueError('Invalid ticker')
    name = hashlib.sha256(symbol.encode()).hexdigest()
    return Path(directory) / f'{name}-{kind}.json', symbol


def _write(directory, ticker, kind, payload):
    path, symbol = _path(directory, ticker, kind)
    encoded = json.dumps({'ticker': symbol, 'kind': kind, 'saved_at': datetime.now(timezone.utc).isoformat(),
                          'data': payload}, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
            temporary = file.name
            file.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _read(directory, ticker, kind):
    path, symbol = _path(directory, ticker, kind)
    with path.open(encoding='utf-8') as file:
        content = file.read(4_000_001)
    if len(content) > 4_000_000:
        raise ValueError('Snapshot is too large')
    payload = json.loads(content)
    if payload['ticker'] != symbol or payload['kind'] != kind:
        raise ValueError('Snapshot does not match the requested data')
    saved_at = datetime.fromisoformat(payload['saved_at'])
    if saved_at.tzinfo is None or saved_at.utcoffset() is None or saved_at > datetime.now(timezone.utc):
        raise ValueError('Invalid snapshot time')
    return payload['data'], saved_at


def _validated_prices(frame):
    technical_evidence(frame)
    if 'Open' not in frame:
        raise ValueError('Open prices are required')
    data = frame[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='raise').astype(float)
    if (not np.isfinite(data.to_numpy()).all() or (data['Open'] <= 0).any()
            or (data['Open'] > data['High']).any() or (data['Open'] < data['Low']).any()):
        raise ValueError('Invalid open prices')
    return data


def save_prices(directory, ticker, frame):
    data = _validated_prices(frame)
    _write(directory, ticker, 'prices', {'timezone': str(data.index.tz), 'index_name': data.index.name,
                                        'dates': [date.isoformat() for date in data.index],
                                        'rows': data.to_numpy().tolist()})


def read_prices(directory, ticker):
    data, saved_at = _read(directory, ticker, 'prices')
    dates = pd.to_datetime(data['dates'], utc=True).tz_convert(data['timezone'])
    dates.name = data['index_name']
    frame = pd.DataFrame(data['rows'], index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    return _validated_prices(frame), saved_at


def save_fundamentals(directory, ticker, info):
    from math import isfinite
    from adam.fundamentals import fundamentals_evidence

    data = {key: value for key in (*FIELDS.values(), 'mostRecentQuarter')
            if isinstance((value := info.get(key)), (int, float)) and not isinstance(value, bool) and isfinite(value)}
    evidence = fundamentals_evidence(data)
    if not evidence['metrics'] or evidence['freshness'] != 'current':
        raise ValueError('Only current fundamentals with metrics can be saved')
    _write(directory, ticker, 'fundamentals', data)


def read_fundamentals(directory, ticker):
    data, saved_at = _read(directory, ticker, 'fundamentals')
    if not isinstance(data, dict) or set(data) - {*FIELDS.values(), 'mostRecentQuarter'}:
        raise ValueError('Invalid fundamentals snapshot')
    return data, saved_at


def _validated_news(data):
    from math import isclose, isfinite

    if not isinstance(data, dict) or data.get('basis') not in ('articles', 'headlines'):
        raise ValueError('Invalid news snapshot')
    items = data.get(data['basis'])
    scores = data.get('mean_scores')
    labels = {'positive', 'neutral', 'negative'}
    if (not isinstance(items, list) or not 1 <= len(items) <= 20 or not isinstance(scores, dict)
            or set(scores) != labels or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                              or not isfinite(v) or not 0 <= v <= 1 for v in scores.values())
            or not isclose(sum(scores.values()), 1, abs_tol=1e-5)):
        raise ValueError('Invalid news snapshot scores')
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get('title'), str) or not item['title'].strip()
                or not isinstance(item.get('sentiment'), dict) or item['sentiment'].get('label') not in labels
                or not isinstance(item.get('summary', ''), str)
                or not isinstance(item.get('financial_highlights', []), list)
                or any(not isinstance(value, str) for value in item.get('financial_highlights', []))):
            raise ValueError('Invalid news snapshot item')
        try:
            date = datetime.fromisoformat(item.get('published') or '')
        except (ValueError, TypeError) as error:
            raise ValueError('Invalid news publication time') from error
        if date.tzinfo is None or date.utcoffset() is None or date > datetime.now(timezone.utc):
            raise ValueError('Invalid news publication time')
    if len(json.dumps(data, allow_nan=False)) > 1_000_000:
        raise ValueError('News snapshot is too large')
    return data


def save_news(directory, ticker, evidence):
    _write(directory, ticker, 'news', _validated_news(evidence))


def read_news(directory, ticker):
    data, saved_at = _read(directory, ticker, 'news')
    return _validated_news(data), saved_at
