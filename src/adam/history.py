import json
from contextlib import closing
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def record_analysis(path, result, *, risk_evidence, limits):
    """Append one completed risk-gated analysis and its supplied risk context."""
    payload = {
        'ticker': result.ticker,
        'agents': result.agents,
        'evidence': result.evidence,
        'decisions': {name: decision.model_dump(mode='json') for name, decision in result.decisions.items()},
        'risk': result.risk.model_dump(mode='json'),
        'risk_evidence': risk_evidence.model_dump(mode='json'),
        'limits': limits.model_dump(mode='json') if limits is not None else None,
        'errors': result.errors,
        'timeline': result.timeline,
    }
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False)
    created_at = datetime.now(timezone.utc).isoformat()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as database, database:
        database.execute('CREATE TABLE IF NOT EXISTS analysis_history (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)')
        cursor = database.execute('INSERT INTO analysis_history (ticker, created_at, payload) VALUES (?, ?, ?)',
                                  (result.ticker, created_at, encoded))
        return cursor.lastrowid


def read_history(path, *, ticker=None, limit=100):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('limit must be between 1 and 1000')
    if not Path(path).exists():
        return []
    with closing(sqlite3.connect(path)) as database, database:
        if ticker is None:
            rows = database.execute('SELECT id, created_at, payload FROM analysis_history ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        else:
            rows = database.execute('SELECT id, created_at, payload FROM analysis_history WHERE ticker = ? ORDER BY id DESC LIMIT ?',
                                    (ticker.strip().upper(), limit)).fetchall()
    return [{'id': record_id, 'created_at': created_at, **json.loads(payload)}
            for record_id, created_at, payload in rows]
