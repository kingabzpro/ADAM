"""Local single-ticker paper ledger; never connects to a broker."""
import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

from adam.decisions import Decision, consensus
from adam.risk import RiskEvidence, RiskLimits, RiskResult, derive_risk_evidence, evaluate_risk


def _symbol(ticker):
    if not isinstance(ticker, str) or not re.fullmatch(r'[A-Z0-9.^=\-]{1,20}', ticker.strip().upper()):
        raise ValueError('Invalid ticker')
    return ticker.strip().upper()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _database(path):
    database = sqlite3.connect(path, timeout=10)
    database.execute('PRAGMA foreign_keys = ON')
    return database


def _has_table(database, name):
    return database.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def create_account(path, ticker, *, initial_cash=10000.0, cost_bps=0.0):
    symbol = _symbol(ticker)
    cash = _positive(initial_cash, 'Initial cash')
    if isinstance(cost_bps, bool) or not isinstance(cost_bps, (int, float)) or not isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError('Invalid transaction cost')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_database(path)) as database, database:
        database.execute('CREATE TABLE IF NOT EXISTS paper_accounts (ticker TEXT PRIMARY KEY, initial_cash REAL NOT NULL, cash REAL NOT NULL, shares REAL NOT NULL, cost_basis REAL NOT NULL, cost_bps REAL NOT NULL)')
        database.execute('CREATE TABLE IF NOT EXISTS paper_trades (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL REFERENCES paper_accounts(ticker), analysis_id INTEGER NOT NULL, time TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL, shares REAL NOT NULL, fee REAL NOT NULL, realized_pnl REAL NOT NULL, risk_json TEXT NOT NULL, UNIQUE(ticker, analysis_id))')
        try:
            database.execute('INSERT INTO paper_accounts VALUES (?, ?, ?, 0, 0, ?)', (symbol, cash, cash, float(cost_bps)))
        except sqlite3.IntegrityError as error:
            raise ValueError('Paper account already exists for this ticker') from error


def execute_paper(path, analysis_id, *, price, now=None, expected_ticker=None):
    """Apply one saved, revalidated analysis at a manually supplied hypothetical price."""
    price = _positive(price, 'Price')
    if type(analysis_id) is not int or analysis_id < 1:
        raise ValueError('Invalid analysis ID')
    expected = _symbol(expected_ticker) if expected_ticker is not None else None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Time must be timezone-aware')
    if not Path(path).exists():
        raise ValueError('Paper account and saved analysis are required')
    with closing(_database(path)) as database:
        database.execute('BEGIN IMMEDIATE')
        try:
            if not _has_table(database, 'analysis_history'):
                raise ValueError('Saved analysis not found')
            row = database.execute('SELECT ticker, created_at, payload FROM analysis_history WHERE id = ?', (analysis_id,)).fetchone()
            if row is None:
                raise ValueError('Saved analysis not found')
            ticker, created_at, raw = row
            payload = json.loads(raw)
            if payload.get('ticker') != ticker or _symbol(ticker) != ticker:
                raise ValueError('Invalid analysis ticker')
            if expected is not None and ticker != expected:
                raise ValueError('Saved analysis ticker does not match the selected account')
            if not _has_table(database, 'paper_accounts') or not _has_table(database, 'paper_trades'):
                raise ValueError('Create a paper account for this ticker first')
            account = database.execute('SELECT cash, shares, cost_basis, cost_bps FROM paper_accounts WHERE ticker = ?', (ticker,)).fetchone()
            if account is None:
                raise ValueError('Create a paper account for this ticker first')
            if database.execute('SELECT 1 FROM paper_trades WHERE ticker = ? AND analysis_id = ?', (ticker, analysis_id)).fetchone():
                raise ValueError('This analysis has already been used')
            original = RiskResult.model_validate(payload['risk'])
            if original.status not in ('pass', 'warning') or original.action not in ('BUY', 'SELL'):
                raise ValueError('Saved risk result does not permit a paper trade')
            limits = RiskLimits.model_validate(payload['limits'])
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None or created.utcoffset() is None or not 0 <= (now - created).total_seconds() <= limits.max_age_seconds:
                raise ValueError('Saved analysis is stale or has an invalid timestamp')
            decisions = [Decision.model_validate(value) for value in payload['decisions'].values()]
            if any(item.ticker != ticker for item in decisions):
                raise ValueError('Decision ticker mismatch')
            available = {item.agent for item in decisions if item.status == 'completed' and item.reliable and not item.stale}
            if not {'market', 'forecast'} <= available:
                raise ValueError('Reliable market and forecast decisions are required')
            vote = consensus(decisions)
            if vote.signal != original.action:
                raise ValueError('Saved action does not match consensus')
            evidence = payload['evidence']
            if (not isinstance(evidence, dict) or not isinstance(evidence.get('market'), dict)
                    or not isinstance(evidence.get('forecast'), dict)):
                raise ValueError('Market and forecast evidence are required')
            if evidence['market'].get('comparison_status') != 'available':
                raise ValueError('Market benchmark evidence is required')
            inputs = derive_risk_evidence(RiskEvidence.model_validate(payload['evaluated_risk_evidence']),
                                          evidence['market'], evidence['forecast'])
            cash, shares, cost_basis, cost_bps = account
            rate = cost_bps / 10000
            if original.action == 'BUY':
                if shares != 0 or inputs.position_fraction is None or inputs.position_fraction <= 0:
                    raise ValueError('Buy requires a flat account and positive position fraction')
                allocation = cash * inputs.position_fraction
                quantity = allocation / (price * (1 + rate))
                if allocation <= 0 or quantity <= 0:
                    raise ValueError('Paper position is too small to represent')
                fee = quantity * price * rate
                new_cash, new_shares, new_basis = cash - allocation, quantity, allocation
                realized = 0.0
            else:
                if shares <= 0:
                    raise ValueError('Sell requires an existing paper position')
                quantity = shares
                fee = quantity * price * rate
                new_cash, new_shares, new_basis = cash + quantity * price - fee, 0.0, 0.0
                realized = quantity * price - fee - cost_basis
            if not all(isfinite(value) for value in (new_cash, new_shares, new_basis, fee, realized)):
                raise ValueError('Paper trade calculation is not finite')
            equity = new_cash + new_shares * price
            if not isfinite(equity) or equity <= 0:
                raise ValueError('Invalid paper account equity')
            exposure = new_shares * price / equity
            inputs = inputs.model_copy(update={
                'position_fraction': max(inputs.position_fraction, exposure) if inputs.position_fraction is not None else None,
                'concentration': max(inputs.concentration, exposure) if inputs.concentration is not None else None,
                'transaction_cost_fraction': max(inputs.transaction_cost_fraction, rate) if inputs.transaction_cost_fraction is not None else None,
            })
            risk = evaluate_risk(vote, inputs, limits, now=now)
            if risk.status not in ('pass', 'warning') or risk.action != original.action:
                raise ValueError('Paper trade failed risk recheck: ' + ', '.join(risk.reasons))
            database.execute('UPDATE paper_accounts SET cash = ?, shares = ?, cost_basis = ? WHERE ticker = ?',
                             (new_cash, new_shares, new_basis, ticker))
            risk_check = {'result': risk.model_dump(mode='json'), 'inputs': inputs.model_dump(mode='json'),
                          'limits': limits.model_dump(mode='json')}
            cursor = database.execute('INSERT INTO paper_trades (ticker, analysis_id, time, side, price, shares, fee, realized_pnl, risk_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                      (ticker, analysis_id, now.isoformat(), original.action, price, quantity, fee, realized, json.dumps(risk_check, allow_nan=False)))
            database.commit()
            return {'id': cursor.lastrowid, 'ticker': ticker, 'analysis_id': analysis_id, 'time': now.isoformat(),
                    'side': original.action, 'price': price, 'shares': quantity, 'fee': fee, 'realized_pnl': realized}
        except Exception:
            database.rollback()
            raise


def paper_status(path, ticker, *, price=None):
    symbol = _symbol(ticker)
    mark = None if price is None else _positive(price, 'Price')
    if not Path(path).exists():
        raise ValueError('Paper account not found')
    with closing(_database(path)) as database:
        if not _has_table(database, 'paper_accounts') or not _has_table(database, 'paper_trades'):
            raise ValueError('Paper account not found')
        account = database.execute('SELECT initial_cash, cash, shares, cost_basis, cost_bps FROM paper_accounts WHERE ticker = ?', (symbol,)).fetchone()
        if account is None:
            raise ValueError('Paper account not found')
        count, fees, realized = database.execute('SELECT COUNT(*), COALESCE(SUM(fee), 0), COALESCE(SUM(realized_pnl), 0) FROM paper_trades WHERE ticker = ?', (symbol,)).fetchone()
    initial, cash, shares, basis, cost_bps = account
    equity = cash + shares * mark if mark is not None else cash if shares == 0 else None
    if equity is not None and not isfinite(equity):
        raise ValueError('Paper valuation is not finite')
    return {'ticker': symbol, 'initial_cash': initial, 'cash': cash, 'shares': shares, 'cost_basis': basis,
            'cost_bps': cost_bps, 'trade_count': count, 'fees': fees, 'realized_pnl': realized,
            'mark_price': mark, 'equity': equity, 'unrealized_pnl': shares * mark - basis if mark is not None else None}


def paper_trades(path, ticker, *, limit=100):
    symbol = _symbol(ticker)
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError('limit must be between 1 and 1000')
    if not Path(path).exists():
        raise ValueError('Paper account not found')
    with closing(_database(path)) as database:
        if not _has_table(database, 'paper_accounts') or not _has_table(database, 'paper_trades'):
            raise ValueError('Paper account not found')
        if not database.execute('SELECT 1 FROM paper_accounts WHERE ticker = ?', (symbol,)).fetchone():
            raise ValueError('Paper account not found')
        rows = database.execute('SELECT id, analysis_id, time, side, price, shares, fee, realized_pnl, risk_json FROM paper_trades WHERE ticker = ? ORDER BY id DESC LIMIT ?', (symbol, limit)).fetchall()
    rows.reverse()
    return [{'id': row[0], 'analysis_id': row[1], 'time': row[2], 'side': row[3], 'price': row[4],
             'shares': row[5], 'fee': row[6], 'realized_pnl': row[7], 'risk_check': json.loads(row[8])} for row in rows]


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Manage local paper accounts; no brokerage orders are placed.')
    parser.add_argument('--db', default=str(Path.home() / '.adam' / 'history.sqlite'))
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create one single-ticker paper account')
    init.add_argument('ticker')
    init.add_argument('--cash', type=float, default=10000.0)
    init.add_argument('--cost-bps', type=float, default=0.0)
    trade = commands.add_parser('trade', help='Simulate a saved risk-approved analysis at a manually entered price')
    trade.add_argument('analysis_id', type=int)
    trade.add_argument('--price', type=float, required=True)
    status = commands.add_parser('status', help='Show paper account balances')
    status.add_argument('ticker')
    status.add_argument('--price', type=float)
    history = commands.add_parser('history', help='Show recent simulated trades')
    history.add_argument('ticker')
    analyses = commands.add_parser('analyses', help='List recent saved analyses to find an ID for trade')
    analyses.add_argument('ticker')
    analyses.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    try:
        if args.command == 'init':
            create_account(args.db, args.ticker, initial_cash=args.cash, cost_bps=args.cost_bps)
            result = paper_status(args.db, args.ticker)
        elif args.command == 'trade':
            result = execute_paper(args.db, args.analysis_id, price=args.price)
        elif args.command == 'status':
            result = paper_status(args.db, args.ticker, price=args.price)
        elif args.command == 'analyses':
            from adam.history import read_history
            records = read_history(args.db, ticker=args.ticker, limit=args.limit)
            result = [{'id': record['id'], 'time': record['created_at'], 'ticker': record['ticker'],
                       'action': record['risk']['action'], 'risk': record['risk']['status']} for record in records]
        else:
            result = paper_trades(args.db, args.ticker)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        print(f'Paper simulation failed: {error}', file=sys.stderr)
        raise SystemExit(1) from error
