from datetime import datetime, timezone
from math import isfinite

import yfinance as yf

from adam.yahoo import yahoo_request

FIELDS = {
    "revenue_growth": "revenueGrowth", "earnings_growth": "earningsGrowth",
    "profit_margin": "profitMargins", "operating_margin": "operatingMargins",
    "return_on_equity": "returnOnEquity", "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE", "price_to_book": "priceToBook",
    "debt_to_equity": "debtToEquity", "current_ratio": "currentRatio",
    "free_cash_flow": "freeCashflow", "operating_cash_flow": "operatingCashflow",
}


def fetch_fundamentals(ticker: str) -> dict:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker must not be blank")
    source = yf.Ticker(ticker)
    return yahoo_request(source.get_info)


def fundamentals_evidence(info: dict, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    metrics, missing = {}, []
    for name, key in FIELDS.items():
        value = info.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value):
            metrics[name] = float(value)
        else:
            missing.append(name)
    date = None
    timestamp = info.get("mostRecentQuarter")
    try:
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) and isfinite(timestamp):
            date = datetime.fromtimestamp(timestamp, timezone.utc)
    except (OverflowError, OSError, ValueError):
        pass
    age = (now - date).total_seconds() if date else None
    freshness = "unknown" if age is None or age < 0 else "stale" if age > 180 * 86400 else "current"
    return {"metrics": metrics, "missing": missing,
            "report_date": date.date().isoformat() if date else None,
            "freshness": freshness, "freshness_basis": "most recent fiscal quarter end; 180-day threshold",
            "note": "Individual metric dates may differ; market valuation ratios can change daily."}
