"""Human-readable summaries of supplied evidence, not model explanations."""
from datetime import datetime
from math import isfinite

from adam.decisions import consensus


def vote_summary(decisions):
    vote = consensus(decisions)
    return {'signal': vote.signal, 'counts': vote.counts,
            'disagreement': vote.disagreement, 'abstained': list(vote.abstained)}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _session_date(value):
    """Format an ISO timestamp as a readable calendar date, or None when unparsable."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return f'{parsed:%b} {parsed.day}, {parsed.year}'


def forecast_chart(evidence):
    horizons = evidence.get('horizons')
    series = [evidence.get(key) for key in ('lower_returns', 'median_returns', 'upper_returns')]
    if (not isinstance(horizons, list) or not 1 <= len(horizons) <= 256
            or any(type(value) is not int or value < 1 for value in horizons)
            or horizons != sorted(set(horizons))
            or any(not isinstance(values, list) or len(values) != len(horizons)
                   or any(not _number(value) or not isfinite(value * 100) for value in values) for values in series)
            or any(not low <= mid <= high for low, mid, high in zip(*series, strict=True))):
        return None
    return {'tooltip': {'trigger': 'axis'}, 'legend': {'data': ['10th percentile', 'Median', '90th percentile']},
            'xAxis': {'type': 'category', 'name': 'Sessions', 'data': horizons},
            'yAxis': {'type': 'value', 'name': 'Return (%)'},
            'series': [{'name': name, 'type': 'line', 'data': [round(value * 100, 4) for value in values]}
                       for name, values in zip(('10th percentile', 'Median', '90th percentile'), series, strict=True)]}


def evidence_summary(evidence):
    """Facts plus a short plain-language reading per block, so a reader can follow the decision."""
    result = {}
    market = evidence.get('market', {})
    if _number(market.get('last_close')):
        text = f"Last close ${market['last_close']:,.2f}"
        date = _session_date(market.get('as_of'))
        if date:
            text += f' on {date}'
        meaning = None
        if _number(market.get('annualized_volatility')):
            volatility = market['annualized_volatility']
            text += f" · annualized volatility {volatility:.1%}"
            meaning = ('Volatility is high, so very large daily swings are normal.' if volatility >= .5 else
                       'Volatility is elevated, so expect wide daily swings.' if volatility >= .3 else
                       'Volatility is unremarkable for a stock.')
        result['market'] = {'facts': text, 'meaning': meaning}
    technical = evidence.get('technical', {})
    fields, readings = [], []
    rsi = technical.get('rsi_14')
    if _number(rsi):
        fields.append(f"RSI(14) {rsi:.1f}")
        readings.append('RSI is oversold' if rsi <= 30 else 'RSI is overbought' if rsi >= 70 else 'RSI is neutral')
    average = technical.get('sma_20')
    trend = None
    if _number(average):
        fields.append(f"20-session average ${average:,.2f}")
        if _number(market.get('last_close')):
            trend = 'above' if market['last_close'] > average else 'below'
            readings.append(f"price sits {trend} its 20-session average")
    if fields:
        meaning = None
        if readings:
            direction = 'up' if trend == 'above' else 'down' if trend == 'below' else None
            head = readings[0] if len(readings) == 1 else ' and '.join(readings)
            meaning = head + (f", so the short-term trend points {direction}" if direction else '') + '.'
        result['technical'] = {'facts': ' · '.join(fields), 'meaning': meaning}
    fundamentals = evidence.get('fundamentals', {})
    growth = fundamentals.get('metrics', {}).get('revenue_growth')
    if _number(growth):
        freshness = {'current': 'fiscal data is current', 'stale': 'fiscal data is stale'}.get(
            fundamentals.get('freshness'), 'fiscal quarter date is unknown')
        result['fundamentals'] = {'facts': f'Revenue growth {growth:+.1%} · {freshness}',
                                  'meaning': ('Revenue is expanding quickly.' if growth >= .2 else
                                              'Revenue is shrinking.' if growth < 0 else
                                              'Revenue is growing modestly.')}
    news = evidence.get('news', {})
    if news.get('basis') in ('articles', 'headlines'):
        basis = news['basis']
        items = news.get(basis, [])
        labels = [(item.get('sentiment') or {}).get('label') for item in items if isinstance(item, dict)]
        positive, negative = labels.count('positive'), labels.count('negative')
        if len(items) < 3:
            meaning = 'Only a few stories were scored, so news adds little signal today.'
        else:
            lean = 'lean positive' if positive > negative else 'lean negative' if negative > positive else 'are mixed'
            meaning = f'Headlines {lean}, though coverage stays limited.'
        breakdown = [f'{count} {name}' for name, count in
                     (('positive', positive), ('negative', negative), ('neutral', labels.count('neutral'))) if count]
        facts = f"{len(items)} {basis[:-1] if len(items) == 1 else basis} scored"
        if breakdown:
            facts += ' · ' + ', '.join(breakdown)
        result['news'] = {'facts': facts, 'meaning': meaning}
    return result
