"""Human-readable summaries of supplied evidence, not model explanations."""
from ipaddress import ip_address
from math import isfinite
from urllib.parse import urlsplit

from adam.decisions import consensus


def vote_summary(decisions):
    vote = consensus(decisions)
    return {'signal': vote.signal, 'counts': vote.counts,
            'disagreement': vote.disagreement, 'abstained': list(vote.abstained)}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


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
    result = {}
    market = evidence.get('market', {})
    if _number(market.get('last_close')):
        text = f"Close {market['last_close']:.2f}; as of {market.get('as_of', 'unknown')}."
        if _number(market.get('annualized_volatility')):
            text += f" Annualized volatility {market['annualized_volatility']:.1%}."
        result['market'] = text
    technical = evidence.get('technical', {})
    fields = []
    if _number(technical.get('rsi_14')):
        fields.append(f"RSI(14) {technical['rsi_14']:.1f}")
    if _number(technical.get('sma_20')):
        fields.append(f"20-session average {technical['sma_20']:.2f}")
    if fields:
        result['technical'] = '; '.join(fields) + '.'
    fundamentals = evidence.get('fundamentals', {})
    growth = fundamentals.get('metrics', {}).get('revenue_growth')
    if _number(growth):
        result['fundamentals'] = f"Revenue growth {growth:.1%}; fiscal data freshness: {fundamentals.get('freshness', 'unknown')}."
    news = evidence.get('news', {})
    if news.get('basis') in ('articles', 'headlines'):
        basis = news['basis']
        result['news'] = f"{len(news.get(basis, []))} {basis} scored; coverage and relevance are limited."
    return result


def safe_news_link(value):
    if not isinstance(value, str) or any(ord(character) < 32 or ord(character) == 127 for character in value):
        return None
    try:
        url = urlsplit(value)
        host = url.hostname
        if url.scheme not in ('https', 'http') or not host or url.username is not None or url.password is not None:
            return None
        _ = url.port
        host = host.rstrip('.')
        if host == 'localhost' or host.endswith(('.localhost', '.local')):
            return None
        try:
            if not ip_address(host).is_global:
                return None
        except ValueError:
            pass
        return value
    except ValueError:
        return None
