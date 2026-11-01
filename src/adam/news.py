import re
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

MAX_FEED_BYTES = 2_000_000
ARTICLE_HOSTS = {'coincentral.com', 'www.coincentral.com', 'blockonomi.com', 'www.blockonomi.com'}
NEWS_USER_AGENT = 'ADAMResearchBot/0.1'


def parse_feed(content: bytes) -> list[dict]:
    if len(content) > MAX_FEED_BYTES:
        raise ValueError("News feed is too large")
    try:
        root = ElementTree.fromstring(content)
    except (ParseError, DefusedXmlException) as error:
        raise ValueError("Invalid news feed") from error
    if root.tag != "rss" or root.find("channel") is None:
        raise ValueError("Expected an RSS feed")
    articles, seen = [], set()
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        url = urlsplit(link)
        if not title or url.scheme not in ("https", "http") or not url.hostname or link in seen:
            continue
        seen.add(link)
        published = None
        try:
            date = parsedate_to_datetime(item.findtext("pubDate") or "")
            if date.tzinfo is not None:
                published = date.astimezone(timezone.utc).isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
        articles.append({"title": title, "url": link, "published": published,
                         "source": (item.findtext("source") or "").strip()})
    return articles


async def discover_news(ticker: str, client) -> list[dict]:
    symbol = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=\-]{1,20}", symbol):
        raise ValueError("Invalid ticker")
    async with client.stream("GET", "https://news.google.com/rss/search",
                             params={"q": f"{symbol} stock", "hl": "en-US", "gl": "US", "ceid": "US:en"},
                             follow_redirects=False, timeout=15) as response:
        response.raise_for_status()
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > MAX_FEED_BYTES:
                raise ValueError("News feed is too large")
    return parse_feed(bytes(content))


def extract_article(html: str) -> str | None:
    from trafilatura import extract

    if not html or not html.strip():
        return None
    return extract(html, include_comments=False, include_tables=False)


def analyze_articles(records: list[dict], classifier) -> dict:
    from adam.sentiment import sentiment_evidence

    articles = []
    chunks = []
    seen = set()
    for record in records:
        article = dict(record)
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            article["status"] = "unavailable"
        else:
            key = " ".join(text.casefold().split())
            if key in seen:
                article["status"] = "duplicate"
            else:
                seen.add(key)
                article["status"] = "scored"
                normalized = ' '.join(text.split())
                article['text_truncated'] = len(normalized) > 20000
                normalized = normalized[:20000]
                pieces = [normalized[i:i + 800] for i in range(0, len(normalized), 800)]
                article['sentiment_chunks'] = len(pieces)
                chunks.extend(pieces)
        articles.append(article)
    if not chunks:
        return {"articles": articles, "scored_count": 0, "mean_scores": None}
    evidence = sentiment_evidence(chunks, classifier)
    sentiments = iter(evidence["articles"])
    scores = []
    for article in articles:
        if article["status"] == "scored":
            group = [next(sentiments)['scores'] for _ in range(article['sentiment_chunks'])]
            averaged = {label: sum(item[label] for item in group) / len(group) for label in ('positive', 'neutral', 'negative')}
            article['sentiment'] = {'label': max(averaged, key=averaged.get), 'scores': averaged}
            scores.append(averaged)
    return {"articles": articles, "scored_count": len(scores),
            "mean_scores": {label: sum(item[label] for item in scores) / len(scores) for label in ('positive', 'neutral', 'negative')}}


def financial_highlights(text):
    sentences = re.split(r'(?<=[.!?])\s+', ' '.join(text.split()))
    pattern = re.compile(r'\b(revenue|earnings|profit|loss|margin|cash flow|sales|debt|dividend|guidance)\b', re.I)
    return [sentence for sentence in sentences if len(sentence) <= 240 and pattern.search(sentence)
            and re.search(r'\d', sentence)][:3]


async def article_evidence(ticker, client, classifier, *, now=None):
    """Search Bing and score recent accessible articles from an explicit host list."""
    import asyncio
    from datetime import datetime, timedelta, timezone
    import httpx

    symbol = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9.^=\-]{3,20}', symbol):
        raise ValueError('Invalid ticker for article search')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must be timezone-aware')
    async with client.stream('GET', 'https://www.bing.com/news/search',
                             params={'q': f'{symbol} stock', 'format': 'rss'},
                             follow_redirects=False, timeout=15) as response:
        response.raise_for_status()
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > MAX_FEED_BYTES:
                raise ValueError('News feed is too large')
    records = []
    attempts = 0
    robots = {}
    for item in parse_feed(bytes(content)):
        wrapper = urlsplit(item['url'])
        if wrapper.hostname != 'www.bing.com' or wrapper.path != '/news/apiclick.aspx':
            continue
        candidates = parse_qs(wrapper.query).get('url', [])
        if len(candidates) != 1:
            continue
        url = urlsplit(candidates[0])
        try:
            published = datetime.fromisoformat(item.get('published') or '')
            age = now - published
            if (url.scheme != 'https' or url.hostname not in ARTICLE_HOSTS or url.username or url.password
                    or url.port not in (None, 443) or age < timedelta(0) or age > timedelta(days=7)):
                continue
        except (ValueError, TypeError):
            continue
        attempts += 1
        if attempts > 10:
            break
        try:
            host = url.hostname
            if host not in robots:
                robots[host] = None
                async with client.stream('GET', f'https://{host}/robots.txt', follow_redirects=False,
                                         headers={'User-Agent': NEWS_USER_AGENT}, timeout=10) as response:
                    if response.status_code == 404:
                        robots[host] = True
                    elif response.status_code == 200:
                        rules = bytearray()
                        async for chunk in response.aiter_bytes():
                            rules.extend(chunk)
                            if len(rules) > 256_000:
                                raise ValueError('Robots file is too large')
                        lines = bytes(rules).decode('utf-8', errors='replace').splitlines()
                        # robotparser does not implement wildcard or end-anchor path matching.
                        if any(re.match(r'^\s*(allow|disallow)\s*:.*[*$]', line, re.I) for line in lines):
                            continue
                        parser = RobotFileParser()
                        parser.parse(lines)
                        # Conservatively honor every disallow, including more-specific rules.
                        denied = [unquote(match.group(1).split('#', 1)[0].strip()) for line in lines
                                  if (match := re.match(r'^\s*disallow\s*:\s*(.*)', line, re.I))]
                        robots[host] = (parser, [path for path in denied if path])
            policy = robots[host]
            path_query = unquote(url.path + (f'?{url.query}' if url.query else ''))
            if (policy is None or (policy is not True and
                    (not policy[0].can_fetch(NEWS_USER_AGENT, candidates[0])
                     or any(path_query.startswith(path) for path in policy[1])))):
                continue
            async with client.stream('GET', candidates[0], follow_redirects=False,
                                     headers={'User-Agent': NEWS_USER_AGENT}, timeout=15) as response:
                if response.status_code != 200 or 'text/html' not in response.headers.get('content-type', '').lower():
                    continue
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > 1_000_000:
                        raise ValueError('Article is too large')
                html = bytes(body).decode(response.encoding or 'utf-8', errors='replace')
            text = await asyncio.to_thread(extract_article, html)
        except (httpx.HTTPError, ValueError, LookupError):
            continue
        if not text or not re.search(r'(?<![A-Za-z0-9])' + re.escape(symbol) + r'(?![A-Za-z0-9])', text):
            continue
        records.append({'title': item['title'], 'url': candidates[0], 'published': item['published'],
                        'source': url.hostname, 'text': text})
        if len(records) >= 5:
            break
    result = await asyncio.to_thread(analyze_articles, records, classifier)
    articles = []
    for article in result['articles']:
        if article['status'] == 'scored':
            text = article.pop('text')
            article['summary'] = ' '.join(text.split())[:300]
            article['financial_highlights'] = financial_highlights(text[:20000])
            articles.append(article)
    return {'basis': 'articles', 'articles': articles, 'mean_scores': result['mean_scores'],
            'note': 'Limited publisher coverage; article text is capped and FinBERT may truncate chunks. Ticker mention does not guarantee relevance.'}


async def news_evidence(ticker, client, classifier):
    import httpx

    try:
        articles = await article_evidence(ticker, client, classifier)
        if articles['articles']:
            return articles
    except (httpx.HTTPError, ValueError):
        pass
    result = await headline_evidence(ticker, client, classifier)
    result['note'] = 'Article retrieval unavailable or yielded no usable articles; search headlines may be unrelated.'
    return result


async def headline_evidence(ticker, client, classifier, *, now=None, max_age_days=7, limit=20):
    import asyncio
    from datetime import datetime, timedelta, timezone

    from adam.sentiment import sentiment_evidence

    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('now must be timezone-aware')
    if type(max_age_days) is not int or max_age_days < 1 or type(limit) is not int or limit < 1:
        raise ValueError('Age and limit must be positive integers')
    items = await discover_news(ticker, client)
    selected, seen = [], set()
    for item in items:
        title = (item.get('title') or '').strip()
        key = ' '.join(title.casefold().split())
        try:
            published = datetime.fromisoformat(item.get('published') or '')
            age = now - published
        except (ValueError, TypeError):
            continue
        if not title or key in seen or age < timedelta(0) or age > timedelta(days=max_age_days):
            continue
        seen.add(key)
        if len(selected) < limit:
            selected.append(dict(item))
    if not selected:
        return {'basis': 'headlines', 'headlines': [], 'mean_scores': None, 'excluded_count': len(items)}
    scored = await asyncio.to_thread(sentiment_evidence, [item['title'] for item in selected], classifier)
    for item, sentiment in zip(selected, scored['articles'], strict=True):
        item['sentiment'] = sentiment
    return {'basis': 'headlines', 'headlines': selected, 'mean_scores': scored['mean_scores'],
            'excluded_count': len(items) - len(selected),
            'note': 'Search results may be unrelated; sentiment covers headlines, not full articles.'}


def compact_news_evidence(evidence):
    if evidence['basis'] == 'articles':
        return {'basis': 'articles', 'total_articles': len(evidence['articles']),
                'articles': [{'title': item.get('title', '')[:200], 'summary': item.get('summary', '')[:300],
                              'published': item.get('published'), 'sentiment': item.get('sentiment', {}).get('label'),
                              'financial_highlights': [s for s in item.get('financial_highlights', []) if len(s) <= 180][:1]}
                             for item in evidence['articles'][:5]],
                'mean_scores': {name: round(value, 4) for name, value in evidence['mean_scores'].items()},
                'note': 'Limited publisher coverage; text and chunks may be truncated. Ticker mention does not guarantee relevance.'}
    # shortcut: D3 sees at most 10 shortened titles; increase after context budgeting is measured.
    headlines = evidence['headlines']
    return {
        'basis': 'headlines',
        'total_headlines': len(headlines),
        'headlines': [{'title': item.get('title', '')[:240],
                       'published': item.get('published'),
                       'sentiment': item.get('sentiment', {}).get('label')}
                      for item in headlines[:10]],
        'mean_scores': {name: round(value, 4) for name, value in evidence['mean_scores'].items()},
        'note': 'Headline sentiment only; search results may be unrelated. The aggregate covers all scored headlines; titles shown are limited and may be shortened.',
    }
