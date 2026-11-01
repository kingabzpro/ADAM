import re
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

MAX_FEED_BYTES = 2_000_000


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
    unique_texts = []
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
                unique_texts.append(text)
        articles.append(article)
    if not unique_texts:
        return {"articles": articles, "scored_count": 0, "mean_scores": None}
    evidence = sentiment_evidence(unique_texts, classifier)
    sentiments = iter(evidence["articles"])
    for article in articles:
        if article["status"] == "scored":
            article["sentiment"] = next(sentiments)
    return {"articles": articles, "scored_count": len(unique_texts), "mean_scores": evidence["mean_scores"]}
