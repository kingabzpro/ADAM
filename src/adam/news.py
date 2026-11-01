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
