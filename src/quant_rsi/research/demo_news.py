"""Read one registered issuer feed into the demo; reuse the existing RSS transport."""

import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from quant_rsi.market import news_feed


class _Text(HTMLParser):
    def __init__(self, value):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.feed(value)
        self.close()

    def handle_data(self, data):
        self.parts.append(data)


def issuer_news(context, *, limit=5):
    """Current RSS titles/summaries only; no search, peer retrieval or article downloads."""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("--news-limit must be an integer between 1 and 20")
    symbol = context["symbol"]
    try:
        source = news_feed.source_for(symbol)
    except FileNotFoundError:
        raise ValueError(
            f"No registered issuer RSS feed for {symbol}; currently configured: KO"
        ) from None
    body, receipt = news_feed.fetch(source, 262144)
    if receipt.get("error"):
        raise ValueError(f"Issuer RSS retrieval failed: {receipt['error']}")
    try:
        records = news_feed.parse(body, source, 128)
    except (ValueError, ET.ParseError):
        raise ValueError("Registered issuer RSS response could not be parsed") from None
    if not records:
        raise ValueError("Registered issuer RSS returned no news; demo stopped")
    records.sort(
        key=lambda item: (item["published_at"] or item["published_date"] or "", item["event_id"]),
        reverse=True,
    )
    items = []
    for record in records[:limit]:
        fields = record["payload"]["item"]
        title = " ".join(" ".join(_Text(fields["title"]).parts).split())
        summary = " ".join(" ".join(_Text(fields["description"]).parts).split())
        items.append(
            {
                "id": record["event_id"],
                "symbol": symbol,
                "title": title[:300],
                "summary": summary[:2000],
                "text": title[:300] + "\n" + summary[:2000],
                "text_truncated": len(title) > 300 or len(summary) > 2000,
                "url": fields["link"],
                "published_at": record["published_at"],
                "published_date": record["published_date"],
                "publication_precision": record["publication_precision"],
                "received_at": receipt["received_at"],
            }
        )
    return {
        "provider": "issuer_rss",
        "source_url": source.url,
        "scope": "target_issuer_feed_only",
        "content_extent": source.content_extent,
        "received_at": receipt["received_at"],
        "available_items": len(records),
        "omitted_items": len(records) - len(items),
        "selection": "newest_publication_first_unknown_last",
        "items": items,
    }
