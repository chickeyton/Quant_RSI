"""Read registered issuer feeds into the demo; reuse the existing RSS transport."""

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
    """Read target news and disclose coverage of configured business relationships."""
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("--news-limit must be an integer between 1 and 20")
    symbol = context["symbol"]
    try:
        source = news_feed.source_for(symbol)
    except FileNotFoundError:
        raise ValueError(
            f"No registered issuer RSS feed for {symbol}; currently configured: KO"
        ) from None
    target = _read_source(source, limit)
    related = context.get("related", {})
    if related.get("provider") != "instrument_relationship_profile":
        return target
    coverage = [_coverage(symbol, target)]
    items = list(target["items"])
    for peer in related["stocks"]:
        try:
            peer_source = news_feed.source_for(peer)
        except FileNotFoundError:
            coverage.append({"symbol": peer, "status": "not_configured", "selected_items": 0})
            continue
        try:
            result = _read_source(peer_source, limit)
        except ValueError as exc:
            coverage.append(
                {
                    "symbol": peer,
                    "status": "unavailable",
                    "source_url": peer_source.url,
                    "selected_items": 0,
                    "reason": str(exc),
                }
            )
            continue
        coverage.append(_coverage(peer, result))
        items.extend(result["items"])
    return {
        "provider": "issuer_rss",
        "scope": "target_and_configured_related_issuers",
        "content_extent": target["content_extent"],
        "limit_per_issuer": limit,
        "selection": "target_first_then_profile_order_newest_within_each_feed_unknown_last",
        "coverage_status": "complete"
        if all(c["status"] == "available" for c in coverage)
        else "partial",
        "coverage": coverage,
        "items": items,
    }


def _coverage(symbol, result):
    return {
        "symbol": symbol,
        "status": "available",
        "source_url": result["source_url"],
        "received_at": result["received_at"],
        "available_items": result["available_items"],
        "omitted_items": result["omitted_items"],
        "selected_items": len(result["items"]),
    }


def _read_source(source, limit):
    symbol = source.instrument
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
