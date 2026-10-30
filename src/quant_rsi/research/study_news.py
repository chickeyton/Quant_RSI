"""Bounded receipt-preserving news snapshots and retrospective selected-corpus counts."""

import re
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.market.market_contracts import timestamp, utc_text

MAX_FILES, MAX_BYTES, MAX_FILE_BYTES = 256, 8 * 1024 * 1024, 512 * 1024


def _time(value):
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("News timestamps must be bounded timezone-aware ISO strings")
    return timestamp(value)


def _text(value, limit):
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError("Invalid bounded news string")
    return value


def _url(value):
    parsed = urlsplit(_text(value, 2048))
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("News source/item URLs must be explicit public HTTPS URLs")


def _validate(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("provider") not in {
        "issuer_rss",
        "simulated_news_archive",
    }:
        raise ValueError("Unsupported news archive provider")
    if len(canonical(snapshot)) > MAX_FILE_BYTES:
        raise ValueError("News snapshot exceeds byte limit")
    items, sources = snapshot.get("items"), {}
    if not isinstance(items, list) or len(items) > 80:
        raise ValueError("News snapshot needs at most 80 selected items")
    if snapshot.get("scope") == "target_issuer_feed_only":
        symbols = {item.get("symbol") for item in items if isinstance(item, dict)}
        if len(symbols) != 1:
            raise ValueError("Single-source news must identify exactly one target")
        sources[next(iter(symbols))] = dict(snapshot, status="available")
    elif snapshot.get("scope") == "target_and_configured_related_issuers":
        coverage = snapshot.get("coverage")
        if not isinstance(coverage, list) or not 1 <= len(coverage) <= 4:
            raise ValueError("News coverage must contain one to four issuers")
        for source in coverage:
            if not isinstance(source, dict) or source.get("symbol") in sources:
                raise ValueError("Invalid or duplicate news coverage")
            sources[source.get("symbol")] = source
    else:
        raise ValueError("Unsupported news snapshot scope")
    receipts = []
    for symbol, source in sources.items():
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", symbol):
            raise ValueError("Invalid news instrument")
        if source.get("status") not in {"available", "not_configured", "unavailable"}:
            raise ValueError("Invalid news coverage status")
        if source.get("received_at") is not None:
            receipts.append(_time(source["received_at"]))
        if source["status"] == "available":
            _url(source.get("source_url"))
            _time(source.get("received_at"))
            synthetic_url = urlsplit(source["source_url"]).hostname == "simulated.invalid"
            if synthetic_url != (snapshot["provider"] == "simulated_news_archive"):
                raise ValueError("Simulated archive sources must be explicitly labeled")
    seen = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("News items must be objects")
        identifier = _text(item.get("id"), 160)
        source = sources.get(item.get("symbol"))
        if source is None or source["status"] != "available":
            raise ValueError("News item lacks available source coverage")
        if _time(item.get("received_at")) != _time(source["received_at"]):
            raise ValueError("Item receipt must match its actual source receipt")
        _url(item.get("url"))
        _text(item.get("title"), 300)
        if not isinstance(item.get("summary", ""), str) or len(item.get("summary", "")) > 2000:
            raise ValueError("Invalid bounded news summary")
        precision = item.get("publication_precision")
        if precision == "second":
            _time(item.get("published_at"))
            if item.get("published_date") is not None:
                raise ValueError("Precise publication cannot also carry a date-only value")
        elif precision == "day":
            date.fromisoformat(_text(item.get("published_date"), 10))
            if item.get("published_at") is not None:
                raise ValueError("Date-only publication cannot invent a timestamp")
        elif precision != "unknown" or any(
            item.get(key) is not None for key in ("published_at", "published_date")
        ):
            raise ValueError("Invalid publication precision")
        if identifier in seen and seen[identifier] != item:
            raise ValueError("Conflicting duplicate news ID within one snapshot")
        seen[identifier] = item
    if not receipts:
        raise ValueError("A snapshot needs at least one actual source receipt")
    if any(sum(item["symbol"] == symbol for item in items) > 20 for symbol in sources):
        raise ValueError("News snapshot exceeds 20 selected items per issuer")
    return sources, max(receipts)


def load_archive(path):
    """Read only bounded content-addressed JSON; missing directories represent no history."""
    root, snapshots, total = Path(path), [], 0
    if not root.exists():
        return snapshots
    for index, file in enumerate(root.iterdir()):
        if index >= MAX_FILES or not re.fullmatch(r"[0-9a-f]{64}\.json", file.name):
            raise ValueError("News archive has too many files or an unsupported filename")
        size = file.stat().st_size
        total += size
        if file.is_symlink() or not file.is_file() or size > MAX_FILE_BYTES or total > MAX_BYTES:
            raise ValueError("News archive file or byte bound exceeded")
        with file.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) != size:
            raise ValueError("News archive file changed during bounded read")
        snapshot = decode(raw)
        _validate(snapshot)
        if file.stem != digest(canonical(snapshot)):
            raise ValueError("News archive filename does not match snapshot content")
        snapshots.append(snapshot)
    return snapshots


def retain_snapshot(path, snapshot):
    """Append exact supplied receipt data; never synthesize a historical reception time."""
    _validate(snapshot)
    root, raw = Path(path), canonical(snapshot)
    existing = load_archive(root)
    file = root / (digest(raw) + ".json")
    if file.exists():
        return str(file)
    retained_bytes = sum(file.stat().st_size for file in root.iterdir()) if root.exists() else 0
    if len(existing) >= MAX_FILES or retained_bytes + len(raw) > MAX_BYTES:
        raise ValueError("News archive capacity exceeded")
    root.mkdir(parents=True, exist_ok=True)
    with file.open("xb") as stream:
        stream.write(raw)
    return str(file)


class NewsHistory:
    def __init__(self, snapshots):
        if len(snapshots) > MAX_FILES or sum(len(canonical(s)) for s in snapshots) > MAX_BYTES:
            raise ValueError("News history exceeds archive bounds")
        self._snapshots = [(deepcopy(s), *_validate(s)) for s in snapshots]
        self.providers = tuple(sorted({s[0]["provider"] for s in self._snapshots}))
        self.snapshot_count = len(self._snapshots)

    def evaluate(self, recipe, symbol, cutoffISO):
        keywords, days = recipe.get("keywords"), recipe.get("window_days")
        if (
            recipe.get("op") != "headline_keyword_count"
            or not isinstance(keywords, list)
            or not 1 <= len(keywords) <= 5
            or any(not isinstance(k, str) or not k.strip() or len(k) > 80 for k in keywords)
            or type(days) is not int
            or not 1 <= days <= 30
        ):
            raise ValueError("Invalid bounded headline keyword recipe")
        cutoff, candidates = _time(cutoffISO), []
        result = dict(
            value=None,
            reasons=[],
            evidence_news_ids=[],
            snapshot_received_at=None,
            provider=None,
            scope="retained_selected_target_issuer_headlines",
        )
        for snapshot, sources, anchor in self._snapshots:
            source = sources.get(symbol)
            if source is None:
                continue
            receipt = (
                _time(source["received_at"]) if source and source.get("received_at") else anchor
            )
            if receipt <= cutoff:
                candidates.append((receipt, snapshot, source))
        if not candidates:
            result["reasons"] = ["no_snapshot_received_by_cutoff"]
            return result
        latest = max(row[0] for row in candidates)
        selected = [row for row in candidates if row[0] == latest]
        identities = {
            digest(
                canonical(
                    [
                        row[1]["provider"],
                        {key: row[2].get(key) for key in ("status", "source_url", "received_at")},
                        {item["id"]: item for item in row[1]["items"] if item["symbol"] == symbol},
                    ]
                )
            )
            for row in selected
        }
        if len(identities) > 1:
            result["reasons"] = ["ambiguous_snapshots_at_same_receipt"]
            return result
        receipt, snapshot, source = selected[0]
        result.update(provider=snapshot["provider"], snapshot_received_at=utc_text(receipt))
        if not source or source["status"] != "available":
            result["reasons"] = ["target_coverage_unavailable"]
        elif cutoff - receipt > timedelta(hours=24):
            result["reasons"] = ["snapshot_older_than_24_hours"]
        else:
            items = [item for item in snapshot["items"] if item["symbol"] == symbol]
            if any(item["publication_precision"] != "second" for item in items):
                result["reasons"] = ["publication_time_not_precise"]
            else:
                result["evidence_news_ids"] = sorted(
                    {
                        item["id"]
                        for item in items
                        if cutoff - timedelta(days=days) <= _time(item["published_at"]) < cutoff
                        and any(word.casefold() in item["title"].casefold() for word in keywords)
                    }
                )
                result["value"] = float(len(result["evidence_news_ids"]))
        return result


def simulated_history(data, symbol):
    """Invented daily snapshots for synthetic-price demos only; no market-news claim."""
    snapshots = []
    for index, session in enumerate(data.dates):
        received = utc_text(timestamp(data.timestamp(index, "close")) - timedelta(minutes=1))
        snapshots.append(
            dict(
                provider="simulated_news_archive",
                scope="target_issuer_feed_only",
                schema_version=1,
                source_url=f"https://simulated.invalid/{symbol}/rss",
                received_at=received,
                items=[
                    dict(
                        id=f"simulated.{symbol}.{session}.v1",
                        symbol=symbol,
                        title="SIMULATED "
                        + ("earnings growth" if index % 2 else "demand weakness"),
                        summary="Invented fixture, not real issuer news.",
                        url=f"https://simulated.invalid/{symbol}/{session}",
                        published_at=utc_text(timestamp(received) - timedelta(hours=1)),
                        published_date=None,
                        publication_precision="second",
                        received_at=received,
                    )
                ],
            )
        )
    return snapshots
