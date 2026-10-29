"""Registered issuer RSS transport and parsing; no implicit search or article fetching."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import Field, model_validator

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.core.contracts_base import StrictModel
from quant_rsi.factors.lifecycle import EventTemplate
from quant_rsi.market import market_contracts
from quant_rsi.market.signal_sources import _publication


class FeedSource(StrictModel):
    schema_version: Literal[1] = 1
    instrument: str = Field(pattern=r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
    provider_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    source_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    url: str
    allowed_item_hosts: tuple[str, ...] = Field(min_length=1, max_length=4)
    capability: Literal["rss_polling"]
    content_extent: Literal["rss_title_and_summary"]
    event_template: EventTemplate

    @model_validator(mode="after")
    def public_url(self):
        value = urlparse(self.url)
        if (
            value.scheme != "https"
            or not value.hostname
            or value.username
            or value.password
            or value.port not in (None, 443)
            or value.query
            or value.fragment
            or value.hostname not in self.allowed_item_hosts
        ):
            raise ValueError("registered feed requires an explicit public HTTPS URL")
        return self


def source_for(instrument):
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,31}", instrument):
        raise ValueError("invalid feed instrument")
    path = (
        Path(__file__).resolve().parents[3] / "instruments" / "US" / instrument / "news_feed.json"
    )
    source = FeedSource.model_validate_json(path.read_bytes())
    if source.instrument != instrument:
        raise ValueError("feed binding differs from selected instrument")
    return source


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "feed redirects are not authorized", headers, fp)


def fetch(source, byte_limit):
    """One bounded request, no redirects or retries. Keep error bodies and actual receipt time."""
    body, metadata = b"", {}
    try:
        try:
            response = build_opener(NoRedirect()).open(
                Request(source.url, headers={"User-Agent": "QuantRSI public issuer RSS research"}),
                timeout=15,
            )
        except HTTPError as exc:
            response = exc
        with response:
            body = response.read(byte_limit)
            metadata.update(
                http_status=response.status,
                response_url=response.geturl(),
                content_type=response.headers.get("Content-Type"),
            )
        if metadata["http_status"] != 200:
            metadata["error"] = "http_error"
        elif metadata["response_url"] != source.url:
            metadata["error"] = "unexpected_response_url"
        elif len(body) >= byte_limit:
            metadata["error"] = "response_byte_limit_reached"
    except Exception as exc:
        metadata["error"] = type(exc).__name__
    metadata["received_at"] = market_contracts.trusted_now()
    return body, metadata


def parse(body, source, max_items):
    text = body.decode("utf-8-sig")
    if re.search(r"<!DOCTYPE|<!ENTITY", text, re.IGNORECASE):
        raise ValueError("RSS document declarations/entities are unsupported")
    channel = ET.fromstring(text).find("channel")
    if channel is None:
        raise ValueError("expected RSS channel")
    items = channel.findall("item")
    if len(items) > max_items:
        raise ValueError("RSS item allowance exceeded")
    records = {}
    for item in items:
        fields = {
            key: item.findtext(key, "").strip()
            for key in ("guid", "link", "title", "description", "pubDate")
        }
        identity = fields["guid"] or fields["link"]
        if not identity or not fields["title"]:
            raise ValueError("RSS item requires identity and title")
        link = urlparse(fields["link"])
        if fields["link"] and (
            link.scheme != "https"
            or link.hostname not in source.allowed_item_hosts
            or link.username
            or link.password
        ):
            raise ValueError("RSS item link outside registered issuer hosts")
        published, day, precision = _publication(fields["pubDate"])
        record = dict(
            event_id="feed-" + digest(canonical([source.provider_id, identity]))[:48],
            payload=dict(title=fields["title"], item=fields, provider_id=source.provider_id),
            published_at=published,
            published_date=day,
            publication_precision=precision,
        )
        if identity in records and records[identity] != record:
            raise ValueError("conflicting duplicate RSS item")
        records[identity] = record
    return sorted(records.values(), key=lambda record: record["event_id"])
