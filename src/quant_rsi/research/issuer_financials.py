"""Bounded official earnings archives and explicit GAAP quarterly PDF rows."""

import io
import json
import re
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlsplit
from urllib.request import Request, build_opener

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.market.news_feed import NoRedirect

NOTICE = (
    "Official issuer earnings statements, reconstructed retrospectively from retained public "
    "documents, not a historical vendor archive. Only explicitly reported three-month USD "
    "columns are used; no YTD subtraction or invented quarter starts. Date-only publication "
    "and PDF-declared revisions become usable at 05:00 UTC the following day. "
    "PDF metadata is provenance, not proof that this exact version was historically observed. "
    "Page maintenance dates are recorded separately; "
    "without PDF dates they are used conservatively."
)


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.meta = [], {}
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag == "meta":
            self.meta[attrs.get("name", attrs.get("property"))] = attrs.get("content")


def statement_rows(text, profile):
    """Read the first two columns only after the expected quarterly statement heading."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if not any(profile["statement_title"].lower() in line.lower() for line in lines[:5]):
        return None
    if not any("in millions" in line.lower() for line in lines[:6]):
        raise ValueError("Statement currency scale is not explicitly millions")
    header = next(
        (i for i, line in enumerate(lines[:10]) if line.startswith("Three Months Ended")), None
    )
    if header is None:
        return None  # Annual and YTD-only pages cannot supply quarterly facts.
    if len(lines) <= header + 2:
        raise ValueError("Missing quarterly statement column dates")
    months = re.findall(r"([A-Z][a-z]+ \d{1,2}),", lines[header + 1])
    years = re.findall(r"\b(20\d{2})\b", lines[header + 2])
    if len(months) != len(years) or len(years) not in {2, 4}:
        raise ValueError("Ambiguous quarterly statement column dates")
    ends = [
        datetime.strptime(f"{m} {y}", "%B %d %Y").date()
        for m, y in zip(months[:2], years[:2], strict=True)
    ]
    if not 350 <= (ends[0] - ends[1]).days <= 380:
        raise ValueError("Quarterly columns are not comparable year-over-year periods")
    rows = []
    for field, label in profile["rows"].items():
        # PDF glyph positioning can insert spaces inside words ("To t a l").
        # Match only the configured label, without relaxing numeric parsing.
        label_pattern = r"\s*".join(re.escape(c) for c in label if not c.isspace())
        pattern = re.compile(r"^" + label_pattern + r"\s+(.*)$", re.I)
        matches = [pattern.match(line) for line in lines[header + 3 :]]
        candidates = []
        for match in filter(None, matches):
            values = match[1].replace("$", "").replace("%", "").split()
            expected = len(years) + int("%" in lines[header + 1])
            if len(values) == expected + 1 and re.fullmatch(r"\(\d{1,2}\)", values[0]):
                values = values[1:]
            if len(values) != expected:
                raise ValueError("Ambiguous quarterly statement value columns")
            if any(v in {"—", "–"} for v in values[:2]):
                raise ValueError("A dash is not an executable financial value")
            magnitude = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
            number = rf"(?:-?{magnitude}|\({magnitude}\))"
            if any(not re.fullmatch(number, v) for v in values[:2]):
                raise ValueError("Ambiguous quarterly numeric value")
            pair = [
                -float(v[1:-1].replace(",", "")) if v.startswith("(") else float(v.replace(",", ""))
                for v in values[:2]
            ]
            candidates.append(pair)
        if not candidates or any(pair != candidates[0] for pair in candidates):
            raise ValueError(f"Missing or conflicting quarterly row: {label}")
        for end, value in zip(ends, candidates[0], strict=True):
            if field == "revenue" and value <= 0:
                raise ValueError("Reported quarterly revenue must be positive")
            rows.append({"field": field, "tag": label, "end": str(end), "value": value * 1_000_000})
    return rows


def parse_statement(raw, profile):
    reader = PdfReader(io.BytesIO(raw))
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 40:
        raise ValueError("Unsupported encrypted or oversized earnings statement")
    found = []
    for number, page in enumerate(reader.pages):
        contents = page.get_contents()
        if contents and len(contents.get_data()) > 4_000_000:
            raise ValueError("Statement page content exceeds the parsing bound")
        text = page.extract_text(extraction_mode="layout")
        rows = statement_rows(text, profile)
        if rows:
            found.append((rows, number + 1))
    if len(found) != 1:
        raise ValueError("Expected exactly one supported quarterly income statement")
    return found[0]


def publication(text):
    page = Page(text)
    published = page.meta.get("published_time")
    if not published:
        match = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', text)
        published = match[1] if match else None
    if not published:
        raise ValueError("Release lacks a supported publication date")
    published = date.fromisoformat(published[:10])
    revisions = re.findall(r'"dateModified"\s*:\s*"(\d{4}-\d{2}-\d{2})', text)
    effective = max([published] + [date.fromisoformat(d) for d in revisions])
    return published, effective


def statement_date(raw, published, page_modified):
    metadata = PdfReader(io.BytesIO(raw)).metadata
    dates = [metadata.creation_date, metadata.modification_date] if metadata else []
    dates = [d.date() for d in dates if d is not None]
    return max([published] + dates) if dates else max(published, page_modified)


def collect_issuer(symbol, directory, profile):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = profile["issuer"]
    resources, failures = [], []

    def fetch(url, extension):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in source["allowed_hosts"]
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            raise ValueError("Financial document URL is outside configured issuer hosts")
        with build_opener(NoRedirect()).open(
            Request(url, headers={"User-Agent": "QuantRSI public financial research"}), timeout=25
        ) as response:
            raw = response.read(8_000_001)
        if len(raw) > 8_000_000:
            raise ValueError("Financial document exceeds 8 MB")
        if extension == "pdf" and not raw.startswith(b"%PDF-"):
            raise ValueError("Expected a PDF financial statement")
        sha = digest(raw)
        file = f"{sha}.{extension}"
        atomic_write(directory / file, raw)
        resources.append({"url": url.split("?")[0], "file": file, "sha256": sha})
        return raw

    index = fetch(source["index_url"], "html").decode("utf-8")
    if source["archive"] == "q4_public_feed":
        key = re.search(r"Q4ApiKey\s*=\s*'([^']+)'", index)
        if not key:
            raise ValueError("Issuer public feed configuration is unavailable")
        params = {
            "apiKey": key[1],
            "LanguageId": 1,
            "pageSize": 8,
            "pageNumber": 0,
            "year": -1,
            "excludeSelection": 1,
            "reportTypes": "First Quarter|Second Quarter|Third Quarter|Fourth Quarter",
        }
        feed_url = urljoin(source["index_url"], "/feed/FinancialReport.svc/GetFinancialReportList")
        feed = json.loads(fetch(feed_url + "?" + urlencode(params), "json"))
        releases = [
            d["DocumentPath"]
            for r in feed["GetFinancialReportListResult"][:8]
            for d in r["Documents"]
            if d["DocumentCategory"] == "news"
        ]
    else:
        releases = list(
            dict.fromkeys(
                urljoin(source["index_url"], link)
                for link in Page(index).links
                if re.search(source["release_pattern"], link)
            )
        )[:8]
    records = []
    for release in releases:
        try:
            text = fetch(release, "html").decode("utf-8")
            published, page_modified = publication(text)
            links = list(
                dict.fromkeys(
                    urljoin(release, link)
                    for link in Page(text).links
                    if re.search(source["pdf_pattern"], link, re.I)
                )
            )
            if len(links) != 1:
                raise ValueError("Expected one supported issuer financial PDF")
            raw = fetch(links[0], "pdf")
            rows, page_number = parse_statement(raw, source)
            effective = statement_date(raw, published, page_modified)
            available = datetime.combine(
                effective + timedelta(days=1), datetime.min.time(), timezone.utc
            ) + timedelta(hours=5)
            if any(date.fromisoformat(r["end"]) > published for r in rows):
                raise ValueError("Statement period ends after its release date")
            records += [
                row
                | {
                    "priority": 0,
                    "start": None,
                    "period_kind": "reported_three_months",
                    "published_date": str(published),
                    "page_modified_date": str(page_modified),
                    "revision_date": str(effective),
                    "available_at": available.isoformat(),
                    "statement_id": digest(raw),
                    "unit": "USD",
                    "source_url": links[0],
                    "release_url": release,
                    "page": page_number,
                }
                for row in rows
            ]
        except (OSError, ValueError, KeyError, PyPdfError) as exc:
            failures.append({"release_url": release, "error": str(exc)})
    if not records:
        atomic_write(directory / "issuer_failures.json", canonical(failures))
        raise ValueError("No usable quarterly issuer statements; see issuer_failures.json")
    return {
        "symbol": symbol,
        "provider": "issuer_earnings",
        "profile": profile,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "sha256": digest(canonical(records)),
        "resources": resources,
        "failures": failures,
        "notice": NOTICE,
    }
