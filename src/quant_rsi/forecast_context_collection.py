"""Collect a roles-bound, resumable context audit package; never ingest or fit.

The caller owns the workflow lock. Each source has at most three recorded attempts,
a 15-second socket timeout, and an 8 MiB response limit. A saved successful receipt
is reused without HTTP. An interrupted body without a receipt is retained as an
uncommitted attempt, never assigned a fabricated receipt or historical clock.
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from quant_rsi import daily_data
from quant_rsi.artifacts import atomic_write, canonical, decode, digest
from quant_rsi.forecast_context_data import CALENDAR_SOURCES
from quant_rsi.market_contracts import timestamp

ROLE_NAMES = ("target", "industry", "market")
MAX_ATTEMPTS = 3
MAX_RESPONSE_BYTES = daily_data.MAX_RESPONSE_BYTES
TIMEOUT_SECONDS = 15


def _now():
    return datetime.now(timezone.utc)


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or non-regular collection evidence: {path.name}")
    return path.read_bytes()


def _write_new(path, body):
    if path.exists() or path.is_symlink():
        if _read(path) != body:
            raise ValueError(f"Collection evidence conflict: {path.name}")
    else:
        atomic_write(path, body)


def _roles(value):
    if not isinstance(value, dict) or set(value) != set(ROLE_NAMES):
        raise ValueError("roles must contain exactly target, industry, and market")
    result = {}
    for role in ROLE_NAMES:
        ticker = value[role]
        if not isinstance(ticker, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{0,14}", ticker):
            raise ValueError(f"Invalid plain US equity/ETF ticker for {role}")
        result[role] = ticker.upper()
    return result


def _sources(roles):
    charts = [
        {
            "name": f"{ticker}_chart_5y.json",
            "url": daily_data.CHART_URL
            + quote(ticker, safe="")
            + "?range=5y&interval=1d&events=div%2Csplits",
            "ticker": ticker,
        }
        for ticker in dict.fromkeys(roles.values())
    ]
    return charts + [
        {"name": name, "url": url, "ticker": None} for name, url in CALENDAR_SOURCES.items()
    ]


def _config(roles):
    return {
        "kind": "forecast_context_collection",
        "schema_version": 1,
        "roles": roles,
        "roles_ref": digest(canonical(roles)),
        "sources": _sources(roles),
        "max_attempts_per_source": MAX_ATTEMPTS,
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "socket_timeout_seconds": TIMEOUT_SECONDS,
    }


def _members(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Collection evidence contains a symbolic link")
        if path.is_file() and path.name != "manifest.json":
            body = _read(path)
            result.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(body),
                    "sha256": digest(body),
                }
            )
        elif not path.is_file() and not path.is_dir():
            raise ValueError("Collection evidence is not a regular file or directory")
    return result


def _validate_chart(body, source):
    if source["ticker"] is not None:
        _, _, quotes = daily_data._chart(body, source["ticker"])
        # Missing/incomplete bars are decided with the official calendar downstream.
        # Keep all returned rows/actions; do not turn a null bar into a usable price.
        for column in ("open", "close"):
            for value in quotes[column]:
                if value is not None:
                    daily_data._price(value)


def _receipt(attempt, source, config):
    intent_body = _read(attempt / "intent.json")
    intent = decode(intent_body)
    expected = {
        "source": source,
        "roles_ref": config["roles_ref"],
        "attempt": int(attempt.name),
        "requested_at": intent.get("requested_at"),
    }
    if intent != expected or timestamp(intent["requested_at"]) > _now():
        raise ValueError("Collection attempt identity or clock mismatch")
    receipt_path = attempt / "receipt.json"
    body_path = attempt / "body"
    if not receipt_path.exists():
        # An actual raw prefix may survive a crash; it is not a completed receipt.
        if body_path.exists():
            if len(_read(body_path)) > MAX_RESPONSE_BYTES + 1:
                raise ValueError("Interrupted response exceeds the recorded size limit")
        return None
    receipt_body = _read(receipt_path)
    receipt = decode(receipt_body)
    if (
        receipt.get("kind") != "forecast_context_response"
        or receipt.get("schema_version") != 1
        or receipt.get("source_name") != source["name"]
        or receipt.get("url") != source["url"]
        or receipt.get("roles_ref") != config["roles_ref"]
        or receipt.get("intent_ref") != digest(intent_body)
        or receipt.get("requested_at") != intent["requested_at"]
        or timestamp(receipt["received_at"]) < timestamp(intent["requested_at"])
        or timestamp(receipt["received_at"]) > _now()
    ):
        raise ValueError("Collection response identity or clock mismatch")
    if receipt["sha256"] is None:
        if body_path.exists() or receipt["bytes"] != 0 or receipt["error"] is None:
            raise ValueError("Collection receipt claims a missing response body")
        return None
    body = _read(body_path)
    if (
        len(body) > MAX_RESPONSE_BYTES + 1
        or receipt["bytes"] != len(body)
        or receipt["sha256"] != digest(body)
        or receipt["truncated"] != (len(body) > MAX_RESPONSE_BYTES)
    ):
        raise ValueError("Collection response body/hash mismatch")
    if receipt["error"] is not None:
        return None
    if receipt["status"] != 200 or receipt["truncated"] or not body:
        raise ValueError("Invalid successful collection receipt")
    _validate_chart(body, source)
    return body, receipt_body


def _pending(path):
    """Atomic-write remnants are evidence, never a committed body or receipt."""
    if not path.name.startswith(".pending-"):
        return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RESPONSE_BYTES + 1:
        raise ValueError("Invalid or oversized uncommitted collection evidence")
    return True


def _inspect(root, config):
    """Validate every saved source before repairing publication or starting HTTP."""
    sources = config["sources"]
    allowed = {"collection.json", "attempts", "raw", "manifest.json"}
    allowed.update(source["name"] + ".receipt.json" for source in sources)
    if any(
        not _pending(path) and (path.name not in allowed or path.is_symlink())
        for path in root.iterdir()
    ):
        raise ValueError("Unexpected or linked collection output")
    attempts_root = root / "attempts"
    raw_root = root / "raw"
    for path in (attempts_root, raw_root):
        if path.exists() and not path.is_dir():
            raise ValueError("Collection directory was replaced")
    names = {source["name"] for source in sources}
    if attempts_root.exists() and any(p.name not in names for p in attempts_root.iterdir()):
        raise ValueError("Unexpected source attempt directory")
    if raw_root.exists() and any(
        not _pending(p) and p.name not in names for p in raw_root.iterdir()
    ):
        raise ValueError("Unexpected raw response")
    states = {}
    for source in sources:
        directory = attempts_root / source["name"]
        attempts = []
        if directory.exists():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("Invalid source attempt directory")
            attempts = sorted(directory.iterdir())
        if [p.name for p in attempts] != [f"{i:04d}" for i in range(1, len(attempts) + 1)]:
            raise ValueError("Missing or unexpected collection attempt")
        if len(attempts) > MAX_ATTEMPTS:
            raise ValueError("Collection attempt limit exceeded")
        success = None
        for attempt in attempts:
            if attempt.is_symlink() or not attempt.is_dir():
                raise ValueError("Invalid collection attempt")
            entries = [p for p in attempt.iterdir() if not _pending(p)]
            if any(
                p.name not in {"intent.json", "body", "receipt.json"}
                or p.is_symlink()
                or not p.is_file()
                for p in entries
            ):
                raise ValueError("Unexpected attempt evidence")
            if success is not None:
                raise ValueError("A successful source has unexpected later attempts")
            if not entries:
                # A crash between mkdir and the pre-request intent consumes an attempt.
                continue
            current = _receipt(attempt, source, config)
            if current is not None:
                success = current
        raw = raw_root / source["name"]
        published_receipt = root / (source["name"] + ".receipt.json")
        if published_receipt.exists() and not raw.is_file():
            raise ValueError("A published receipt has no preceding body")
        for path, index in ((raw, 0), (published_receipt, 1)):
            if path.exists() or path.is_symlink():
                if success is None or _read(path) != success[index]:
                    raise ValueError("Published source differs from recorded successful receipt")
        states[source["name"]] = (len(attempts), success)
    return states


def _download(root, source, config, number):
    attempt = root / "attempts" / source["name"] / f"{number:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    intent = {
        "source": source,
        "roles_ref": config["roles_ref"],
        "attempt": number,
        "requested_at": _now().isoformat(),
    }
    intent_body = canonical(intent)
    _write_new(attempt / "intent.json", intent_body)
    body = None
    response = None
    error = None
    status = None
    final_url = None
    headers = {}
    caught = None
    try:
        request = Request(
            source["url"], headers={"User-Agent": "Mozilla/5.0 QuantRSI daily research"}
        )
        try:
            response = urlopen(request, timeout=TIMEOUT_SECONDS)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.status
            final_url = response.geturl()
            headers = dict(response.headers.items())
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if status != 200:
            raise ValueError(f"Public source HTTP {status}")
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Public response exceeds 8 MiB; bounded prefix preserved")
        if not body:
            raise ValueError("Public source returned an empty response")
        _validate_chart(body, source)
    except BaseException as exc:
        caught = exc
        error = {"type": type(exc).__name__, "message": str(exc)}
    receipt = {
        "kind": "forecast_context_response",
        "schema_version": 1,
        "source_name": source["name"],
        "url": source["url"],
        "final_url": final_url,
        "roles_ref": config["roles_ref"],
        "intent_ref": digest(intent_body),
        "requested_at": intent["requested_at"],
        "received_at": _now().isoformat(),
        "status": status,
        "headers": headers,
        "bytes": len(body) if body is not None else 0,
        "sha256": digest(body) if body is not None else None,
        "truncated": body is not None and len(body) > MAX_RESPONSE_BYTES,
        "error": error,
    }
    if body is not None:
        _write_new(attempt / "body", body)
    receipt_body = canonical(receipt)
    _write_new(attempt / "receipt.json", receipt_body)
    if caught is not None:
        if not isinstance(caught, Exception):
            raise caught
        raise ValueError(
            f"Context source failed: {source['name']}; evidence: {attempt}"
        ) from caught
    return body, receipt_body


def collect_context(output_dir, *, roles) -> Path:
    """Return a frozen audit directory; the caller must serialize this workflow.

    Repeated calls with the same roles validate and reuse saved success. Failed or
    interrupted attempts remain visible and consume their finite attempt slots.
    The resulting raw bytes retain every provider action and its original units;
    calendar, historical-availability assumptions, action eligibility, and the
    close-plus-15-minute completion policy are enforced by the data adapter.
    """
    config = _config(_roles(roles))
    root = Path(output_dir).absolute()
    if root.is_symlink():
        raise ValueError("Collection output must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "collection.json"
    if config_path.exists():
        if decode(_read(config_path)) != config:
            raise ValueError("Collection roles or source policy changed")
    else:
        if any(not _pending(p) for p in root.iterdir()):
            raise ValueError("Uninitialized collection directory contains evidence")
        _write_new(config_path, canonical(config))
    states = _inspect(root, config)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = decode(_read(manifest_path))
        if (
            manifest.get("kind") != "forecast_context_source_manifest"
            or manifest.get("schema_version") != 1
            or manifest.get("roles_ref") != config["roles_ref"]
            or manifest.get("files") != _members(root)
            or any(success is None for _, success in states.values())
            or any(
                not (root / "raw" / source["name"]).is_file()
                or not (root / (source["name"] + ".receipt.json")).is_file()
                for source in config["sources"]
            )
        ):
            raise ValueError("Frozen context collection manifest mismatch")
        return root
    for source in config["sources"]:
        count, success = states[source["name"]]
        if success is None:
            if count >= MAX_ATTEMPTS:
                raise ValueError(f"Context source attempt budget exhausted: {source['name']}")
            success = _download(root, source, config, count + 1)
        body, receipt = success
        _write_new(root / "raw" / source["name"], body)
        _write_new(root / (source["name"] + ".receipt.json"), receipt)
    manifest = {
        "kind": "forecast_context_source_manifest",
        "schema_version": 1,
        "roles_ref": config["roles_ref"],
        "files": _members(root),
    }
    _write_new(manifest_path, canonical(manifest))
    return root
