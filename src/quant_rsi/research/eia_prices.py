"""Daily observations from dated EIA Table 11, excluding its weekly averages."""

import csv
import io
import math
import re
from datetime import date, datetime, timedelta, timezone

PRICE_FIELDS = {"wti": "WTI - Cushing", "brent": "Brent"}


def parse_prices(raw, released, week_ending):
    """Compare the last daily quote with its fifth preceding available observation."""
    if not 0 < (released - week_ending).days <= 14:
        raise ValueError("EIA price archive week does not precede its release")
    rows = list(csv.reader(io.StringIO(raw.decode("cp1252"))))
    if len(rows) > 100:
        raise ValueError("Unsupported EIA Table 11 layout")
    headers = [
        i
        for i, row in enumerate(rows)
        if row[:3] == ["STUB_1", "STUB_2", "STUB_3"]
        and 14 <= len(row) <= 16
        and re.fullmatch(r"[A-Z][a-z]{2} \d{1,2}/\d{1,2}", row[3])
    ]
    if len(headers) != 1:
        raise ValueError("Missing unambiguous EIA recent-price section")
    header = headers[0]
    labels = rows[header]
    width = len(labels) - int(labels[-1] == "")
    if width not in (14, 15):
        raise ValueError("Unsupported EIA recent-price width")
    # Some CSVs omit one weekly-average column or append an empty column.
    # The final ten dated columns are the two Monday-Friday daily blocks.
    daily_start = width - 10
    if any(not cell.startswith("Fri ") for cell in labels[3:daily_start]):
        raise ValueError("Unsupported EIA weekly-average columns")
    dates = []
    for index, cell in enumerate(labels[daily_start:width]):
        match = re.fullmatch(r"([A-Z][a-z]{2}) (\d{1,2})/(\d{1,2})", cell)
        if not match:
            raise ValueError("Unsupported EIA daily-price date")
        observed = date(released.year, int(match[2]), int(match[3]))
        if observed > released:
            observed = observed.replace(year=observed.year - 1)
        if (
            not 0 < (released - observed).days <= 40
            or observed.strftime("%a") != match[1]
            or observed.weekday() != index % 5
        ):
            raise ValueError("EIA daily-price dates do not match release year")
        dates.append(observed)
    if dates != sorted(set(dates)) or not 0 <= (week_ending - dates[-1]).days <= 3:
        raise ValueError("EIA price observations do not match the archive week")
    available = datetime.combine(released + timedelta(days=1), datetime.min.time(), timezone.utc)
    available += timedelta(hours=5)
    records = []
    for field, label in PRICE_FIELDS.items():
        selected = [
            row
            for row in rows[header + 1 :]
            if len(row) >= 2 and row[0].strip() == "Crude Oil" and row[1].strip() == label
        ]
        if len(selected) != 1 or len(selected[0]) != len(labels) or any(selected[0][width:]):
            raise ValueError(f"Missing or duplicate EIA daily-price row: {label}")
        observations = []
        for observed, cell in zip(dates, selected[0][daily_start:width], strict=True):
            if cell.strip() in {"", "-", "–", "—", "NA", "N/A"}:
                continue
            value = float(cell.replace(",", ""))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid EIA oil price: {label}")
            observations.append((observed, value))
        if len(observations) < 6 or observations[-1][0] != dates[-1]:
            continue
        prior, current = observations[-6], observations[-1]
        records.append(
            {
                "field": field,
                "observation_date": current[0].isoformat(),
                "prior_observation_date": prior[0].isoformat(),
                "released_on": released.isoformat(),
                "available_at": available.isoformat(),
                "value": current[1],
                "prior_value": prior[1],
                "unit": "usd_per_barrel",
                "row_label": label,
                "daily_observation_dates": [d.isoformat() for d, _ in observations[-6:]],
            }
        )
    return records
