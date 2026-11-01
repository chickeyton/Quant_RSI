"""Original small statement fixtures; public archives are mocked in automated tests."""

import io
import json
from datetime import date

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from test_research_study import command
from test_study_queue import OFFLINE, finish_queue

from quant_rsi.cli import main
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import issuer_financials as issuer
from quant_rsi.research import study, study_financials
from quant_rsi.research.study_workbench import Workbench

PROFILE = {
    "statement_title": "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "rows": {"revenue": "Total net sales", "operating_income": "Operating income"},
}
TEXT = """Example Company
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per-share amounts)
Three Months Ended Nine Months Ended
June 27, June 28, June 27, June 28,
2026 2025 2026 2025
To t a l n e t s a l e s (1) $ 1,200 $ 1,000 $ 9,000 $ 8,000
Operating income 120 80 600 500
"""


def test_quarterly_columns_and_spaced_labels_never_use_ytd():
    rows = issuer.statement_rows(TEXT, PROFILE)
    assert [r["value"] for r in rows] == [1.2e9, 1e9, 120e6, 80e6]
    assert {r["end"] for r in rows} == {"2026-06-27", "2025-06-28"}
    assert issuer.statement_rows(TEXT.replace("Three Months Ended", "Year Ended"), PROFILE) is None
    assert issuer.statement_rows("Unrelated reconciliation\n" + TEXT, PROFILE) == rows


@pytest.mark.parametrize(
    "before,after",
    [
        ("In millions", "In thousands"),
        ("1,200", "—"),
        ("1,200", "1, 2 0 0"),
        ("2026 2025", "2026 2023"),
        ("Operating income 120", "Operating profit 120"),
    ],
)
def test_ambiguous_or_incompatible_statement_rejected(before, after):
    with pytest.raises(ValueError):
        issuer.statement_rows(TEXT.replace(before, after), PROFILE)


def test_conflicting_repeated_rows_and_negative_operating_income():
    with pytest.raises(ValueError, match="conflicting"):
        issuer.statement_rows(TEXT + "Total net sales 5 4 3 2", PROFILE)
    rows = issuer.statement_rows(TEXT.replace("income 120", "income (120)"), PROFILE)
    assert rows[2]["value"] == -120e6


def test_real_pdf_text_extraction_from_original_fixture():
    writer = PdfWriter()
    page = writer.add_blank_page(width=700, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    lines = [line.replace("(", r"\(").replace(")", r"\)") for line in TEXT.splitlines()]
    stream.set_data(
        (
            "BT /F1 10 Tf 20 760 Td 18 TL " + " T* ".join(f"({line}) Tj" for line in lines) + " ET"
        ).encode()
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    raw = io.BytesIO()
    writer.write(raw)
    rows, page_number = issuer.parse_statement(raw.getvalue(), PROFILE)
    assert page_number == 1
    assert rows == issuer.statement_rows(TEXT, PROFILE)
    published, maintained = date(2026, 7, 30), date(2026, 8, 19)
    assert issuer.statement_date(raw.getvalue(), published, maintained) == maintained
    writer.add_metadata({"/ModDate": "D:20260730171504Z"})
    raw = io.BytesIO()
    writer.write(raw)
    assert issuer.statement_date(raw.getvalue(), published, maintained) == published
    writer.add_metadata({"/ModDate": "D:20260820171504Z"})
    raw = io.BytesIO()
    writer.write(raw)
    assert issuer.statement_date(raw.getvalue(), published, maintained) == date(2026, 8, 20)


def test_publication_uses_disclosed_revision_and_rejects_missing_date():
    assert issuer.publication('<meta name="published_time" content="2026-07-30">') == (
        date(2026, 7, 30),
        date(2026, 7, 30),
    )
    assert issuer.publication('"datePublished":"2026-07-30Z", "dateModified":"2026-08-19Z"') == (
        date(2026, 7, 30),
        date(2026, 8, 19),
    )
    with pytest.raises(ValueError, match="publication date"):
        issuer.publication("No date")


def test_fallback_retains_attempts_raw_identity_and_explicit_source(tmp_path, monkeypatch):
    profile = {"symbol": "KO", "cik": 1, "issuer": PROFILE}
    profiles = tmp_path / "profiles/US/KO"
    profiles.mkdir(parents=True)
    (profiles / "financials.json").write_bytes(canonical(profile))
    directory = tmp_path / "retained"

    def unavailable(*args):
        raise OSError("SEC denied fixture request")

    def collect(symbol, directory, profile):
        raw = b"original fixture statement"
        (directory / "statement.pdf").write_bytes(raw)
        records = issuer.statement_rows(TEXT, PROFILE)
        return {
            "symbol": symbol,
            "profile": profile,
            "provider": "issuer_earnings",
            "records": records,
            "sha256": digest(canonical(records)),
            "resources": [{"file": "statement.pdf", "sha256": digest(raw)}],
        }

    monkeypatch.setattr(study_financials, "collect_sec", unavailable)
    monkeypatch.setattr(issuer, "collect_issuer", collect)
    snapshot = study_financials.collect_financials("KO", directory, tmp_path / "profiles")
    assert [a["status"] for a in snapshot["source_attempts"]] == ["unavailable", "available"]
    assert study_financials.collect_financials("KO", directory, tmp_path / "profiles") == snapshot
    with pytest.raises(ValueError, match="provider differs"):
        study_financials.collect_financials("KO", directory, tmp_path / "profiles", "sec")
    (directory / "statement.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="document identity"):
        study_financials.collect_financials("KO", directory, tmp_path / "profiles")
    monkeypatch.setattr(issuer, "collect_issuer", unavailable)
    with pytest.raises(OSError):
        study_financials.collect_financials("KO", tmp_path / "failed", tmp_path / "profiles")
    attempts = json.loads((tmp_path / "failed/source_attempts.json").read_bytes())
    assert [a["status"] for a in attempts] == ["unavailable", "unavailable"]


def test_input_check_never_constructs_llm_fits_or_registers_forecast(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Input check attempted a paid call or fit")

    monkeypatch.setattr(study, "LLMHypothesizer", forbidden)
    monkeypatch.setattr(study.LightGBMDemo, "predict", forbidden)
    root = tmp_path / "check"
    assert (
        main(
            command(
                root,
                "--llm",
                "codex",
                "--financials",
                "--news-research",
                "--check-inputs",
                "--no-ui",
            )
        )
        == 0
    )
    report = json.loads((root / "input_report.json").read_bytes())
    assert report["financials"]["coverage"]["runnable_factors"] == 6
    assert not (root / "result.json").exists()
    runtime = json.loads((root / "runtime.json").read_bytes())
    assert runtime["status"] == "completed"
    assert runtime["usage"] == []
    assert runtime["counts"]["llm_requests_started"] == 0
    assert runtime["counts"]["model_stages_started"] == 0
    assert "Executable financial factors" in (root / "input_report.html").read_text()


def test_workbench_checks_are_separate_from_research_and_not_deduplicated(tmp_path):
    workbench = Workbench(tmp_path)
    for _ in range(2):
        batch = workbench.batch(
            ["KO", "AAPL"],
            OFFLINE
            | {
                "financials": True,
                "financials_source": "issuer",
                "only_new_evidence": True,
            },
            check_inputs=True,
        )
        assert len(batch["tasks"]) == 2
        finish_queue(workbench)
    tasks = workbench.listing()
    assert len(tasks) == 4
    assert all(t["status"] == "completed" and t["action"] == "check_inputs" for t in tasks)
    assert not list(tmp_path.glob("tasks/*/trigger.json"))
    assert not list(tmp_path.glob("tasks/*/study/result.json"))
    assert all(not h["entries"] for h in workbench.overview()["histories"].values())
