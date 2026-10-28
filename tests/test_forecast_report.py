"""Offline presentation checks using the verified public report structure."""

import builtins
import copy
import json
import socket
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from quant_rsi.forecast.forecast_report import render_forecast_report


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.fields, self.rows, self.tags = {}, [], []
        self.label = None
        self.kind = None
        self.buffer = []
        self.row = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))
        if tag == "tr":
            self.row = []
        if tag in ("dt", "dd", "th", "td"):
            self.kind, self.buffer = tag, []

    def handle_data(self, data):
        if self.kind is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == self.kind:
            value = "".join(self.buffer)
            if tag == "dt":
                self.label = value
            elif tag == "dd":
                self.fields[self.label] = value
            else:
                self.row.append(value)
            self.kind = None
        elif tag == "tr":
            self.rows.append(self.row)


@pytest.fixture
def report():
    binding = dict(
        authority_ref="authority",
        campaign_ref="campaign",
        allocation_ref="allocation",
        exposure_ref="exposure",
        campaign_id="registered",
        instrument_id="SOXL",
        run_id="soxl-first",
        workspace="/workspace",
    )
    metrics = dict(mae_pp=2.5, rmse_pp=3.5, observations=75, negative_mae_pp=-2.5)
    return dict(
        workflow_id="soxl-first",
        status="complete",
        workspace="/workspace",
        managed_run=binding,
        config=dict(target="SOXL", industry="XSD", market="SPY"),
        config_ref="config",
        protocol_ref="protocol",
        experiment=dict(
            experiment_id="soxl-first",
            input_ref="input",
            protocol_ref="protocol",
            budget=dict(max_attempts=10, max_worker_seconds=1800, worker_timeout_seconds=180),
            fit_reservations=1,
            reserved_worker_seconds=180,
            attempts=[
                dict(
                    ordinal=0,
                    name="B0-fold-0",
                    status="completed",
                    worker_seconds=180,
                    elapsed_seconds=0.25,
                    error=None,
                ),
                dict(
                    ordinal=1,
                    name="B1-fold-0",
                    status="rejected",
                    worker_seconds=0,
                    elapsed_seconds=None,
                    error="fixed input unavailable",
                ),
            ],
            selection=dict(selected_recipe="B1", candidates={"B1": {"pooled": metrics}}),
            qualifications=["historical availability is an explicit assumption"],
            forecast=dict(
                instrument="SOXL",
                currency="USD",
                price_unit="raw_currency_per_share",
                price_unit_claim="explicit_assumption",
                status="forecast",
                reasons=[],
                origin_session="2026-10-23",
                origin_close_at="2026-10-23T20:00:00+00:00",
                target_session="2026-10-30",
                target_close_at="2026-10-30T20:00:00+00:00",
                horizon_sessions=5,
                origin_price=83.56,
                forecast_price=85.2312,
                forecast_return=0.02,
                recipe_group="B1",
                feature_names=["momentum_3", "momentum_7"],
                model_inputs=["target_prices"],
                sidecar_inputs_used_in_training=[],
                training_rows=691,
                model_ref="publication-model",
                feature_view_ref="view",
                forecast_ref="forecast",
                source_refs=["target-source"],
                mode="forward",
                forward=True,
                frozen_input_asof="2026-10-23T21:00:00+00:00",
                training_information_cutoff="2026-10-20T20:01:00+00:00",
                model_effective_at="2026-10-23T21:05:00+00:00",
                issued_at="2026-10-23T21:06:00+00:00",
                post_commit_confirmed_at="2026-10-23T21:06:01+00:00",
                forward_deadline="2026-10-26T13:30:00+00:00",
                publication_model_validation_metrics=None,
                publication_model_inherits_evaluation_scores=False,
                qualifications=["uncalibrated_point_forecast_no_probability_interval"],
                action_evidence=[],
            ),
        ),
        development_comparison=dict(
            selected_recipe="B1",
            selected_development_metrics=metrics,
            naive_controls=dict(
                unchanged_price=dict(mae_pp=2.0, rmse_pp=4.0, observations=75),
                training_mature_label_mean=dict(mae_pp=3.0, rmse_pp=3.0, observations=75),
            ),
            selected_beats_each_control=dict(
                unchanged_price=dict(mae_pp=False, rmse_pp=True),
                training_mature_label_mean=dict(mae_pp=True, rmse_pp=False),
            ),
            predictive_validity_established=False,
            qualification="Development comparison only; publication uses a separate model.",
        ),
        coverage=dict(
            inputs=["target_prices", "industry_proxy_prices", "market_proxy_prices"],
            news="not_collected_or_used",
            fundamentals="not_collected_or_used",
            corporate_actions="completeness unknown",
        ),
    )


def campaign_for(report):
    return dict(
        kind="ForecastCampaignReport",
        schema_version=1,
        campaign_id="registered",
        authority_ref="authority",
        campaign_ref="campaign",
        workspace="/workspace",
        budget=dict(max_fit_reservations=10, max_worker_seconds=1800),
        allocated=dict(max_attempts=10, max_worker_seconds=1800),
        actual=dict(fit_reservations=2, charged_fits=1, reserved_worker_seconds=180),
        runs=[
            dict(
                run_id=report["workflow_id"],
                binding=copy.deepcopy(report["managed_run"]),
                allocation=dict(max_attempts=10, max_worker_seconds=1800),
                actual=dict(fit_reservations=2, charged_fits=1, reserved_worker_seconds=180),
            )
        ],
    )


@pytest.mark.parametrize("roles", [("SOXL", "XSD", "SPY"), ("KO", "XLP", "SPY")])
def test_explicit_profile_origin_target_units_and_used_features(report, roles):
    target, industry, market = roles
    report["config"] = dict(target=target, industry=industry, market=market)
    report["experiment"]["forecast"]["instrument"] = target
    page = Page(render_forecast_report(report))
    assert page.fields["Target"] == target
    assert page.fields["Industry proxy"] == industry
    assert page.fields["Market proxy"] == market
    assert page.fields["Observed origin session"] == "2026-10-23"
    assert page.fields["Forecast target session"] == "2026-10-30"
    assert page.fields["Observed origin price"] == "83.5600"
    assert page.fields["Forecast price"] == "85.2312"
    assert page.fields["Forecast simple return"] == "2.0000%"
    assert page.fields["Currency"] == "USD"
    assert "momentum_3" in page.fields["Actual feature names"]
    assert "target_prices" in page.fields["Actual model input groups"]
    assert "industry_prices" not in page.fields["Actual model input groups"]
    assert page.fields["Publication-model validation metrics"] == "Unavailable"
    assert page.fields["Publication model inherits development scores"] == "False"


def test_each_failed_control_is_explicit_with_same_sample_metrics_and_counts(report):
    html = render_forecast_report(report)
    rows = Page(html).rows
    unchanged = next(row for row in rows if row[0] == "Unchanged-price control")
    mean = next(row for row in rows if row[0] == "Same-training-sample mean control")
    assert unchanged == [
        "Unchanged-price control",
        "2.0000",
        "4.0000",
        "75",
        "No — model did not beat this control",
        "Yes",
    ]
    assert mean == [
        "Same-training-sample mean control",
        "3.0000",
        "3.0000",
        "75",
        "Yes",
        "No — model did not beat this control",
    ]
    assert "MAE (pp)" in html and "RMSE (pp)" in html
    assert "separate fit" in html and "do not validate" in html
    assert Page(html).fields["Predictive validity established"] == "False"


def test_late_and_abstained_outputs_preserve_reasons_without_prediction(report):
    forecast = report["experiment"]["forecast"]
    forecast.update(
        status="abstain",
        mode="retrospective_or_abstained",
        forward=False,
        forecast_price=None,
        forecast_return=None,
        reasons=["unsupported_split", "origin_close_missing"],
    )
    page = Page(render_forecast_report(report))
    assert page.fields["Publication status"] == "abstain"
    assert page.fields["Recorded mode"] == "retrospective_or_abstained"
    assert page.fields["Confirmed forward qualification"] == "False"
    assert page.fields["Forecast price"] == page.fields["Forecast simple return"] == "Unavailable"
    assert "unsupported_split" in page.fields["Abstention or failure reasons"]
    assert page.fields["Forward deadline (UTC)"] == "2026-10-26T13:30:00+00:00"


def test_absent_publication_and_comparison_never_become_zero(report):
    report.update(status="exhausted_without_publication", development_comparison=None)
    report["experiment"]["forecast"] = None
    page = Page(render_forecast_report(report))
    assert page.fields["Status"] == "exhausted_without_publication"
    assert page.fields["Forecast price"] == "Unavailable"
    row = next(row for row in page.rows if row[0] == "Unchanged-price control")
    assert row[1:] == ["Unavailable"] * 5
    assert page.fields["Publication-model validation metrics"] == "Unavailable"


@pytest.mark.parametrize("recoverable", [True, False, None])
def test_pre_workflow_failure_and_only_explicit_recoverable_command(recoverable):
    report = dict(
        workflow_id="new",
        status="failed",
        config={},
        experiment=None,
        development_comparison=None,
        coverage={},
        failure=dict(
            category="network",
            stage="collection",
            message="HTTP request failed",
            recovery_command="quant-rsi forecast-workflow run new",
            recoverable=recoverable,
            budget_unavailable_reason="parent authority missing",
        ),
    )
    page = Page(render_forecast_report(report))
    assert page.fields["Failure stage"] == "collection"
    assert page.fields["Message"] == "HTTP request failed"
    assert page.fields["Budget unavailable reason"] == "parent authority missing"
    assert page.fields["Forecast price"] == "Unavailable"
    assert ("Supplied safe recovery command" in page.fields) is (recoverable is True)
    if recoverable is True:
        assert (
            page.fields["Supplied safe recovery command"] == report["failure"]["recovery_command"]
        )


def test_attempt_failure_keeps_registered_slots_separate_from_measured_runtime(report):
    page = Page(render_forecast_report(report))
    row = next(row for row in page.rows if row[0] == "1")
    assert row == ["1", "B1-fold-0", "rejected", "0", "Unavailable", "fixed input unavailable"]


def test_budget_uses_matching_parent_attempt_count_not_child_charged_count(report):
    parent = campaign_for(report)
    page = Page(render_forecast_report(report, campaign_report=parent))
    assert ["Attempt reservations", "2", "2"] in page.rows
    assert ["Charged fit slots", "1", "1"] in page.rows
    assert ["Charged worker-slot seconds", "180", "180"] in page.rows


@pytest.mark.parametrize("field", ["campaign_id", "authority_ref", "workspace"])
def test_budget_from_other_authority_is_rejected(report, field):
    campaign = campaign_for(report)
    campaign[field] = "other"
    with pytest.raises(ValueError, match="managed identity"):
        render_forecast_report(report, campaign_report=campaign)


def test_budget_copy_or_duplicate_run_binding_is_rejected(report):
    campaign = campaign_for(report)
    campaign["runs"][0]["binding"]["allocation_ref"] = "other"
    with pytest.raises(ValueError, match="exactly this managed run"):
        render_forecast_report(report, campaign_report=campaign)
    campaign = campaign_for(report)
    campaign["runs"].append(copy.deepcopy(campaign["runs"][0]))
    with pytest.raises(ValueError, match="exactly this managed run"):
        render_forecast_report(report, campaign_report=campaign)


def test_missing_and_nonfinite_metrics_are_unavailable_but_real_zero_is_displayed(report):
    metrics = report["development_comparison"]["selected_development_metrics"]
    metrics.update(mae_pp=float("nan"), rmse_pp=float("inf"), observations=None)
    report["development_comparison"]["naive_controls"]["unchanged_price"].update(
        mae_pp=0.0, rmse_pp=0.0, observations=0
    )
    page = Page(render_forecast_report(report))
    row = next(row for row in page.rows if row[0].startswith("Selected development model"))
    assert row[1:4] == ["Unavailable", "Unavailable", "Unavailable"]
    assert ["Unchanged-price control", "0.0000", "0.0000", "0"] == next(
        row[:4] for row in page.rows if row[0] == "Unchanged-price control"
    )


def test_dynamic_text_is_escaped_in_title_fields_controls_errors_and_recovery(report):
    hostile = '<script>alert("x")</script><img src=x onerror=alert(1)>&'
    report["config"]["target"] = hostile
    report["coverage"]["news"] = hostile
    report["experiment"]["forecast"]["feature_names"] = [hostile]
    report["development_comparison"]["selected_recipe"] = hostile
    report["experiment"]["attempts"][1]["error"] = hostile
    report["failure"] = dict(
        category=hostile,
        stage=hostile,
        message=hostile,
        recovery_command=hostile,
        recoverable=True,
        budget_unavailable_reason=hostile,
    )
    html = render_forecast_report(report)
    assert hostile not in html
    assert "&lt;script&gt;" in html and "&amp;" in html and "&quot;" in html
    page = Page(html)
    assert page.fields["Target"] == hostile
    assert page.fields["Supplied safe recovery command"] == hostile
    assert not any(tag in {"script", "img", "iframe", "link"} for tag, _ in page.tags)
    assert not any(key in {"src", "href"} for _, attrs in page.tags for key, _ in attrs)


def test_rendering_is_deterministic_and_has_no_io_or_input_mutation(report, monkeypatch):
    parent = campaign_for(report)
    before = copy.deepcopy((report, parent))

    def forbidden(*args, **kwargs):
        raise AssertionError("renderer attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    html = render_forecast_report(report, campaign_report=parent)
    assert html == render_forecast_report(report, campaign_report=parent)
    assert html.startswith("<!doctype html>") and html.endswith("</body></html>")
    assert (report, parent) == before


def test_late_forecast_keeps_price_without_becoming_forward(report):
    forecast = report["experiment"]["forecast"]
    forecast.update(
        forward=False,
        mode="retrospective_or_abstained",
        issued_at="2026-10-27T14:00:00+00:00",
        post_commit_confirmed_at="2026-10-27T14:00:01+00:00",
    )
    page = Page(render_forecast_report(report))
    assert page.fields["Publication status"] == "forecast"
    assert page.fields["Forecast price"] == "85.2312"
    assert page.fields["Confirmed forward qualification"] == "False"
    assert page.fields["Recorded mode"] == "retrospective_or_abstained"
    assert page.fields["Forecast issued at (UTC)"] == "2026-10-27T14:00:00+00:00"


def test_parent_ceiling_uses_retained_actual_campaign_budget_schema(report):
    evidence_path = Path(__file__).parents[1] / "docs/evidence/enhanced_forecast_20261027.json"
    evidence = json.loads(evidence_path.read_text())
    parent = campaign_for(report)
    parent["budget"] = evidence["budget"]
    parent["allocated"] = evidence["allocated"]
    page = Page(render_forecast_report(report, campaign_report=parent))
    assert ["Registered attempt ceiling", "20", "10"] in page.rows
    assert page.fields["Campaign allocated attempt ceiling"] == "20"
