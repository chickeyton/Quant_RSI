import pytest

from quant_rsi.core.journal import BudgetExhausted, Journal


def test_reservation_and_completion_conserve_budget(tmp_path):
    journal = Journal(tmp_path, "manifest")
    try:
        journal.reserve(0, "candidate", "proposal", 1)
        with pytest.raises(BudgetExhausted):
            journal.reserve(1, "second_candidate", "other", 1)
        with pytest.raises(ValueError, match="invalid attempt state transition"):
            journal.finish(0, "completed", result_ref="result")
        journal.start(0, 20, 20)
        journal.finish(0, "failed", error="expected_failure", elapsed=0.5)
        events = journal.events()
        journal.finish(0, "failed", error="expected_failure", elapsed=0.5)
        assert journal.events() == events
        assert journal.attempts()[0]["worker_seconds"] == 20
        with pytest.raises(ValueError, match="conflicting completion"):
            journal.finish(0, "completed", result_ref="result")
        with pytest.raises(BudgetExhausted):
            journal.reserve(1, "retry", "proposal", 1)
    finally:
        journal.close()
