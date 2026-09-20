"""
tests/unit/test_metrics.py
============================
Unit tests for infra/metrics.MetricsStore.
"""
from __future__ import annotations

import time

import pytest

from infra.metrics import MetricsStore, _DURATION_BUCKETS


@pytest.fixture
def store() -> MetricsStore:
    """Fresh MetricsStore per test — avoids singleton state pollution."""
    return MetricsStore()


class TestCounters:
    def test_cases_total_starts_at_zero(self, store: MetricsStore) -> None:
        assert store.cases_total.value == 0

    def test_inc_cases_total(self, store: MetricsStore) -> None:
        store.inc_cases_total()
        store.inc_cases_total()
        assert store.cases_total.value == 2

    def test_inc_cases_hitl(self, store: MetricsStore) -> None:
        store.inc_cases_hitl()
        assert store.cases_hitl.value == 1

    def test_inc_cases_failed(self, store: MetricsStore) -> None:
        store.inc_cases_failed()
        assert store.cases_failed.value == 1

    def test_inc_cases_complete(self, store: MetricsStore) -> None:
        store.inc_cases_complete()
        assert store.cases_complete.value == 1

    def test_hitl_review_approved(self, store: MetricsStore) -> None:
        store.inc_hitl_review(approved=True)
        assert store.hitl_reviews_total.value == 1
        assert store.hitl_approved.value == 1
        assert store.hitl_rejected.value == 0

    def test_hitl_review_rejected(self, store: MetricsStore) -> None:
        store.inc_hitl_review(approved=False)
        assert store.hitl_reviews_total.value == 1
        assert store.hitl_rejected.value == 1
        assert store.hitl_approved.value == 0

    def test_multiple_reviews_accumulated(self, store: MetricsStore) -> None:
        store.inc_hitl_review(approved=True)
        store.inc_hitl_review(approved=True)
        store.inc_hitl_review(approved=False)
        assert store.hitl_reviews_total.value == 3
        assert store.hitl_approved.value == 2
        assert store.hitl_rejected.value == 1


class TestGauge:
    def test_queue_pending_set(self, store: MetricsStore) -> None:
        store.set_queue_pending(5)
        assert store.queue_pending.value == 5

    def test_queue_pending_updates(self, store: MetricsStore) -> None:
        store.set_queue_pending(3)
        store.set_queue_pending(7)
        assert store.queue_pending.value == 7

    def test_queue_pending_to_zero(self, store: MetricsStore) -> None:
        store.set_queue_pending(10)
        store.set_queue_pending(0)
        assert store.queue_pending.value == 0


class TestHistogram:
    def test_observe_increments_count(self, store: MetricsStore) -> None:
        store.observe_pipeline_duration(2.5)
        snap = store.pipeline_duration.snapshot()
        assert snap["count"] == 1

    def test_observe_accumulates_sum(self, store: MetricsStore) -> None:
        store.observe_pipeline_duration(1.0)
        store.observe_pipeline_duration(3.0)
        snap = store.pipeline_duration.snapshot()
        assert snap["count"] == 2
        assert snap["sum"] == pytest.approx(4.0)

    def test_observe_bucket_placement(self, store: MetricsStore) -> None:
        store.observe_pipeline_duration(0.3)   # ≤ 0.5 bucket
        snap = store.pipeline_duration.snapshot()
        # First bucket (≤0.5) should have count 1
        first_count = snap["buckets"][0][1]
        assert first_count == 1

    def test_large_value_in_inf_bucket_only(self, store: MetricsStore) -> None:
        store.observe_pipeline_duration(9999.0)
        snap = store.pipeline_duration.snapshot()
        # Only the +Inf bucket (last) should have count 1
        for bound, count in snap["buckets"][:-1]:
            assert count == 0
        assert snap["buckets"][-1][1] == 1


class TestToDict:
    def test_has_required_keys(self, store: MetricsStore) -> None:
        data = store.to_dict()
        for key in (
            "uptime_seconds", "cases_total", "cases_hitl_triggered",
            "cases_failed", "hitl_reviews_total", "hitl_queue_pending",
            "pipeline_duration_seconds",
        ):
            assert key in data

    def test_values_reflect_increments(self, store: MetricsStore) -> None:
        store.inc_cases_total()
        store.inc_cases_total()
        store.inc_cases_hitl()
        data = store.to_dict()
        assert data["cases_total"] == 2
        assert data["cases_hitl_triggered"] == 1


class TestToPrometheus:
    def test_output_is_nonempty_string(self, store: MetricsStore) -> None:
        text = store.to_prometheus()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_ends_with_newline(self, store: MetricsStore) -> None:
        assert store.to_prometheus().endswith("\n")

    def test_contains_counter_names(self, store: MetricsStore) -> None:
        text = store.to_prometheus()
        assert "icsr_cases_total" in text
        assert "icsr_hitl_reviews_total" in text
        assert "icsr_hitl_queue_pending" in text

    def test_contains_histogram_buckets(self, store: MetricsStore) -> None:
        text = store.to_prometheus()
        assert "icsr_pipeline_duration_seconds_bucket" in text
        assert '+Inf"' in text

    def test_counter_value_in_output(self, store: MetricsStore) -> None:
        store.inc_cases_total()
        store.inc_cases_total()
        store.inc_cases_total()
        text = store.to_prometheus()
        assert "icsr_cases_total 3" in text

    def test_type_annotations_present(self, store: MetricsStore) -> None:
        text = store.to_prometheus()
        assert "# TYPE icsr_cases_total counter" in text
        assert "# TYPE icsr_hitl_queue_pending gauge" in text
        assert "# TYPE icsr_pipeline_duration_seconds histogram" in text

    def test_help_lines_present(self, store: MetricsStore) -> None:
        text = store.to_prometheus()
        assert "# HELP icsr_cases_total" in text
