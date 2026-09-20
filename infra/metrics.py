"""
infra/metrics.py
=================
Lightweight in-process metrics store with Prometheus text exposition format.

WHY NO prometheus_client DEPENDENCY:
  - Zero extra dependency; the text format is trivial to produce correctly.
  - Thread-safe using simple locks (acceptable for synchronous Uvicorn workers).
  - Drop-in compatible: Prometheus, Grafana Agent, and VictoriaMetrics all
    scrape the same text/plain; version=0.0.4 format.

METRICS EXPOSED:
  Counters:
    icsr_cases_total            — Total cases submitted to /cases/run
    icsr_cases_complete_total   — Cases that completed without HITL
    icsr_cases_hitl_total       — Cases that triggered HITL interrupt
    icsr_cases_failed_total     — Cases that raised pipeline errors
    icsr_hitl_reviews_total     — HITL corrections submitted via /submit
    icsr_hitl_approved_total    — Reviews where approved=True
    icsr_hitl_rejected_total    — Reviews where approved=False

  Gauges:
    icsr_hitl_queue_pending     — Current length of the HITL queue

  Histograms:
    icsr_pipeline_duration_seconds — Pipeline wall-clock time (bucketed)

Usage:
  from infra.metrics import metrics

  # Record events
  metrics.inc_cases_total()
  metrics.inc_cases_hitl()
  metrics.observe_pipeline_duration(elapsed_seconds)
  metrics.set_queue_pending(len(hitl_queue))

  # Export
  text = metrics.to_prometheus()   # returns str in Prometheus text format
  data = metrics.to_dict()         # returns dict for JSON /stats endpoint
"""
from __future__ import annotations

import threading
import time
from typing import Any


# Histogram bucket boundaries (seconds)
_DURATION_BUCKETS = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf")]


class _Counter:
    """Thread-safe monotonically increasing counter."""

    def __init__(self) -> None:
        self._value = 0
        self._lock  = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value


class _Gauge:
    """Thread-safe gauge (can go up or down)."""

    def __init__(self) -> None:
        self._value = 0
        self._lock  = threading.Lock()

    def set(self, value: int | float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: int = 1) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> int | float:
        return self._value


class _Histogram:
    """Thread-safe histogram with configurable buckets."""

    def __init__(self, buckets: list[float]) -> None:
        self._buckets  = sorted(buckets)
        self._counts   = [0] * len(self._buckets)   # cumulative ≤ bucket upper bound
        self._total    = 0.0
        self._count    = 0
        self._lock     = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._total += value
            for i, bound in enumerate(self._buckets):
                if value <= bound:
                    self._counts[i] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "buckets": list(zip(self._buckets, self._counts)),
                "sum":     self._total,
                "count":   self._count,
            }


# ─────────────────────────────────────────────────────────────────────────────
# MetricsStore
# ─────────────────────────────────────────────────────────────────────────────

class MetricsStore:
    """Central singleton metrics registry for the ICSR HITL server."""

    def __init__(self) -> None:
        self._start_time = time.time()

        # Counters
        self.cases_total          = _Counter()
        self.cases_complete       = _Counter()
        self.cases_hitl           = _Counter()
        self.cases_failed         = _Counter()
        self.hitl_reviews_total   = _Counter()
        self.hitl_approved        = _Counter()
        self.hitl_rejected        = _Counter()

        # Gauges
        self.queue_pending        = _Gauge()

        # Histograms
        self.pipeline_duration    = _Histogram(_DURATION_BUCKETS)

    # ── Convenience increment methods ─────────────────────────────────────────

    def inc_cases_total(self)    -> None: self.cases_total.inc()
    def inc_cases_complete(self) -> None: self.cases_complete.inc()
    def inc_cases_hitl(self)     -> None: self.cases_hitl.inc()
    def inc_cases_failed(self)   -> None: self.cases_failed.inc()

    def inc_hitl_review(self, approved: bool) -> None:
        self.hitl_reviews_total.inc()
        if approved:
            self.hitl_approved.inc()
        else:
            self.hitl_rejected.inc()

    def set_queue_pending(self, n: int) -> None:
        self.queue_pending.set(n)

    def observe_pipeline_duration(self, seconds: float) -> None:
        self.pipeline_duration.observe(seconds)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return all metrics as a plain dict (for /api/v1/stats JSON response)."""
        hist = self.pipeline_duration.snapshot()
        return {
            "uptime_seconds":             round(time.time() - self._start_time, 1),
            "cases_total":                self.cases_total.value,
            "cases_complete":             self.cases_complete.value,
            "cases_hitl_triggered":       self.cases_hitl.value,
            "cases_failed":               self.cases_failed.value,
            "hitl_reviews_total":         self.hitl_reviews_total.value,
            "hitl_reviews_approved":      self.hitl_approved.value,
            "hitl_reviews_rejected":      self.hitl_rejected.value,
            "hitl_queue_pending":         self.queue_pending.value,
            "pipeline_duration_seconds":  {
                "count": hist["count"],
                "sum":   round(hist["sum"], 3),
                "mean":  round(hist["sum"] / hist["count"], 3) if hist["count"] else 0.0,
            },
        }

    def to_prometheus(self) -> str:
        """
        Return metrics in Prometheus text exposition format 0.0.4.
        https://prometheus.io/docs/instrumenting/exposition_formats/
        """
        lines: list[str] = []

        def _counter(name: str, help_text: str, value: int) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        def _gauge(name: str, help_text: str, value: int | float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        def _histogram(name: str, help_text: str, hist: dict) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} histogram")
            for upper, count in hist["buckets"]:
                le = "+Inf" if upper == float("inf") else str(upper)
                lines.append(f'{name}_bucket{{le="{le}"}} {count}')
            lines.append(f"{name}_sum {hist['sum']}")
            lines.append(f"{name}_count {hist['count']}")

        _counter("icsr_cases_total",          "Total ICSR cases submitted",              self.cases_total.value)
        _counter("icsr_cases_complete_total",  "Cases completed without HITL",           self.cases_complete.value)
        _counter("icsr_cases_hitl_total",      "Cases that triggered HITL interrupt",    self.cases_hitl.value)
        _counter("icsr_cases_failed_total",    "Cases that raised pipeline errors",      self.cases_failed.value)
        _counter("icsr_hitl_reviews_total",    "HITL corrections submitted",             self.hitl_reviews_total.value)
        _counter("icsr_hitl_approved_total",   "HITL reviews approved",                  self.hitl_approved.value)
        _counter("icsr_hitl_rejected_total",   "HITL reviews rejected",                  self.hitl_rejected.value)
        _gauge(  "icsr_hitl_queue_pending",    "Current HITL queue depth",               self.queue_pending.value)
        _gauge(  "icsr_process_uptime_seconds","Process uptime in seconds",              round(time.time() - self._start_time, 1))
        _histogram("icsr_pipeline_duration_seconds", "Pipeline wall-clock time",         self.pipeline_duration.snapshot())

        return "\n".join(lines) + "\n"


# Module-level singleton — import and use directly
metrics = MetricsStore()
