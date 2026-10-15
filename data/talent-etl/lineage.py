"""
Run-level lineage and observability.

A :class:`RunContext` is created once per pipeline execution. Every step
records what it read, what it wrote and how many rows moved, so a manifest at
``_lineage/run_date=<d>/run_id=<id>.json`` answers "where did this Gold row come
from?" and "what happened in this run?". The same manifest is mirrored to
``_lineage/latest.json`` for quick inspection.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterator

from storage import Lake

log = logging.getLogger(__name__)


@dataclass
class StepRecord:
    """Lineage for one transformation step (one output dataset)."""

    layer: str
    dataset: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    rows_read: int = 0
    rows_written: int = 0
    rows_quarantined: int = 0
    status: str = "running"
    started_at: str = ""
    duration_s: float = 0.0
    details: dict[str, object] = field(default_factory=dict)


class RunContext:
    """Collects step lineage for one run and persists it as a manifest."""

    def __init__(self, lake: Lake, run_date: str, lineage_prefix: str) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        self.run_date = run_date
        self.started_at = datetime.now(timezone.utc)
        self.steps: list[StepRecord] = []
        self.status = "running"
        self._lake = lake
        self._prefix = lineage_prefix

    @contextmanager
    def step(self, layer: str, dataset: str) -> Iterator[StepRecord]:
        """Time and record one step; marks it failed (and re-raises) on error."""
        record = StepRecord(layer=layer, dataset=dataset, started_at=datetime.now(timezone.utc).isoformat())
        self.steps.append(record)
        t0 = time.perf_counter()
        try:
            yield record
            record.status = "succeeded"
        except Exception as exc:
            record.status = "failed"
            record.details["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.duration_s = round(time.perf_counter() - t0, 3)
            log.info(
                "step layer=%s dataset=%s status=%s read=%d written=%d quarantined=%d duration_s=%.3f",
                layer, dataset, record.status, record.rows_read, record.rows_written,
                record.rows_quarantined, record.duration_s,
            )

    def manifest(self) -> dict[str, object]:
        """Serializable summary of the run."""
        ended = datetime.now(timezone.utc)
        return {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_s": round((ended - self.started_at).total_seconds(), 3),
            "totals": {
                layer: {
                    "datasets": sum(1 for s in self.steps if s.layer == layer),
                    "rows_written": sum(s.rows_written for s in self.steps if s.layer == layer),
                    "rows_quarantined": sum(s.rows_quarantined for s in self.steps if s.layer == layer),
                }
                for layer in dict.fromkeys(s.layer for s in self.steps)
            },
            "steps": [asdict(s) for s in self.steps],
        }

    def save(self, status: str) -> str:
        """Persist the manifest; returns its URI."""
        self.status = status
        payload = self.manifest()
        key = f"{self._prefix}run_date={self.run_date}/run_id={self.run_id}.json"
        self._lake.put_json(key, payload)
        self._lake.put_json(f"{self._prefix}latest.json", payload)
        log.info("run run_id=%s status=%s duration_s=%s manifest=%s",
                 self.run_id, status, payload["duration_s"], self._lake.uri(key))
        return self._lake.uri(key)
