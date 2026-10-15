"""
Source-system connectivity: PostgreSQL (HR system of record).

Reads use ``COPY ... TO STDOUT`` so Bronze receives the exact text the source
emitted. A Bronze load reads every table inside one read-only REPEATABLE READ
transaction, so all seven tables come from the same consistent snapshot and
the ``count(*)`` used for reconciliation matches the rows exported.
Connections are retried with exponential backoff because Aurora Serverless
may be resuming from zero capacity on first touch.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import pandas as pd
import psycopg
from psycopg import IsolationLevel, sql

from config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class TableExport:
    """One table exported from the source."""

    frame: pd.DataFrame      # every column as string, exactly as exported
    csv_bytes: bytes         # the raw COPY output (byte-identical source export)
    source_row_count: int    # SELECT count(*) in the same snapshot


def with_retries(fn: Callable[[], T], what: str, attempts: int = 5, base_delay: float = 2.0) -> T:
    """Call ``fn``; on connection-level errors retry with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except psycopg.OperationalError as exc:
            if attempt == attempts:
                log.error("%s failed after %d attempts: %s", what, attempts, exc)
                raise
            delay = base_delay * 2 ** (attempt - 1)
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", what, attempt, attempts, exc, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def connect(settings: Settings, snapshot: bool = False) -> psycopg.Connection:
    """
    Open a PostgreSQL connection with retries.

    Args:
        snapshot: read-only REPEATABLE READ, so every query in the transaction
            sees the same data (used by Bronze extraction).
    """
    conn = with_retries(lambda: psycopg.connect(settings.postgres_conninfo), "postgres connect")
    if snapshot:
        conn.isolation_level = IsolationLevel.REPEATABLE_READ
        conn.read_only = True
    return conn


def export_table(conn: psycopg.Connection, schema: str, table: str) -> TableExport:
    """Export ``schema.table`` as CSV text plus its row count, in the connection's current snapshot."""
    target = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(target))
        source_row_count = cur.fetchone()[0]
        buffer = io.BytesIO()
        query = sql.SQL("COPY (SELECT * FROM {}) TO STDOUT WITH (FORMAT csv, HEADER true)").format(target)
        with cur.copy(query) as copy:
            for chunk in copy:
                buffer.write(chunk)
    csv_bytes = buffer.getvalue()
    frame = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
    return TableExport(frame=frame, csv_bytes=csv_bytes, source_row_count=source_row_count)
