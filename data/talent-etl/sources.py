"""
Source-system connectivity: PostgreSQL (HR system of record).

Reads use ``COPY ... TO STDOUT`` so Bronze receives the exact text the source
emitted, and connections are retried with exponential backoff because Aurora
Serverless may be resuming from zero capacity on first touch.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Callable, TypeVar

import pandas as pd
import psycopg
from psycopg import sql

from config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T")


def with_retries(fn: Callable[[], T], what: str, attempts: int = 5, base_delay: float = 1.0) -> T:
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


def connect(settings: Settings) -> psycopg.Connection:
    """Open a PostgreSQL connection with retries."""
    return with_retries(lambda: psycopg.connect(settings.postgres_conninfo), "postgres connect")


def read_table_raw(conn: psycopg.Connection, schema: str, table: str) -> pd.DataFrame:
    """Export ``schema.table`` as CSV text and load it with every column as string."""
    query = sql.SQL("COPY (SELECT * FROM {}.{}) TO STDOUT WITH (FORMAT csv, HEADER true)").format(
        sql.Identifier(schema), sql.Identifier(table)
    )
    buffer = io.BytesIO()
    with conn.cursor() as cur, cur.copy(query) as copy:
        for chunk in copy:
            buffer.write(chunk)
    buffer.seek(0)
    return pd.read_csv(buffer, dtype=str, keep_default_na=False)
