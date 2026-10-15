"""
ACME talent ETL - batch pipeline entry point.

Runs the medallion pipeline end to end::

    PostgreSQL (HR) ─┐
                     ├─► Bronze (raw) ─► Silver (validated) ─► Gold (business tables)
    S3 landing (LMS) ┘                       └─► quarantine/       └─► _lineage/ manifest

Usage::

    python job.py                         # all layers
    python job.py --layers silver,gold    # rebuild from existing Bronze
    python job.py --force-bronze          # reload today's Bronze (it is write-once otherwise)
    RUN_DATE=2026-10-15 python job.py     # pin the snapshot date (idempotent re-runs)

Environment variables are documented in config.py and README.md. Exit code is
non-zero when any step fails; the lineage manifest is written either way.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from bronze import ingest_events, ingest_postgres
from config import load_settings
from gold import build_gold
from lineage import RunContext
from seed_sources import load_postgres_from_landing
from silver import build_silver
from storage import Lake

LAYERS = ("bronze", "silver", "gold")


def configure_logging() -> None:
    """Structured key=value logs to stdout (picked up by kubectl logs / CloudWatch)."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI options."""
    parser = argparse.ArgumentParser(description="ACME talent medallion ETL")
    parser.add_argument("--layers", default=",".join(LAYERS),
                        help="comma-separated subset of bronze,silver,gold (default: all)")
    parser.add_argument("--force-bronze", action="store_true",
                        help="reload today's Bronze even if already loaded (Bronze is write-once otherwise)")
    args = parser.parse_args(argv)
    args.layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    unknown = set(args.layers) - set(LAYERS)
    if unknown:
        parser.error(f"unknown layer(s): {sorted(unknown)}")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the requested layers; returns a process exit code."""
    configure_logging()
    log = logging.getLogger("job")
    args = parse_args(argv)
    settings = load_settings()
    lake = Lake(settings)
    lake.ensure_bucket()

    ctx = RunContext(lake, settings.run_date.isoformat(), settings.lineage_prefix)
    log.info("run run_id=%s status=started layers=%s settings=%s", ctx.run_id, args.layers, settings.redacted())

    try:
        if os.getenv("SEED_SOURCE_DB", "false").lower() == "true":
            with ctx.step("setup", "seed_source_postgres"):
                load_postgres_from_landing(settings, lake)
        if "bronze" in args.layers:
            ingest_postgres(settings, lake, ctx, force=args.force_bronze)
            ingest_events(settings, lake, ctx, force=args.force_bronze)
        if "silver" in args.layers:
            build_silver(settings, lake, ctx)
        if "gold" in args.layers:
            build_gold(settings, lake, ctx)
    except Exception:
        log.exception("run run_id=%s status=failed", ctx.run_id)
        ctx.save("failed")
        return 1

    ctx.save("succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
