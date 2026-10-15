"""
Stand up the *source systems* with the workshop sample dataset.

This is not part of the pipeline; it simulates the systems the pipeline reads:

1. Uploads ``learning_activity/**.jsonl`` to ``s3://<lake>/landing/learning_activity/``
   (the LMS export drop zone) and the HR CSVs + ``schema.sql`` to ``landing/hr_seed/``.
2. Creates the HR tables in PostgreSQL schema ``hr`` and loads the CSVs.

Locally run it from your machine (``--dataset`` defaults to the sample zip
committed on the ``cwp/sample-data`` branch at ``sampleData/``; an extracted
folder works too)::

    python seed_sources.py
    python seed_sources.py --dataset /path/to/data-engineering-performance-talent-management

In the cloud Aurora is private to the VPC, so step 2 runs inside the EKS job
instead (``SEED_SOURCE_DB=true``) using the CSVs uploaded in step 1.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import zipfile
from pathlib import Path

from psycopg import sql

from config import Settings, load_settings
from sources import connect
from storage import Lake

log = logging.getLogger(__name__)

HR_SEED_PREFIX = "landing/hr_seed/"
# <repo>/data/talent-etl/seed_sources.py -> <repo>/sampleData/...zip (guarded: in the EKS pod the file lives at /app).
_HERE = Path(__file__).resolve()
DEFAULT_DATASET = (
    _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
) / "sampleData" / "data-engineering-sample-performance-scenario.zip"


def resolve_dataset(path: Path, workdir: Path) -> Path:
    """Return the dataset folder (the one holding ``schema.sql``), extracting a zip if given."""
    if path.is_file() and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(workdir)
        path = workdir
    matches = sorted(path.rglob("schema.sql"))
    if not matches:
        raise FileNotFoundError(f"No schema.sql found under {path}")
    return matches[0].parent


def upload_landing(dataset_dir: Path, settings: Settings, lake: Lake) -> None:
    """Copy the dataset's files into the landing zone of the lake bucket."""
    events = sorted((dataset_dir / "learning_activity").rglob("*.jsonl"))
    for path in events:
        key = settings.landing_prefix + path.relative_to(dataset_dir / "learning_activity").as_posix()
        lake.put_bytes(key, path.read_bytes(), "application/x-ndjson")
    hr_files = [dataset_dir / "schema.sql"] + [dataset_dir / f"{t}.csv" for t in settings.source_tables]
    for path in hr_files:
        lake.put_bytes(HR_SEED_PREFIX + path.name, path.read_bytes(), "text/plain")
    log.info("seed uploaded %d event files and %d HR files to %s", len(events), len(hr_files), lake.uri("landing/"))


def load_postgres(settings: Settings, files: dict[str, bytes]) -> None:
    """(Re)create the HR source tables in ``settings.source_schema`` and bulk-load them."""
    schema = sql.Identifier(settings.source_schema)
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema))
        cur.execute(sql.SQL("SET search_path TO {}").format(schema))
        for table in reversed(settings.source_tables):
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
        cur.execute(files["schema.sql"].decode())
        for table in settings.source_tables:
            copy_sql = sql.SQL("COPY {} FROM STDIN WITH (FORMAT csv, HEADER true)").format(sql.Identifier(table))
            with cur.copy(copy_sql) as copy:
                copy.write(files[f"{table}.csv"])
            log.info("seed loaded %s.%s rows=%d", settings.source_schema, table, cur.rowcount)


def source_tables_exist(settings: Settings) -> bool:
    """True if every HR source table already exists in ``settings.source_schema``."""
    with connect(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s AND table_name = ANY(%s)",
            (settings.source_schema, list(settings.source_tables)),
        )
        return cur.fetchone()[0] == len(settings.source_tables)


def load_postgres_from_landing(settings: Settings, lake: Lake) -> None:
    """Cloud path: seed PostgreSQL from the CSVs previously uploaded to S3."""
    names = ["schema.sql"] + [f"{t}.csv" for t in settings.source_tables]
    files = lake.get_many_bytes(HR_SEED_PREFIX + n for n in names)
    load_postgres(settings, {k.removeprefix(HR_SEED_PREFIX): v for k, v in files.items()})


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="sample dataset zip or extracted folder (default: %(default)s)")
    parser.add_argument("--skip-postgres", action="store_true", help="only upload to S3 (cloud: Aurora is private)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = load_settings()
    lake = Lake(settings)
    lake.ensure_bucket()
    with tempfile.TemporaryDirectory() as tmp:
        dataset = resolve_dataset(args.dataset, Path(tmp))
        upload_landing(dataset, settings, lake)
        if not args.skip_postgres:
            load_postgres(settings, {p.name: p.read_bytes() for p in dataset.iterdir() if p.is_file()})


if __name__ == "__main__":
    main()
