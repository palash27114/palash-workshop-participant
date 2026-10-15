"""
Runtime configuration for the talent ETL pipeline.

Every setting comes from an environment variable so the same code runs
unchanged on a laptop (LocalStack + local PostgreSQL) and in the cloud
(Amazon S3 + Aurora PostgreSQL on EKS). Only ``IS_LOCAL`` changes behaviour.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date


def _env(name: str, default: str = "") -> str:
    """Return a stripped environment variable, or ``default`` when unset/blank."""
    value = os.getenv(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    """Immutable pipeline settings resolved from the environment."""

    is_local: bool
    run_date: date
    aws_region: str
    s3_endpoint_url: str | None
    lake_bucket: str
    postgres_host: str
    postgres_port: int
    postgres_name: str
    postgres_user: str
    postgres_pass: str
    source_schema: str
    source_tables: tuple[str, ...] = field(
        default=(
            "teams",
            "competencies",
            "employees",
            "projects",
            "performance_reviews",
            "competency_assessments",
            "talent_review",
        )
    )

    # Prefixes inside the lake bucket. Kept here so every layer agrees on layout.
    landing_prefix: str = "landing/learning_activity/"
    bronze_prefix: str = "bronze/"
    silver_prefix: str = "silver/"
    gold_prefix: str = "gold/"
    quarantine_prefix: str = "quarantine/"
    lineage_prefix: str = "_lineage/"
    state_prefix: str = "_state/"

    @property
    def postgres_conninfo(self) -> str:
        """libpq connection string; Aurora requires TLS, local PostgreSQL does not."""
        parts = [
            f"host={self.postgres_host}",
            f"port={self.postgres_port}",
            f"dbname={self.postgres_name}",
            f"user={self.postgres_user}",
            f"password={self.postgres_pass}",
            "connect_timeout=15",
        ]
        if not self.is_local:
            parts.append("sslmode=require")
        return " ".join(parts)

    def redacted(self) -> dict[str, object]:
        """Settings safe to log (no password)."""
        return {
            "is_local": self.is_local,
            "run_date": self.run_date.isoformat(),
            "aws_region": self.aws_region,
            "s3_endpoint_url": self.s3_endpoint_url,
            "lake_bucket": self.lake_bucket,
            "postgres": f"{self.postgres_user}@{self.postgres_host}:{self.postgres_port}/{self.postgres_name}",
            "source_schema": self.source_schema,
        }


def load_settings() -> Settings:
    """Build :class:`Settings` from environment variables with local-friendly defaults."""
    is_local = _env("IS_LOCAL", "true").lower() == "true"
    app_id = _env("APP_ID", _env("PARTICIPANT_ID", "abcd1234"))
    run_date_raw = _env("RUN_DATE")
    run_date = date.fromisoformat(run_date_raw) if run_date_raw else date.today()

    endpoint = _env("AWS_ENDPOINT_URL_S3") or _env("AWS_ENDPOINT_URL")
    if is_local and not endpoint:
        endpoint = "http://s3.localhost.localstack.cloud:4566"

    return Settings(
        is_local=is_local,
        run_date=run_date,
        # LocalStack resources are created in us-east-1 by the workshop scripts (bin/start-dev.sh).
        aws_region="us-east-1" if is_local else _env("AWS_REGION", "us-east-1"),
        s3_endpoint_url=endpoint or None,
        lake_bucket=_env("LAKE_BUCKET", f"coding-workshop-datalake-{app_id}"),
        postgres_host=_env("POSTGRES_HOST", "localhost"),
        postgres_port=int(_env("POSTGRES_PORT", "5432")),
        postgres_name=_env("POSTGRES_NAME", "postgres"),
        postgres_user=_env("POSTGRES_USER", "postgres"),
        postgres_pass=_env("POSTGRES_PASS", "postgres123" if is_local else ""),
        source_schema=_env("SOURCE_DB_SCHEMA", "hr"),
    )
