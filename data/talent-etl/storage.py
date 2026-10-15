"""
Thin S3 data-lake client.

Wraps boto3 so the rest of the pipeline speaks in "keys" and DataFrames.
Locally the client points at LocalStack; in the cloud it uses real S3.
Transient errors are retried by botocore's adaptive retry mode.
"""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from config import Settings

log = logging.getLogger(__name__)


class Lake:
    """Read/write helpers for one S3 bucket acting as the data lake."""

    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.lake_bucket
        self._is_local = settings.is_local
        self._region = settings.aws_region
        kwargs: dict[str, object] = {
            "region_name": settings.aws_region,
            "config": Config(retries={"max_attempts": 5, "mode": "adaptive"}, max_pool_connections=32),
        }
        if settings.s3_endpoint_url:
            kwargs["endpoint_url"] = settings.s3_endpoint_url
        if settings.is_local:
            # LocalStack accepts any credentials; avoid picking up real AWS ones.
            kwargs["aws_access_key_id"] = "test"
            kwargs["aws_secret_access_key"] = "test"
        self.s3 = boto3.client("s3", **kwargs)

    # ------------------------------------------------------------------ bucket
    def ensure_bucket(self) -> None:
        """Create the lake bucket locally if missing. In the cloud Terraform owns it."""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            if not self._is_local:
                raise
            log.info("Creating local lake bucket s3://%s", self.bucket)
            extra = {} if self._region == "us-east-1" else {
                "CreateBucketConfiguration": {"LocationConstraint": self._region}}
            self.s3.create_bucket(Bucket=self.bucket, **extra)

    # ------------------------------------------------------------------ listing
    def list_objects(self, prefix: str) -> list[dict]:
        """Return every object (Key, ETag, Size) under ``prefix``."""
        objects: list[dict] = []
        for page in self.s3.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            objects.extend(page.get("Contents", []))
        return objects

    def list_keys(self, prefix: str, suffix: str = "") -> list[str]:
        """Return sorted keys under ``prefix`` ending with ``suffix``."""
        return sorted(o["Key"] for o in self.list_objects(prefix) if o["Key"].endswith(suffix))

    def list_partitions(self, prefix: str) -> list[str]:
        """Return immediate child "directories" under ``prefix`` (e.g. ``ingest_date=...``)."""
        result = self.s3.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/"
        )
        return sorted(cp["Prefix"] for page in result for cp in page.get("CommonPrefixes", []))

    def delete_prefix(self, prefix: str) -> int:
        """Delete everything under ``prefix``; used to make partition rewrites idempotent."""
        keys = [o["Key"] for o in self.list_objects(prefix)]
        for i in range(0, len(keys), 1000):
            self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in keys[i : i + 1000]], "Quiet": True},
            )
        return len(keys)

    # ------------------------------------------------------------------ bytes/json
    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to ``key`` and return its ``s3://`` URI."""
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return self.uri(key)

    def copy_object(self, source_key: str, target_key: str) -> str:
        """Server-side copy inside the bucket; the bytes never leave S3, so the copy is exact."""
        self.s3.copy_object(Bucket=self.bucket, Key=target_key, CopySource={"Bucket": self.bucket, "Key": source_key})
        return self.uri(target_key)

    def head(self, key: str) -> dict:
        """Object metadata (ContentLength, ETag, ...)."""
        return self.s3.head_object(Bucket=self.bucket, Key=key)

    def exists(self, key: str) -> bool:
        """True if ``key`` exists."""
        try:
            self.head(key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_bytes(self, key: str) -> bytes:
        """Download ``key`` as bytes."""
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def get_many_bytes(self, keys: Iterable[str], workers: int = 16) -> dict[str, bytes]:
        """Download many small objects concurrently."""
        keys = list(keys)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(zip(keys, pool.map(self.get_bytes, keys)))

    def put_json(self, key: str, payload: object) -> None:
        """Write ``payload`` as pretty JSON."""
        body = json.dumps(payload, indent=2, default=str).encode()
        self.put_bytes(key, body, "application/json")

    def get_json(self, key: str, default: object = None) -> object:
        """Read JSON from ``key``; return ``default`` if it does not exist."""
        try:
            return json.loads(self.get_bytes(key))
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return default
            raise

    # ------------------------------------------------------------------ parquet
    def write_parquet(self, key: str, df: pd.DataFrame) -> str:
        """Write one DataFrame as a single Parquet object and return its ``s3://`` URI."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        self.put_bytes(key, buffer.getvalue())
        return self.uri(key)

    def read_parquet(self, prefix_or_key: str) -> pd.DataFrame:
        """Read one Parquet object, or concatenate every ``.parquet`` under a prefix."""
        keys = (
            [prefix_or_key]
            if prefix_or_key.endswith(".parquet")
            else self.list_keys(prefix_or_key, ".parquet")
        )
        if not keys:
            raise FileNotFoundError(f"No parquet data at {self.uri(prefix_or_key)}")
        blobs = self.get_many_bytes(keys)
        frames = [pd.read_parquet(io.BytesIO(blobs[k]), engine="pyarrow") for k in keys]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def replace_dataset(self, prefix: str, df: pd.DataFrame, partition_col: str | None = None) -> list[str]:
        """
        Atomically-enough replace a dataset: delete the prefix, then write it again.

        With ``partition_col`` the frame is split into Hive-style folders
        (``<col>=<value>/part-000.parquet``) so readers can prune by partition.
        """
        self.delete_prefix(prefix)
        if partition_col is None:
            return [self.write_parquet(f"{prefix}part-000.parquet", df)]
        uris = []
        for value, part in df.groupby(partition_col, dropna=False, sort=True):
            label = "__null__" if pd.isna(value) else value
            uris.append(self.write_parquet(f"{prefix}{partition_col}={label}/part-000.parquet", part))
        return uris

    def uri(self, key: str) -> str:
        """Human-readable location used in logs and lineage."""
        return f"s3://{self.bucket}/{key}"
