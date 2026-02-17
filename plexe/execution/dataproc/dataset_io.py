"""
Dataset I/O with format detection and normalization.

Provides format-agnostic dataset reading for plexe.
All format-specific logic is isolated here.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


# ============================================
# Format Enum
# ============================================


class DatasetFormat(str, Enum):
    """Supported dataset formats."""

    PARQUET = "parquet"
    CSV = "csv"
    ORC = "orc"
    AVRO = "avro"


# ============================================
# 1. Format Detection (Pure Logic, No I/O)
# ============================================


class FormatDetector:
    """
    Detects dataset format from URI.

    Single Responsibility: Format detection only (no reading/writing).
    """

    @staticmethod
    def detect(uri: str) -> DatasetFormat:
        """
        Detect format from URI.

        Strategy:
        1. Try extension-based detection (fast)
        2. If directory, inspect contents (requires I/O)

        Args:
            uri: Dataset URI (local path or S3)

        Returns:
            Detected format

        Raises:
            ValueError: If format cannot be determined
        """
        # Extension-based detection
        if FormatDetector._has_extension(uri):
            return FormatDetector._from_extension(uri)

        # Directory detection (requires listing)
        if uri.startswith("s3://"):
            return FormatDetector._from_s3_directory(uri)
        elif Path(uri).is_dir():
            return FormatDetector._from_local_directory(uri)

        raise ValueError(f"Could not detect format: {uri}")

    @staticmethod
    def _has_extension(uri: str) -> bool:
        """Check if URI has file extension."""
        name = Path(uri.rstrip("/")).name
        return "." in name and not uri.endswith("/")

    @staticmethod
    def _from_extension(uri: str) -> DatasetFormat:
        """Detect format from file extension."""
        uri_lower = uri.lower()
        if uri_lower.endswith(".parquet"):
            return DatasetFormat.PARQUET
        elif uri_lower.endswith((".csv", ".tsv")):
            return DatasetFormat.CSV
        elif uri_lower.endswith(".orc"):
            return DatasetFormat.ORC
        elif uri_lower.endswith(".avro"):
            return DatasetFormat.AVRO
        raise ValueError(f"Unknown extension: {uri}")

    @staticmethod
    def _from_s3_directory(uri: str) -> DatasetFormat:
        """Detect format by listing S3 directory."""
        import boto3

        logger.info(f"Inspecting S3 directory for format detection: {uri}")

        # Parse S3 URI
        if not uri.startswith("s3://"):
            raise ValueError(f"Not an S3 URI: {uri}")

        path = uri[5:]  # Remove 's3://'
        parts = path.split("/", 1)

        if len(parts) < 1 or not parts[0]:
            raise ValueError(f"Invalid S3 URI (missing bucket): {uri}")

        bucket = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 and parts[1] else ""

        # List first 10 objects
        s3 = boto3.client("s3")
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=10)

        if "Contents" not in response:
            raise ValueError(f"Empty S3 directory: {uri}")

        # Find first data file
        for obj in response["Contents"]:
            filename = Path(obj["Key"]).name
            if not filename.startswith(("_", ".")):
                try:
                    detected = FormatDetector._from_extension(filename)
                    logger.info(f"Detected format from S3 file: {filename} → {detected}")
                    return detected
                except ValueError:
                    continue  # Try next file

        raise ValueError(f"No recognizable data files in S3 directory: {uri}")

    @staticmethod
    def _from_local_directory(path: str) -> DatasetFormat:
        """Detect format by listing local directory."""
        logger.info(f"Inspecting local directory for format detection: {path}")

        for file_path in Path(path).iterdir():
            if not file_path.name.startswith(("_", ".")):
                try:
                    detected = FormatDetector._from_extension(file_path.name)
                    logger.info(f"Detected format from local file: {file_path.name} → {detected}")
                    return detected
                except ValueError:
                    continue  # Try next file

        raise ValueError(f"No recognizable data files in directory: {path}")


# ============================================
# 2. Dataset Reading (Thin Spark Wrapper)
# ============================================


class DatasetReader:
    """
    Reads datasets in any supported format using Spark.

    Single Responsibility: Format-specific Spark read operations.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def read(self, uri: str, format: DatasetFormat, options: dict | None = None) -> DataFrame:
        """
        Read dataset in specified format.

        Args:
            uri: Dataset URI (file or directory)
            format: Dataset format
            options: Format-specific options (e.g., CSV delimiter)

        Returns:
            Spark DataFrame
        """
        options = options or {}

        logger.info(f"Reading {format} dataset: {uri}")

        if format == DatasetFormat.PARQUET:
            return self.spark.read.parquet(uri)

        elif format == DatasetFormat.CSV:
            # Sensible CSV defaults
            csv_opts = {
                "header": True,
                "inferSchema": True,
                "mode": "PERMISSIVE",
            }
            csv_opts.update(options)
            logger.debug(f"CSV options: {csv_opts}")
            return self.spark.read.options(**csv_opts).csv(uri)

        elif format == DatasetFormat.ORC:
            return self.spark.read.orc(uri)

        elif format == DatasetFormat.AVRO:
            # Requires spark-avro package (included in Spark 3.x+)
            return self.spark.read.format("avro").load(uri)

        else:
            raise ValueError(f"Unsupported format: {format}")


# ============================================
# 3. Dataset Normalization (Orchestrator)
# ============================================


class DatasetNormalizer:
    """
    Normalizes datasets to Parquet format.

    Single Responsibility: Orchestrate detection → read → write pipeline.
    """

    def __init__(self, spark: SparkSession):
        self.detector = FormatDetector()
        self.reader = DatasetReader(spark)
        self.spark = spark

    def normalize(
        self,
        input_uri: str,
        output_uri: str,
        format_hint: DatasetFormat | None = None,
        read_options: dict | None = None,
    ) -> tuple[str, DatasetFormat]:
        """
        Normalize dataset to Parquet format.

        This is the main entry point for integrations. Handles:
        1. Format detection (if not provided)
        2. Skip normalization if already Parquet
        3. Otherwise: read → convert → write

        Args:
            input_uri: Source dataset (any format)
            output_uri: Target Parquet path (only used if conversion needed)
            format_hint: Skip detection if format known
            read_options: Format-specific read options

        Returns:
            (parquet_uri, detected_format): URI to Parquet dataset + original format
        """
        # Detect format (unless provided)
        if format_hint:
            detected_format = format_hint
            logger.info(f"Using provided format: {detected_format}")
        else:
            detected_format = self.detector.detect(input_uri)
            logger.info(f"Detected format: {detected_format}")

        # Skip normalization if already Parquet
        if detected_format == DatasetFormat.PARQUET:
            logger.info("Already Parquet - no normalization needed")
            return input_uri, detected_format

        logger.info(f"Normalizing {detected_format} → Parquet: {input_uri} → {output_uri}")

        # Read source format
        df = self.reader.read(input_uri, detected_format, read_options)

        # Validate non-empty (cheap check - doesn't scan entire dataset)
        if df.limit(1).count() == 0:
            raise ValueError(f"Dataset is empty: {input_uri}")

        logger.info(f"Read dataset with {len(df.columns)} columns")

        # Write as Parquet
        df.write.mode("overwrite").parquet(output_uri)

        logger.info(f"✓ Normalized to Parquet: {output_uri}")
        return output_uri, detected_format
