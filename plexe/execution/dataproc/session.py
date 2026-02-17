"""
Spark session management with singleton pattern.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

_spark_session = None


def get_or_create_spark_session(config=None) -> SparkSession:
    """
    Get or create Spark session based on config.

    Args:
        config: Config object (if None, uses local mode)

    Returns:
        SparkSession singleton instance

    Raises:
        RuntimeError: If Java/Spark/Databricks setup fails
        ValueError: If spark_mode is unknown
    """
    global _spark_session

    if _spark_session is not None:
        logger.debug("Reusing existing Spark session")
        return _spark_session

    # Import Config here to avoid circular dependency
    if config is None:
        from plexe.config import Config

        config = Config()

    # Dispatch based on spark_mode
    if config.spark_mode == "local":
        _spark_session = _create_local_spark(config)
    elif config.spark_mode == "databricks":
        _spark_session = _create_databricks_spark(config)
    else:
        raise ValueError(f"Unknown spark_mode: {config.spark_mode}")

    return _spark_session


def _create_local_spark(config) -> SparkSession:
    """
    Create local Spark session.

    Configuration optimized for single-node execution:
    - Configurable worker threads (default: local[8])
    - Configurable driver memory (default: 8g)
    - Arrow optimization for pandas interop
    - Adaptive query execution
    - S3 support via hadoop-aws 3.3.6 (compatible with Hadoop 3.3 in PySpark 4.0.1)

    Requires Java 17+ (see Dockerfile).

    Args:
        config: Config object with spark_local_cores and spark_driver_memory settings
    """
    import os

    from pyspark.sql import SparkSession

    master_url = f"local[{config.spark_local_cores}]"
    driver_memory = config.spark_driver_memory

    logger.info(f"Creating local Spark session ({master_url}, {driver_memory} driver memory, S3-enabled)")

    # Check if JARs are pre-bundled (Docker image via SPARK_JARS env var)
    spark_jars_env = os.environ.get("SPARK_JARS")

    try:
        builder = (
            SparkSession.builder.appName("plexe")
            .master(master_url)
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            # S3 support configuration
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")  # Map s3:// to S3A
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            # Fix Hadoop 3.3.6 config parsing bug (expects numeric values, not duration strings)
            # PySpark 4.0 sets some defaults with time suffixes that Hadoop 3.3.6 can't parse
            .config("spark.hadoop.fs.s3a.connection.timeout", "60000")  # ms
            .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")  # ms
            .config("spark.hadoop.fs.s3a.connection.idle.time", "60000")  # ms
            .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")  # seconds
            .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")  # seconds (24h)
        )

        if spark_jars_env:
            # Use pre-bundled JARs from Docker image (no Maven download)
            logger.info("Using pre-bundled Spark JARs from /opt/spark-jars/")
            builder = builder.config("spark.jars", spark_jars_env)
        else:
            # Fallback: Download JARs at runtime via Maven (local development)
            logger.info("Downloading Spark JARs from Maven Central (first run may take ~40s)")
            builder = builder.config(
                "spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.367",
            )

        spark = builder.getOrCreate()

        # Suppress verbose Spark logging
        spark.sparkContext.setLogLevel("WARN")

        logger.info(f"Spark session created: {spark.sparkContext.appName}")
        return spark

    except Exception as e:
        error_msg = str(e)

        # Helpful error for Java version issues
        if "JAVA_HOME" in error_msg or "java" in error_msg.lower() or "UnsupportedClassVersionError" in error_msg:
            raise RuntimeError(
                "Java 17+ is required for PySpark 3.5+.\n\n"
                "To install on macOS:\n"
                "  1. brew install openjdk@17\n"
                "  2. export JAVA_HOME=$(/usr/libexec/java_home -v 17)\n"
                "  3. Add to ~/.zshrc: export JAVA_HOME=$(/usr/libexec/java_home -v 17)\n"
                "\nThen restart your terminal and try again."
            ) from e

        raise


def _create_databricks_spark(config) -> SparkSession:
    """
    Create Databricks Connect session.

    Connects to remote Databricks cluster or serverless compute.
    Uses configuration from config object or environment variables.

    Session Isolation (Serverless Mode):
        Each DatabricksSession.builder.serverless().getOrCreate() call creates
        an isolated Spark session with its own context, temp tables, and metadata.
        Multiple concurrent sessions share underlying compute resources (Databricks-managed)
        but are logically isolated via Lakeguard secure containers. This means:
        - Multiple plexe instances can safely run in parallel
        - Each gets its own isolated execution environment
        - No cross-session interference or data leakage

    Args:
        config: Config object with Databricks settings

    Returns:
        SparkSession connected to Databricks

    Raises:
        RuntimeError: If databricks-connect not installed or connection fails
        ValueError: If configuration is invalid (e.g., missing cluster_id for non-serverless)
    """
    logger.info("Creating Databricks Connect session...")

    # Validate databricks-connect is installed
    try:
        from databricks.connect import DatabricksSession
    except ImportError:
        raise RuntimeError(
            "databricks-connect is not installed.\n\n"
            "To use Databricks Connect, install with: pip install plexe[databricks]\n"
        )

    try:
        builder = DatabricksSession.builder

        # Priority 1: Explicit host/token configuration (overrides everything)
        if config.databricks_host and config.databricks_token:
            logger.info(f"Using explicit credentials: {config.databricks_host}")

            if config.databricks_use_serverless:
                logger.info("Mode: Serverless compute")
                builder = builder.remote(host=config.databricks_host, token=config.databricks_token, serverless=True)
            else:
                # Persistent cluster mode - cluster_id is required
                if not config.databricks_cluster_id:
                    raise ValueError(
                        "databricks_cluster_id is required when databricks_use_serverless=false.\n"
                        "Set DATABRICKS_CLUSTER_ID environment variable or pass cluster_id in config."
                    )
                logger.info(f"Mode: Persistent cluster (cluster_id={config.databricks_cluster_id})")
                builder = builder.remote(
                    host=config.databricks_host, token=config.databricks_token, cluster_id=config.databricks_cluster_id
                )

        # Priority 2: Profile-based configuration
        elif config.databricks_profile:
            logger.info(f"Using Databricks config profile: {config.databricks_profile}")
            builder = builder.profile(config.databricks_profile)

            if config.databricks_use_serverless:
                logger.info("Mode: Serverless compute")
                builder = builder.serverless(True)
            else:
                logger.info("Mode: Persistent cluster (cluster_id from profile or env var)")

        # Priority 3: Environment variables only (auto-discovery)
        else:
            logger.info("Using environment variable auto-discovery")
            logger.info("Expected: DATABRICKS_HOST, DATABRICKS_TOKEN")

            if config.databricks_use_serverless:
                logger.info("Mode: Serverless compute")
                builder = builder.serverless(True)
            else:
                logger.info("Mode: Persistent cluster (DATABRICKS_CLUSTER_ID env var required)")
                # cluster_id will be auto-discovered from DATABRICKS_CLUSTER_ID env var
                # If not set, DatabricksSession will raise a clear error

        spark = builder.getOrCreate()

        # NOTE: Databricks Connect uses Spark Connect protocol which doesn't expose sparkContext
        # (client-server architecture with no local JVM). Skip JVM-dependent operations.
        # Logging configuration is handled server-side by Databricks.

        logger.info("✓ Databricks session created successfully")
        logger.info(f"✓ Spark version: {spark.version}")

        return spark

    except ValueError:
        # Re-raise validation errors without wrapping
        raise

    except Exception as e:
        error_msg = str(e)

        # Build helpful troubleshooting message based on mode
        if config.databricks_use_serverless:
            troubleshooting = (
                "Serverless Mode Requirements:\n"
                "  - DATABRICKS_HOST (e.g., https://your-workspace.cloud.databricks.com)\n"
                "  - DATABRICKS_TOKEN (Personal Access Token)\n"
                "  - Serverless compute must be enabled on your workspace\n"
            )
        else:
            troubleshooting = (
                "Persistent Cluster Mode Requirements:\n"
                "  - DATABRICKS_HOST (e.g., https://your-workspace.cloud.databricks.com)\n"
                "  - DATABRICKS_TOKEN (Personal Access Token)\n"
                "  - DATABRICKS_CLUSTER_ID (e.g., 1234-567890-abc123)\n"
                "  - Cluster must be running or set to auto-start\n"
            )

        troubleshooting += (
            "\nAlternatively, use a config profile:\n"
            "  - Set DATABRICKS_CONFIG_PROFILE=<profile-name>\n"
            "  - Ensure ~/.databrickscfg exists with the profile\n"
            "\nFor authentication issues, verify:\n"
            "  - Token is valid (not expired)\n"
            "  - Network connectivity to Databricks workspace\n"
        )

        raise RuntimeError(f"Failed to create Databricks session: {error_msg}\n\n{troubleshooting}") from e


def stop_spark_session():
    """Stop and cleanup Spark session."""
    global _spark_session

    if _spark_session is not None:
        logger.info("Stopping Spark session")
        _spark_session.stop()
        _spark_session = None
    else:
        logger.debug("No Spark session to stop")
