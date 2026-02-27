# Plexe - ML model building framework
# Multi-stage Dockerfile:
#   base      → shared dependencies (no Spark provider)
#   pyspark   → local PySpark execution (DEFAULT)
#   databricks→ remote Databricks Connect execution
#
# Usage:
#   docker build .                        # default: pyspark
#   docker build --target databricks .    # databricks-connect

# ============================================
# Stage: base (shared across all variants)
# ============================================
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

WORKDIR /code

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for smolagents)
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python tooling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Install large stable dependencies before poetry to maximize build cache reuse.
# INSTALL_PYTORCH controls whether CPU-only PyTorch is installed.
ARG INSTALL_PYTORCH="true"
RUN if [ "$INSTALL_PYTORCH" = "true" ]; then \
        pip install --no-cache-dir torch==2.7.1 \
            --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple; \
    fi

# Install main dependencies + optional framework extras (no Spark provider yet)
# XGBoost is core; extras cover optional frameworks.
ARG POETRY_EXTRAS="aws catboost"
COPY pyproject.toml poetry.lock /code/
RUN poetry install --only=main --no-root --extras "${POETRY_EXTRAS}"

# Application code
COPY plexe/ /code/plexe/

# Working directories
RUN mkdir -p /data /logs /workdir /models
VOLUME ["/data", "/logs", "/workdir", "/models"]

# Environment
ENV AWS_REGION="us-west-2"
ENV AWS_DEFAULT_REGION="us-west-2"
ENV PYTHONPATH="/code:${PYTHONPATH}"

# ============================================
# Stage: databricks
# ============================================
FROM base AS databricks

RUN pip install --no-cache-dir 'databricks-connect>=17.3.1,<18.0.0'

CMD ["/bin/bash"]

# ============================================
# Stage: pyspark (DEFAULT — last stage, built by `docker build .`)
# ============================================
FROM base AS pyspark

# Automatic platform variables provided by Docker BuildKit
ARG TARGETARCH

# Java 17+ required for PySpark 4.0+
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-${TARGETARCH}
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Install pyspark extra
RUN pip install --no-cache-dir 'pyspark>=4.0.1,<4.1.0'

# Download Spark JARs for S3 + Avro support (avoids ~40s Maven download at runtime)
RUN mkdir -p /opt/spark-jars && \
    curl -o /opt/spark-jars/hadoop-aws-3.3.6.jar \
        https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar && \
    curl -o /opt/spark-jars/aws-java-sdk-bundle-1.12.367.jar \
        https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.367/aws-java-sdk-bundle-1.12.367.jar && \
    curl -o /opt/spark-jars/wildfly-openssl-1.1.3.Final.jar \
        https://repo1.maven.org/maven2/org/wildfly/openssl/wildfly-openssl/1.1.3.Final/wildfly-openssl-1.1.3.Final.jar && \
    curl -o /opt/spark-jars/spark-avro_2.13-4.0.1.jar \
        https://repo1.maven.org/maven2/org/apache/spark/spark-avro_2.13/4.0.1/spark-avro_2.13-4.0.1.jar

# Spark configuration for local mode
ARG PYTHON_VERSION=3.12
ENV SPARK_HOME="/usr/local/lib/python${PYTHON_VERSION}/site-packages/pyspark"
ENV PYSPARK_PYTHON="python3"
ENV PYSPARK_DRIVER_PYTHON="python3"
ENV SPARK_JARS="/opt/spark-jars/hadoop-aws-3.3.6.jar,/opt/spark-jars/aws-java-sdk-bundle-1.12.367.jar,/opt/spark-jars/wildfly-openssl-1.1.3.Final.jar,/opt/spark-jars/spark-avro_2.13-4.0.1.jar"

CMD ["/bin/bash"]
