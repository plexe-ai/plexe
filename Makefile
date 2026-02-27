# ============================================
# Plexe Makefile
# ============================================
# Quick reference for developers:
#   make help              Show all available commands
#   make test-quick        Fast test (~30s, 1 iteration)
#   make test-xgboost      Test XGBoost only
#   make test-catboost     Test CatBoost only
#   make test-all-models   Test all model types
#   make build             Build local development image

# Default Python version for local development
PYTHON_VERSION ?= 3.12

# Auto-detect config.yaml and set up mounting (optional)
# If config.yaml exists, mount it and set CONFIG_FILE env var
CONFIG_MOUNT := $(if $(wildcard config.yaml),-v $(PWD)/config.yaml:/code/config.yaml:ro,)
CONFIG_ENV := $(if $(wildcard config.yaml),-e CONFIG_FILE=/code/config.yaml,)

# ============================================
# Help
# ============================================
.PHONY: help
help:
	@echo "════════════════════════════════════════════════════════════════"
	@echo "Plexe Development Commands"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make build          Build local development image"
	@echo "  make test-quick     Fast sanity check (~30s, 1 iteration)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test-xgboost       Test XGBoost model type"
	@echo "  make test-catboost      Test CatBoost model type"
	@echo "  make test-lightgbm      Test LightGBM model type"
	@echo "  make test-pytorch       Test PyTorch model type"
	@echo "  make test-keras         Test Keras model type"
	@echo "  make test-all-models    Test all model types (sequential)"
	@echo "  make test-full          Full test run (3 iterations + evaluation)"
	@echo ""
	@echo "📊 Example Datasets:"
	@echo "  make run-titanic        Run on Titanic dataset (medium)"
	@echo "  make run-house-prices   Run on House Prices dataset (regression)"
	@echo ""
	@echo "🏗️  Building:"
	@echo "  make build              Build default image (PySpark)"
	@echo "  make build-databricks   Build Databricks variant"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean              Clean workdir + docker images"
	@echo "  make clean-workdir      Clean only workdir"
	@echo ""
	@echo "⚙️  Advanced:"
	@echo "  make build-multiarch    Build for both arm64 and amd64"
	@echo "  make test-amd64         Test amd64 compatibility"
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"

# ============================================
# Quick Development Tests
# ============================================

# Fast sanity check - 1 iteration, minimal config
.PHONY: test-quick
test-quick: build
	@echo "🚀 Running quick sanity test (1 iteration, ~30s)..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=2 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id quick_test \
			--max-iterations 1 \
			--work-dir /workdir/quick_test \
			--spark-mode local \
			--allowed-model-types xgboost
	@echo "✅ Quick test passed!"

# Test XGBoost specifically
.PHONY: test-xgboost
test-xgboost: build
	@echo "🧪 Testing XGBoost model type..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id test_xgboost \
			--max-iterations 2 \
			--work-dir /workdir/test_xgboost \
			--spark-mode local \
			--allowed-model-types xgboost
	@echo "✅ XGBoost test passed!"

# Test CatBoost specifically
.PHONY: test-catboost
test-catboost: build
	@echo "🧪 Testing CatBoost model type..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id test_catboost \
			--max-iterations 2 \
			--work-dir /workdir/test_catboost \
			--spark-mode local \
			--allowed-model-types catboost
	@echo "✅ CatBoost test passed!"

# Test LightGBM specifically
.PHONY: test-lightgbm
test-lightgbm: build
	@echo "🧪 Testing LightGBM model type..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id test_lightgbm \
			--max-iterations 2 \
			--work-dir /workdir/test_lightgbm \
			--spark-mode local \
			--allowed-model-types lightgbm \
			--enable-final-evaluation
	@echo "✅ LightGBM test passed!"

# Test PyTorch specifically
.PHONY: test-pytorch
test-pytorch: build
	@echo "🧪 Testing PyTorch model type..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/house-prices/train.csv \
			--user-id test_user \
			--intent "predict house sale price" \
			--experiment-id test_pytorch_house_prices \
			--max-iterations 2 \
			--work-dir /workdir/test_pytorch_house_prices \
			--spark-mode local \
			--allowed-model-types pytorch
	@echo "✅ PyTorch test passed!"

# Test Keras specifically (requires appropriate dataset)
.PHONY: test-keras
test-keras: build
	@echo "⚠️  Keras requires IMAGE_PATH or TEXT_STRING data layout"
	@echo "⚠️  Spaceship Titanic is FLAT_NUMERIC - incompatible with Keras"
	@echo "Skipping Keras test (add image/text dataset to enable)"

# Test all model types (let agents explore)
.PHONY: test-all-models
test-all-models: build
	@echo "🧪 Testing all model types (agents can choose between xgboost/catboost)..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id test_all_models \
			--max-iterations 3 \
			--work-dir /workdir/test_all_models \
			--spark-mode local
	@echo "✅ All models test passed!"

# Full test with evaluation
.PHONY: test-full
test-full: build
	@echo "🧪 Running full test (3 iterations + evaluation)..."
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id test_full \
			--max-iterations 3 \
			--work-dir /workdir/test_full \
			--spark-mode local \
			--enable-final-evaluation
	@echo "✅ Full test passed!"

# ============================================
# Example Datasets
# ============================================

# Spaceship Titanic dataset (medium)
.PHONY: run-titanic
run-titanic: build
	@echo "📊 Running on Spaceship Titanic dataset..."
	$(eval TIMESTAMP := $(shell date +%Y%m%d_%H%M%S))
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id dev_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id titanic \
			--max-iterations 5 \
			--work-dir /workdir/titanic/$(TIMESTAMP) \
			--spark-mode local \
			--enable-final-evaluation

# House Prices dataset (regression)
.PHONY: run-house-prices
run-house-prices: build
	@echo "📊 Running on House Prices dataset..."
	$(eval TIMESTAMP := $(shell date +%Y%m%d_%H%M%S))
	docker run --rm \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=4 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION) \
		python -m plexe.main \
			--train-dataset-uri /data/house-prices/train.csv \
			--user-id dev_user \
			--intent "predict house sale price" \
			--experiment-id house_prices \
			--max-iterations 5 \
			--work-dir /workdir/house_prices/$(TIMESTAMP) \
			--spark-mode local \
			--enable-final-evaluation

# ============================================
# Building
# ============================================

# Build default image (PySpark target)
.PHONY: build
build:
	@echo "🏗️  Building default image (Python $(PYTHON_VERSION), PySpark)..."
	docker build \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		-t plexe:py$(PYTHON_VERSION) \
		-f Dockerfile .
	@echo "✅ Build complete: plexe:py$(PYTHON_VERSION)"


# Build Databricks variant
.PHONY: build-databricks
build-databricks:
	@echo "🏗️  Building Databricks variant (Python $(PYTHON_VERSION))..."
	docker build \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		--target databricks \
		-t plexe:py$(PYTHON_VERSION)-databricks \
		-f Dockerfile .
	@echo "✅ Build complete: plexe:py$(PYTHON_VERSION)-databricks"

# Build for multiple architectures (production)
.PHONY: build-multiarch
build-multiarch:
	@echo "🏗️  Building multi-arch image (arm64 + amd64)..."
	docker buildx build --platform linux/arm64,linux/amd64 --provenance=false \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		-t plexe:py$(PYTHON_VERSION) \
		-f Dockerfile .
	@echo "✅ Multi-arch build complete!"

# ============================================
# Advanced Testing
# ============================================

# Test amd64 compatibility (via QEMU emulation)
.PHONY: test-amd64
test-amd64:
	@echo "🧪 Testing amd64 compatibility (via QEMU - will be slower)..."
	docker buildx build --platform linux/amd64 --load --provenance=false \
		--build-arg PYTHON_VERSION=$(PYTHON_VERSION) \
		-t plexe:py$(PYTHON_VERSION)-amd64 \
		-f Dockerfile .
	docker run --rm --platform linux/amd64 \
		--add-host=host.docker.internal:host-gateway \
		$(CONFIG_MOUNT) \
		$(CONFIG_ENV) \
		-v $(PWD)/examples/datasets:/data:ro \
		-v $(PWD)/workdir:/workdir \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e ANTHROPIC_API_KEY=$(ANTHROPIC_API_KEY) \
		-e SPARK_LOCAL_CORES=2 \
		-e SPARK_DRIVER_MEMORY=4g \
		plexe:py$(PYTHON_VERSION)-amd64 \
		python -m plexe.main \
			--train-dataset-uri /data/spaceship-titanic/train.parquet \
			--user-id test_user \
			--intent "predict whether a passenger was transported" \
			--experiment-id test_amd64 \
			--max-iterations 1 \
			--work-dir /workdir/test_amd64 \
			--spark-mode local \
			--allowed-model-types xgboost
	@echo "✅ amd64 test passed!"

# ============================================
# Cleanup
# ============================================

.PHONY: clean-workdir
clean-workdir:
	@echo "🧹 Cleaning workdir..."
	rm -rf workdir/*
	@echo "✅ Workdir cleaned!"

.PHONY: clean-images
clean-images:
	@echo "🧹 Removing docker images..."
	-docker rmi plexe:py$(PYTHON_VERSION) 2>/dev/null
	-docker rmi plexe:py$(PYTHON_VERSION)-databricks 2>/dev/null
	@echo "✅ Images removed!"

.PHONY: clean
clean: clean-workdir clean-images
	@echo "✅ Full cleanup complete!"

# Include local overrides (not committed to git)
-include Makefile.local
