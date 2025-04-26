# GPU Support in Plexe

This document describes how GPU support works in Plexe for accelerating machine learning model training and inference.

## Overview

Plexe automatically detects and utilizes available GPUs for machine learning tasks, providing acceleration for:
- Model training
- Feature engineering
- Inference/prediction

The GPU integration works across multiple machine learning frameworks and is designed to be transparent to users.

## How It Works

### 1. GPU Detection

Plexe uses a multi-framework approach to detect available GPUs:

```python
# From plexe/internal/common/utils/model_utils.py
def is_gpu_available() -> bool:
    # Check for PyTorch GPU
    if is_package_available("torch"):
        import torch
        if torch.cuda.is_available():
            return True
    
    # Check for TensorFlow GPU
    if is_package_available("tensorflow"):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return True
    
    return False
```

### 2. Framework-Specific Configuration

For each supported ML framework, Plexe automatically configures the appropriate GPU parameters:

| Framework | GPU Configuration |
|-----------|------------------|
| XGBoost   | `tree_method: gpu_hist`, `predictor: gpu_predictor` |
| LightGBM  | `device: gpu` |
| CatBoost  | `task_type: GPU` |
| PyTorch   | `device: cuda` |
| TensorFlow| Memory growth enabled |

### 3. Ray Integration

Plexe uses Ray for distributed execution, with GPU support integrated in the `RayExecutor`:

1. GPU resources are explicitly requested for Ray tasks
2. GPU detection code is injected into training scripts
3. Memory usage is monitored throughout execution
4. Environment variables are set for framework-specific GPU usage

```python
@ray.remote(num_gpus=1)  # Request 1 GPU for this task
def _run_code(code: str, working_dir: str, dataset_files: List[str], timeout: int) -> dict:
    # GPU detection and code enhancement logic
    ...
```

### 4. Code Enhancement

When running code, the executor automatically enhances it with GPU detection:

```python
# GPU detection code injected into execution
gpu_detection_code = """
# GPU detection for ML frameworks
import torch
if torch.cuda.is_available():
    print(f"GPU available for training: {torch.cuda.get_device_name(0)}")
    device = "cuda"
    # Try to enable GPU for common ML frameworks
    try:
        import os
        # XGBoost
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # TensorFlow
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    except Exception as e:
        print(f"Error setting up GPU environment: {e}")
else:
    print("No GPU available for training")
    device = "cpu"
"""
```

## Usage

GPU support is transparent to users - no additional configuration is required:

```python
# GPU will be automatically used if available
model = plexe.Model(intent="Predict the target variable")
model.build(
    datasets=[data],
    provider="openai/gpt-4o-mini"
)
```

You can monitor GPU usage with the `GPUMonitorCallback`:

```python
class GPUMonitorCallback(Callback):
    def __init__(self):
        self.gpu_usage = []
        
    def on_build_start(self, build_state_info):
        if torch.cuda.is_available():
            memory = torch.cuda.memory_allocated() / (1024**2)
            logger.info(f"Build start - GPU memory: {memory:.2f} MB")
            self.gpu_usage.append(("build_start", memory))
```

## Performance Considerations

- GPU acceleration typically provides the most benefit for:
  - Deep learning models with PyTorch/TensorFlow
  - Gradient boosting frameworks (XGBoost, LightGBM, CatBoost)
  - Large datasets with complex feature engineering

- For small datasets or simple models, the CPU may be more efficient due to the overhead of GPU data transfer.

## Requirements

To use GPU acceleration in Plexe:

1. CUDA-compatible GPU hardware
2. Appropriate drivers (NVIDIA drivers)
3. CUDA toolkit and cuDNN installed
4. GPU-enabled versions of machine learning frameworks (PyTorch, TensorFlow, XGBoost, etc.)