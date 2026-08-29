# VCSDeploy---Production-Deployment

Deployment techniques for computer vision.

## Services

All services are defined in `docker-compose.yml`.

| Service              | Target               | Backend        | GPU |
| -------------------- | -------------------- | -------------- | --- |
| `training`           | `training`           | PyTorch        | yes |
| `export`             | `export`             | PyTorch+TRT+OV | yes |
| `inference-onnx`     | `inference-onnx`     | ONNX Runtime   | yes |
| `inference-tensorrt` | `inference-tensorrt` | TensorRT       | yes |
| `inference-openvino` | `inference-openvino` | OpenVINO       | no  |

### training
- **Inputs:** the routine is configured entirely through `configs/training.json`
  (validated by `TrainingConfig`). Expected fields:

  | Field | Type | Default | Description |
  | --- | --- | --- | --- |
  | `model_local_path` | str | `checkpoints/swinv2-tiny-patch4-window16-256` | Path to the local pretrained model directory. |
  | `dataset_name` | str | `AI-Lab-Makerere/beans` | HuggingFace `datasets` dataset identifier. |
  | `dataset_cache_dir` | str | `data` | Directory where the dataset is cached (persisted). |
  | `label_column` | str | `labels` | Name of the dataset label column. |
  | `image_size` | int | `224` | Square input resolution fed to the model. |
  | `batch_size` | int | `8` | Per-device batch size. |
  | `num_epochs` | int | `5` | Number of training epochs. |
  | `learning_rate` | float | `2e-4` | Optimizer learning rate. |
  | `freeze_backbone` | bool | `True` | When true only the classification head is trained. |
  | `gradient_accumulation_steps` | int | `2` | Optimizer steps before a weight update. |
  | `mixed_precision` | bool | `True` | Enable AMP when a CUDA device is available. |
  | `mlflow_tracking_uri` | str | `sqlite:///mlflow.db` | MLflow backend store URI. |
  | `mlflow_experiment_name` | str | `swinv2-beans-clf` | MLflow experiment name. |
  | `experiments_dir` | str | `experiments` | Root directory for exported artifacts. |
  | `output_model_subdir` | str | `best` | Subdirectory (under `experiments_dir`) for the best model. |
  | `seed` | int | `42` | Random seed for reproducibility. |
  | `num_workers` | int | `2` | DataLoader worker processes. |
- Usage: `docker compose run --rm training python src/services/training.py`
- Behavior: fine-tunes the local SwinV2-tiny classifier. Backbone is frozen
  by default (`freeze_backbone` in the config) so only the head is trained;
  full fine-tuning is enabled by setting `freeze_backbone: false`. AMP and
  gradient accumulation keep VRAM usage low (4 GB target).
- Outputs: metrics (accuracy, precision, recall, f1) tracked in MLflow
  (`sqlite:///mlflow.db`), best model exported to `experiments/best/`
  (`pytorch_model.bin`, `config.json`, `preprocessor_config.json`).
- Libraries: torch, torchvision, transformers, accelerate, datasets, mlflow.

### export
- Inputs: PyTorch checkpoint in `checkpoints/`, export config in `configs/`.
- Usage: `docker compose run --rm export python src/export/export.py`
- Outputs: ONNX model, TensorRT engine, and OpenVINO IR under `checkpoints/`.
- Libraries: torch, torchvision, onnx, onnxruntime-gpu, tensorrt-cu13,
  openvino, optimum.

### inference-onnx (evaluation + inference)
- Inputs: ONNX model in `checkpoints/`, data in `data/`, config in `configs/`.
- Usage: `docker compose run --rm inference-onnx python src/inference/onnx.py`
- Outputs: predictions / metrics to `checkpoints/` or stdout.
- Libraries: torch, torchvision, onnx, onnxruntime-gpu, nvidia-cudnn-cu13.

### inference-tensorrt (evaluation + inference)
- Inputs: TensorRT engine in `checkpoints/`, data in `data/`, config in `configs/`.
- Usage: `docker compose run --rm inference-tensorrt python src/inference/tensorrt.py`
- Outputs: predictions / metrics to `checkpoints/` or stdout.
- Libraries: torch, torchvision, tensorrt-cu13, nvidia-cudnn-cu13.

### inference-openvino (evaluation + inference)
- Inputs: OpenVINO IR in `checkpoints/`, data in `data/`, config in `configs/`.
- Usage: `docker compose run --rm inference-openvino python src/inference/openvino.py`
- Outputs: predictions / metrics to `checkpoints/` or stdout.
- Libraries: torch, torchvision, onnx, openvino (CPU execution).

## Roadmap

- [x] Training of a base model image classifier (transformers, pytorch)
- [ ] Evaluation of a base model (pytorch)
- [ ] Conversion and evaluation in ONNX
- [ ] Conversion and evaluation in TensorRT
- [ ] Conversion and evaluation in OpenVINO
