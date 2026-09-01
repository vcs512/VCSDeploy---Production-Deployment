# VCSDeploy---Production-Deployment

Deployment techniques for computer vision.

## Services

All services are defined in `docker-compose.yml`.

| Service              | Target               | Backend        | GPU |
| -------------------- | -------------------- | -------------- | --- |
| `training`           | `training`           | PyTorch        | yes |
| `export`             | `export`             | PyTorch+TRT+OV | yes |
| `inference-pytorch`  | `inference-pytorch`  | PyTorch        | yes |
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

### inference-pytorch (evaluation)
- **Inputs:** trained Transformers model in `experiments/`, data in `data/`,
  config in `configs/evaluation.json` (validated by `EvaluationConfig`).
  Expected fields:

  | Field | Type | Default | Description |
  | --- | --- | --- | --- |
  | `model_path` | str | `experiments/best` | Directory of the trained Transformers model. |
  | `dataset_name` | str | `AI-Lab-Makerere/beans` | HuggingFace `datasets` dataset identifier. |
  | `dataset_cache_dir` | str | `data` | Directory where the dataset is cached (persisted). |
  | `label_column` | str | `labels` | Name of the dataset label column. |
  | `split` | str | `test` | Dataset split used for evaluation. |
  | `image_size` | int or null | `null` | Square input resolution; when `null` read from the model processor. |
  | `batch_size` | int | `8` | Per-device batch size. |
  | `num_workers` | int | `0` | DataLoader worker processes. |
  | `device` | str | `auto` | Inference device: `auto`, `cuda` or `cpu`. |
  | `sample_freq_hz` | float | `2.0` | Sampling frequency of the resource monitor. |
  | `output_dir` | str | `experiments` | Directory where the CSV report is written. |
  | `seed` | int | `42` | Random seed for reproducibility. |

- Usage: `docker compose run --rm inference-pytorch`
- Behavior: loads the trained model, evaluates it on the configured split, and
  collects CPU, RAM, VRAM and GPU usage by sampling on a background thread.
  Reports classification metrics (accuracy, precision, recall, f1) plus peak
  and 50/90/95/99 percentiles for each resource and for per-batch inference
  latency (measured after the first, warmup batch) and the elapsed time.
- Outputs: classification metrics, a resource summary and a timestamped CSV
  report (`evaluation_<backend>_<timestamp>.csv` in `output_dir`) printed to
  stdout.
- Libraries: torch, torchvision, transformers, datasets, psutil, nvidia-ml-py.

### export
- **Inputs:** trained model and export config in `configs/export.json`. Expected
  fields:

  | Field | Type | Default | Description |
  | --- | --- | --- | --- |
  | `model_path` | str | `experiments/best` | Directory of the trained Transformers model. |
  | `output_subdir` | str | `export` | Subdirectory under `model_path` for converted artifacts. |
  | `image_size` | int or null | `null` | Square input resolution; when `null` read from the model config. |
  | `batch_size` | int | `1` | Fixed batch dimension used by the TensorRT engine. |
  | `num_channels` | int | `3` | Number of input image channels. |
  | `targets` | list[str] | `["onnx", "tensorrt", "openvino"]` | Backends to convert to. |
  | `onnx_opset` | int | `17` | ONNX operator-set version used during export. |
  | `tensorrt_precision` | str | `fp16` | TensorRT engine precision (`fp16` or `fp32`). |
  | `tensorrt_workspace_mb` | int | `2048` | TensorRT builder workspace memory in MiB. |
  | `openvino_precision` | str | `fp16` | OpenVINO IR precision (`fp16` or `fp32`). |
  | `run_validation` | bool | `True` | Compare converted outputs against the PyTorch model. |
  | `seed` | int | `42` | Random seed for validation inputs. |
- Usage: `docker compose run --rm export`
- Behavior: loads the trained model, exports it to ONNX (dynamic batch), then
  builds a fixed-batch TensorRT engine and an OpenVINO IR from the ONNX graph.
  TensorRT export requires a CUDA device.
- Outputs: `experiments/best/export/onnx/model.onnx`,
  `experiments/best/export/tensorrt/model.engine`,
  `experiments/best/export/openvino/model.xml` + `model.bin`. When
  `run_validation` is enabled each backend is compared against the PyTorch
  reference (argmax match + max absolute logit difference).
- Libraries: torch, torchvision, onnx, onnxruntime-gpu, tensorrt-cu13,
  openvino, optimum, ml-dtypes.

### inference-onnx (evaluation)
- **Inputs:** exported ONNX graph and its source model in `experiments/`, data
  in `data/`, config in `configs/onnx_evaluation.json`.
  Expected fields:

  | Field | Type | Default | Description |
  | --- | --- | --- | --- |
  | `onnx_path` | str | `experiments/best/export/onnx/model.onnx` | Path to the exported ONNX graph. |
  | `model_path` | str | `experiments/best` | Source Transformers model (used for the image processor). |
  | `dataset_name` | str | `AI-Lab-Makerere/beans` | HuggingFace `datasets` dataset identifier. |
  | `dataset_cache_dir` | str | `data` | Directory where the dataset is cached (persisted). |
  | `label_column` | str | `labels` | Name of the dataset label column. |
  | `split` | str | `test` | Dataset split used for evaluation. |
  | `image_size` | int or null | `256` | Square input resolution fed to the ONNX graph. |
  | `batch_size` | int | `8` | Per-device batch size. |
  | `num_workers` | int | `0` | DataLoader worker processes. |
  | `device` | str | `auto` | Device used for preprocessing. |
  | `provider` | str | `cuda` | Preferred ONNX Runtime provider (`cuda` or `cpu`); falls back to CPU when unavailable. |
  | `sample_freq_hz` | float | `2.0` | Sampling frequency of the resource monitor. |
  | `output_dir` | str | `experiments` | Directory where the CSV report is written. |
  | `seed` | int | `42` | Random seed for reproducibility. |

- Usage: `docker compose run --rm inference-onnx`
- Behavior: runs the exported ONNX graph through ONNX Runtime, evaluating it on
  the configured split. It produces classification metrics (accuracy, precision,
  recall, f1) and peak/50/90/95/99 percentiles for CPU, RAM, VRAM, GPU and
  per-batch inference latency (excluding the warmup batch).
- Outputs: classification metrics, a resource summary and a timestamped CSV
  report (`evaluation_<backend>_<timestamp>.csv` in `output_dir`).
- Libraries: torch, torchvision, datasets, onnx, onnxruntime-gpu, psutil,
  nvidia-ml-py.

### inference-tensorrt (evaluation + inference)
- Inputs: TensorRT engine in `experiments/`, data in `data/`, config in `configs/`.
- Usage: `docker compose run --rm inference-tensorrt python src/inference/tensorrt.py`
- Outputs: predictions / metrics to `checkpoints/` or stdout.
- Libraries: torch, torchvision, tensorrt-cu13, nvidia-cudnn-cu13.

### inference-openvino (evaluation + inference)
- Inputs: OpenVINO IR in `experiments/`, data in `data/`, config in `configs/`.
- Usage: `docker compose run --rm inference-openvino python src/inference/openvino.py`
- Outputs: predictions / metrics to `checkpoints/` or stdout.
- Libraries: torch, torchvision, onnx, openvino (CPU execution).

## Roadmap

- [x] Training of a base model image classifier (transformers, pytorch)
- [x] Conversion in ONNX / TensorRT / OpenVINO (export service)
- [x] Evaluation of the base model (pytorch)
- [x] Evaluation in ONNX (onnxruntime-gpu)
- [ ] Evaluation in TensorRT
- [ ] Evaluation in OpenVINO
