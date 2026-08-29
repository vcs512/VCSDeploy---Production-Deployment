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
- Inputs: dataset in `data/`, training config in `configs/`.
- Usage: `docker compose run --rm training python src/training/train.py`
- Outputs: model checkpoints written to `checkpoints/`, run tracked by MLflow.
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

- [ ] Training of a base model image classifier (transformers, pytorch)
- [ ] Evaluation of a base model (pytorch)
- [ ] Conversion and evaluation in ONNX
- [ ] Conversion and evaluation in TensorRT
- [ ] Conversion and evaluation in OpenVINO
