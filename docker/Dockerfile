# syntax=docker/dockerfile:1

# ---- Base: CUDA 13.2 runtime + uv bootstrap ----
FROM nvidia/cuda:13.2.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        git \
        curl \
        wget \
        python3 \
        python3-venv \
        python3-dev \
        python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev \
    && uv run python -c "import numpy, pydantic, pandas, sklearn, matplotlib, fastapi, bentoml"

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# ---- Training ----
FROM base AS training
COPY . .
RUN uv sync --frozen --no-dev --group training
CMD ["bash"]

# ---- Export ----
FROM base AS export
COPY . .
RUN uv sync --frozen --no-dev --group export
CMD ["bash"]

# ---- Inference / evaluation: PyTorch (native Transformers) ----
FROM base AS inference-pytorch
COPY . .
RUN uv sync --frozen --no-dev --group inference-pytorch
CMD ["bash"]

# ---- Inference / evaluation: ONNX Runtime (CUDA) ----
FROM base AS inference-onnx
COPY . .
RUN uv sync --frozen --no-dev --group inference-onnx
CMD ["bash"]

# ---- Inference / evaluation: TensorRT ----
FROM base AS inference-tensorrt
COPY . .
RUN uv sync --frozen --no-dev --group inference-tensorrt
CMD ["bash"]

# ---- Inference / evaluation: OpenVINO (CPU) ----
FROM base AS inference-openvino
COPY . .
RUN uv sync --frozen --no-dev --group inference-openvino
CMD ["bash"]
