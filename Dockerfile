# CUDA training image
# Build:
#   docker build -t blb-rl:latest .
#
# Run:
#   docker run --gpus all -it --rm \
#     -v "$PWD":/workspace \
#     -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
#     -v "$PWD/outputs":/workspace/outputs \
#     -e GLUE_LOCAL_DATASET_DIR=/data/glue \
#     blb-rl:latest \
#     bash run_search.sh run rl --preset bert-base-mrpc-stage2-rl --fresh
#
ARG CUDA_TAG=12.4.1
FROM nvidia/cuda:${CUDA_TAG}-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git build-essential pkg-config \
        curl ca-certificates \
        less jq \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip wheel setuptools

WORKDIR /workspace

ARG TORCH_CUDA_CHANNEL=cu124
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN pip freeze --exclude-editable > /opt/requirements-frozen.txt \
    && echo "Resolved environment written to /opt/requirements-frozen.txt"

CMD bash -c "\
    echo '--- BLB Stage-2 RL container ---' && \
    nvidia-smi -L && \
    python -c 'import torch, transformers, numpy; \
print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available()); \
print(\"transformers:\", transformers.__version__); \
print(\"numpy:\", numpy.__version__)' && \
    echo '--- Frozen requirements at /opt/requirements-frozen.txt ---' && \
    echo 'Run training with: bash run_search.sh run rl --preset bert-base-mrpc-stage2-rl --fresh' \
    "
