# BLB Stage-2 RL · CUDA-enabled training image
# ----------------------------------------------------------------------
# Build:
#   docker build -t blb-rl:latest .
#
# Run (mount repo + GLUE cache + persistent dir as volumes):
#   docker run --gpus all -it --rm \
#     -v "$PWD":/workspace \
#     -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
#     -v "$PWD/Parting Chapter":/workspace/Parting\ Chapter \
#     -e GLUE_LOCAL_DATASET_DIR=/data/glue \
#     blb-rl:latest \
#     bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
#
# CUDA / PyTorch base. The migration target uses CUDA 12.4 and the official
# PyTorch 2.5.1 cu124 wheel set. Override only when intentionally testing a
# different server runtime.
ARG CUDA_TAG=12.4.1
FROM nvidia/cuda:${CUDA_TAG}-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# System deps: python3.11, git (submodules), build tools (for bitsandbytes
# /scipy if wheels are missing), curl, ca-certificates (HuggingFace HTTPS).
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

# Install PyTorch FIRST from the official CUDA index. This is the wheel
# that must match the host CUDA driver; pip will resolve transitive
# requirements (numpy, sympy, etc.) against this torch.
ARG TORCH_CUDA_CHANNEL=cu124
RUN pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Now the rest of the deps (torch is already satisfied).
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Freeze the actually-resolved versions into a lockfile that lives inside
# the image — easy to diff if you rebuild later.
RUN pip freeze --exclude-editable > /opt/requirements-frozen.txt \
    && echo "Resolved environment written to /opt/requirements-frozen.txt"

# Repository code is mounted at runtime via -v $PWD:/workspace.
# Don't COPY it into the image — that defeats the live-edit workflow.

# Default command: print env summary so users know what they got.
CMD bash -c "\
    echo '--- BLB Stage-2 RL container ---' && \
    nvidia-smi -L && \
    python -c 'import torch, transformers, numpy; \
print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available()); \
print(\"transformers:\", transformers.__version__); \
print(\"numpy:\", numpy.__version__)' && \
    echo '--- Frozen requirements at /opt/requirements-frozen.txt ---' && \
    echo 'Run training with: bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh' \
    "
