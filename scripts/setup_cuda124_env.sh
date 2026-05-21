#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

echo "[env] python: ${PYTHON_BIN}"
echo "[env] venv: ${VENV_DIR}"
echo "[env] installing PyTorch 2.5.1 cu124 first, then project deps"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements-torch-cu124.txt
python -m pip install -r requirements.txt
python -m pip check

python - <<'PY'
import os
import sys

import torch

expected_torch = "2.5.1"
actual_torch = torch.__version__.split("+", 1)[0]
if actual_torch != expected_torch:
    raise SystemExit(f"torch version mismatch: expected {expected_torch}, got {torch.__version__}")

if torch.version.cuda != "12.4":
    raise SystemExit(f"torch CUDA runtime mismatch: expected 12.4, got {torch.version.cuda}")

require_cuda = os.environ.get("REQUIRE_CUDA", "1") not in {"0", "false", "False", "no", "NO"}
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False; CUDA 12.4 wheel installed but GPU is unavailable")

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f"GPU {idx}: {torch.cuda.get_device_name(idx)}")
print("python:", sys.version.replace("\n", " "))
PY

echo "[env] CUDA 12.4 PyTorch environment is ready."
