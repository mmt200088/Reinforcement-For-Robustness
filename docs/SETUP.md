# 环境搭建（Environment setup）

本项目依赖 PyTorch + HuggingFace + 一些科学计算栈。下面给两种安装方式：
**虚拟环境（推荐做开发 / 远端服务器训练）**和 **Docker**（推荐做可
重现的实验）。

---

## 1. 虚拟环境（venv / conda）

### 1.1 前置要求

- Python 3.9–3.12（项目用 3.10/3.11 验证过）
- Linux + NVIDIA GPU（>= 16 GB 显存推荐）；Mac (Apple Silicon) 上 RL
  本身可以跑但 BERT 训练慢、且 `bitsandbytes` 装不上
- CUDA driver 与 PyTorch wheel 匹配。若服务器已经有可用 CUDA PyTorch
  `2.x`，不要主动降级；直接装其余依赖即可。若需要 CUDA 12.4 备用环境，
  使用 `requirements-torch-cu124.txt` 安装 PyTorch 2.5.1 的 `cu124` wheel。

### 1.2 安装步骤

```bash
# 常规服务器：保留已有可用 CUDA torch，只安装其余依赖
pip install -r requirements.txt

# CUDA 12.4 备用环境：自动创建 .venv，安装 PyTorch 2.5.1 cu124，并做版本检查
bash scripts/setup_cuda124_env.sh

# 如果需要手动安装，必须先从官方 cu124 index 安装 PyTorch wheel set：
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch-cu124.txt
pip install -r requirements.txt

# 把当前环境冻结成 lockfile（便于回滚 / 复现）
pip freeze --exclude-editable > requirements-frozen.txt

# 验证安装
python -c "import torch, transformers, numpy; \
print('torch:', torch.__version__, 'cuda_runtime:', torch.version.cuda, 'cuda:', torch.cuda.is_available()); \
print('transformers:', transformers.__version__)"
```

> **macOS 用户**：`bitsandbytes` 行会被 pip 自动跳过；CPU/MPS 训练能跑但很慢，
> 只建议跑 unit tests / 离线 F0 scan。

### 1.3 GLUE 数据预下载（避免网络抖动）

GLUE 数据集通过 HuggingFace `datasets` 加载，但训练时网络如果不稳，
脚本会 fail。强烈推荐预下载到本地：

```bash
# 用 HF 拉一次缓存到当前用户目录
python -c "from datasets import load_dataset; load_dataset('glue', 'mrpc')"

# 或导出成本地 dir，训练时通过 GLUE_LOCAL_DATASET_DIR 指过去
mkdir -p data/glue
python -c "from datasets import load_dataset; \
load_dataset('glue', 'mrpc').save_to_disk('data/glue/mrpc')"
export GLUE_LOCAL_DATASET_DIR="$PWD/data/glue"
```

`rl_tune.py` 会按下面顺序找数据：
1. `GLUE_LOCAL_DATASET_DIR/<task>` 的 DatasetDict
2. `GLUE_DATASET_DIR` 环境变量
3. 本地 parquet
4. HuggingFace cache `local_files_only=True`

### 1.4 BERT 模型预下载

```bash
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-MRPC'); \
AutoTokenizer.from_pretrained('textattack/bert-base-uncased-MRPC')"
```

---

## 2. Docker（可重现实验）

### 2.1 构建镜像

```bash
# 默认用 CUDA 12.4 base + PyTorch 2.5.1 cu124 wheel
docker build -t blb-rl:latest .

# 想要 CUDA 11.8：
docker build --build-arg CUDA_TAG=11.8.0 --build-arg TORCH_CUDA_CHANNEL=cu118 \
    -t blb-rl:cu118 .
```

镜像构建结束会把 pip 解析出的精确版本写到 `/opt/requirements-frozen.txt`
内，方便回头对账。

### 2.2 运行训练（GPU pass-through）

```bash
docker run --gpus all -it --rm \
    -v "$PWD":/workspace \
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
    -v "$PWD/Parting Chapter":/workspace/Parting\ Chapter \
    -e GLUE_LOCAL_DATASET_DIR=/workspace/data/glue \
    blb-rl:latest \
    bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh
```

挂载 `Parting Chapter/` 目录是为了 **持久化 / 续训** 数据在容器外可见；
重启容器后还能 `--preset mrpc-blb-stage2-rl`（不带 `--fresh`）自动续。

### 2.3 快速 sanity check

```bash
docker run --gpus all -it --rm blb-rl:latest
# 默认 CMD 会打印 GPU 列表 + 主要包版本
```

---

## 3. 子模块（EzPC / LLM-Adapters / IST）

`.gitmodules` 声明了 3 个子模块。clone 时：

```bash
git clone --recurse-submodules git@github.com:mmt200088/Reinforcement-For-Robustness.git
# 已经 clone 过的话
git submodule update --init --recursive
```

`Rescale_optimizer/` **不是** submodule，是直接 commit 进来的（见
`CLAUDE.md` 提到的 commit `3af56dd`），不需要额外操作。

---

## 4. 常见安装陷阱

| 现象 | 原因 | 修法 |
|------|------|------|
| `torch.cuda.is_available()` 是 False | PyTorch wheel 不匹配 CUDA driver，或容器/venv 没看到 GPU | 新服务器默认跑 `bash scripts/setup_cuda124_env.sh`；确认 `torch.version.cuda == "12.4"` 且 `nvidia-smi` 正常 |
| `bitsandbytes` import 失败 | macOS 或 CUDA driver 太老 | 这是 optional dep，相关代码路径默认不开 |
| GLUE 下载卡住 | 网络抖动 | 预下载到本地 `GLUE_LOCAL_DATASET_DIR`（见 1.3） |
| 启动报 `static_skeletons_*.json not found` | Rescale_optimizer 的 baseline 缺 | 不要删 `Rescale_optimizer/configs/<dataset>/` 下的 JSON |
| 持久化目录路径含空格 | "Parting Chapter" 有空格 | 命令里加 `\ ` 转义或加双引号 |

---

## 5. 怎么报告环境信息（提 issue 时）

```bash
# 一行抓全
python -c "import torch, transformers, numpy, scipy, sklearn, datasets, sys; \
print(sys.version); \
print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'avail=', torch.cuda.is_available()); \
print('transformers=', transformers.__version__); \
print('datasets=', datasets.__version__); \
print('numpy=', numpy.__version__); \
print('scipy=', scipy.__version__); \
print('sklearn=', sklearn.__version__)"

# 或直接附 lockfile
pip freeze --exclude-editable > /tmp/env.txt
```
