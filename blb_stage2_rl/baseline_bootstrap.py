"""BLB Stage-2 baseline JSON handover —— RL side.

本模块包含两条互补的 baseline 抽取路径：

**路径 1（推荐 / 当前实际使用）**：从 Rescale_optimizer 仓库提供的静态 baseline 归档
（``Rescale_optimizer/configs/<dataset>/static_skeletons_<dataset>.json``）直接读取。
RO 团队人工维护这份文件，每条 entry 对应一个 (block, [degree]) graph 的 baseline。
RL 一侧逐层挑出每层对应的 entry（block3/block5 按 stage-1 degree 选 graph_key），
组装成 RL 动作向量 + 校准过的 ``MaxSFsTable``。入口：``load_static_skeletons_baseline``。

**路径 2（旧 handover 协议，保留向后兼容）**：双向 JSON 文件握手。RL 写 Stage-1
配置请求，RO 写 baseline 响应。详见 ``docs/blb_baseline_handover_protocol.md`` v1：

  * ``write_baseline_request(...)``  → 把 Stage-1 配置序列化为请求 JSON
  * ``read_baseline_response(...)``  → 读取 Rescale_optimizer 写的响应 JSON
                                        并校验 schema / request_id 一致性
  * ``baseline_response_to_cost_stats(...)`` → 把响应转换成
                                                ``BaselineCostStats``，可直接
                                                喂给 ``BLBStage2Env``
  * ``handover_paths(repo_root, dataset)`` → 标准路径辅助

设计目标：
  1. **不引入新依赖**：纯标准库 + numpy。
  2. **failed-block tolerance**：响应里某条 (block, layer) 失败时，向上层抛
     可读异常（caller 可决定是否启动训练）。
  3. **schema 自检**：版本不匹配立即拒绝，而不是带病往下走。
  4. 不实现"通知机制"（人工 / 文件 watch / CI）—— 那是部署侧选择。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .reward import BaselineCostStats
from . import skeleton_stage_map as _ssm


# ---------------------------------------------------------------------------
# Schema 常量
# ---------------------------------------------------------------------------
REQUEST_SCHEMA_V1 = "blb_baseline_request_v1"
RESPONSE_SCHEMA_V1 = "blb_baseline_response_v1"

HANDOVER_DIRNAME = "handover"
REQUEST_FILENAME_FMT = "baseline_request_{dataset}.json"
RESPONSE_FILENAME_FMT = "baseline_response_{dataset}.json"

# Stage-1 取值范围（与协议第 2 节一致）
# degree 0 = ReLU（用 ReLU 替换 GELU）→ block5_n0 graph（无多项式 GELU 节点，只有
# LN tail + Wffn1）。2026-06-02 起 Stage-2 曾支持 degree 0；2026-06-06 起 **关闭**：
# Stage-1 已停止采样 degree 0（f85c77e），Stage-2 在此入口拒绝 degree-0 层。block5_n0
# 的解码 / RO 契约 / handler 仍保留为 dormant（历史配置 / 手工 eval / 一行回退即可恢复）。
ALLOWED_GELU_DEGREES = (1, 2, 4)
ALLOWED_SOFTMAX_DEGREES = (2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# 路径辅助
# ---------------------------------------------------------------------------
def handover_paths(repo_root: str, dataset: str) -> Dict[str, str]:
    """返回 ``{request_path, response_path, dir}``，路径都 join 过 repo_root。"""
    root = os.path.abspath(str(repo_root))
    handover_dir = os.path.join(root, HANDOVER_DIRNAME)
    return {
        "dir": handover_dir,
        "request_path": os.path.join(
            handover_dir, REQUEST_FILENAME_FMT.format(dataset=str(dataset)),
        ),
        "response_path": os.path.join(
            handover_dir, RESPONSE_FILENAME_FMT.format(dataset=str(dataset)),
        ),
    }


# ---------------------------------------------------------------------------
# Request 写入
# ---------------------------------------------------------------------------
def _validate_stage1(
        gelu_per_layer: Sequence[int],
        softmax_per_layer: Sequence[int],
        num_layers: int,
        ) -> None:
    if len(gelu_per_layer) != int(num_layers):
        raise ValueError(
            f"gelu_degree_per_layer length {len(gelu_per_layer)} != num_layers {num_layers}"
        )
    if len(softmax_per_layer) != int(num_layers):
        raise ValueError(
            f"softmax_degree_per_layer length {len(softmax_per_layer)} != num_layers {num_layers}"
        )
    bad_gelu = [int(d) for d in gelu_per_layer if int(d) not in ALLOWED_GELU_DEGREES]
    if bad_gelu:
        hint = (
            " Degree 0 (ReLU / block5_n0) is disabled for Stage-2 since 2026-06-06;"
            " the block5_n0 path is retained only as dormant decode for historical/manual eval."
            if 0 in bad_gelu else ""
        )
        raise ValueError(
            f"gelu_degree_per_layer contains values outside {ALLOWED_GELU_DEGREES}: {bad_gelu}.{hint}"
        )
    bad_sm = [int(d) for d in softmax_per_layer if int(d) not in ALLOWED_SOFTMAX_DEGREES]
    if bad_sm:
        raise ValueError(
            f"softmax_degree_per_layer contains values outside {ALLOWED_SOFTMAX_DEGREES}: {bad_sm}"
        )


def _generate_request_id(dataset: str, model: str) -> str:
    """``<YYYYMMDD-HHMMSS>-<dataset>-<model>``，便于多请求并存追踪。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{str(dataset)}-{str(model)}"


def _git_commit_short(repo_root: str) -> str:
    """``git rev-parse --short HEAD``；失败返回空字符串（不阻塞 baseline）。"""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            capture_output=True, text=True, timeout=5.0,
        )
        if out.returncode == 0:
            return str(out.stdout).strip()
    except Exception:
        pass
    return ""


def write_baseline_request(
        repo_root: str,
        dataset: str,
        stage1_gelu_per_layer: Sequence[int],
        stage1_softmax_per_layer: Sequence[int],
        *,
        num_layers: Optional[int] = None,
        model: str = "bert-base",
        request_id: Optional[str] = None,
        rl_max_sfs: Optional[Mapping[str, Mapping[str, int]]] = None,
        rl_max_fresh_sfs: Optional[Mapping[str, int]] = None,
        rl_max_truncation_k: int = 13,
        blb_first_input_N: int = 8192,
        ) -> str:
    """把 Stage-1 配置写成 baseline_request JSON，返回写入路径。

    Args:
        repo_root: 项目根目录
        dataset:   GLUE 任务名（决定 graph 文件 block1_<dataset>.json 等）
        stage1_gelu_per_layer:    长度 num_layers，元素 ∈ {1,2,4}
        stage1_softmax_per_layer: 长度 num_layers，元素 ∈ {2..6}
        num_layers:               缺省取两个序列长度的较小者
        model / request_id / 其余字段：见
            ``docs/blb_baseline_handover_protocol.md`` §2

    Raises:
        ValueError: Stage-1 配置长度 / 取值不合法
    """
    if num_layers is None:
        num_layers = min(len(stage1_gelu_per_layer), len(stage1_softmax_per_layer))
    _validate_stage1(stage1_gelu_per_layer, stage1_softmax_per_layer, int(num_layers))

    paths = handover_paths(repo_root, dataset)
    os.makedirs(paths["dir"], exist_ok=True)

    if request_id is None:
        request_id = _generate_request_id(dataset, model)

    payload: Dict[str, Any] = {
        "schema": REQUEST_SCHEMA_V1,
        "request_id": str(request_id),
        "dataset": str(dataset),
        "model": str(model),
        "num_layers": int(num_layers),
        "stage1_config": {
            "gelu_degree_per_layer":
                [int(d) for d in stage1_gelu_per_layer],
            "softmax_degree_per_layer":
                [int(d) for d in stage1_softmax_per_layer],
        },
        "rl_max_truncation_k": int(rl_max_truncation_k),
        "blb_first_input_N": int(blb_first_input_N),
        "rl_repo_commit": _git_commit_short(repo_root),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "blb_stage2_rl.baseline_bootstrap.write_baseline_request",
    }
    if rl_max_sfs:
        payload["rl_max_sfs"] = {
            str(blk): {str(node): int(sf) for node, sf in mapping.items()}
            for blk, mapping in rl_max_sfs.items()
        }
    if rl_max_fresh_sfs:
        payload["rl_max_fresh_sfs"] = {
            str(k): int(v) for k, v in rl_max_fresh_sfs.items()
        }

    tmp_path = paths["request_path"] + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, paths["request_path"])
    return paths["request_path"]


# ---------------------------------------------------------------------------
# Response 读取 / 校验
# ---------------------------------------------------------------------------
@dataclass
class BaselineHandoverResultEntry:
    """单条 (block, layer) 的解析结果。"""
    config_name: str
    graph_key: str
    block: int
    layer: int
    success: bool
    skeleton: List[int] = field(default_factory=list)
    t_baseline: List[int] = field(default_factory=list)
    q_bits_baseline: List[int] = field(default_factory=list)
    modulus_chain: Optional[Dict[str, Any]] = None
    fusion_count: int = 0
    invalid_chain: Optional[Dict[str, Any]] = None
    cut_point_sf: List[Dict[str, Any]] = field(default_factory=list)
    effective_rotations: List[str] = field(default_factory=list)
    error_message: str = ""


@dataclass
class BaselineHandoverResult:
    """``read_baseline_response`` 的解析输出。"""
    schema: str
    request_id: str
    dataset: str
    model: str
    num_layers: int
    ok: bool
    error: Optional[str]
    results: List[BaselineHandoverResultEntry]
    aggregate_total_bits_sum: int
    aggregate_total_fusion_count: int
    aggregate_valid_block_count: int
    aggregate_invalid_block_count: int
    warnings: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def by_config_name(self) -> Dict[str, BaselineHandoverResultEntry]:
        return {r.config_name: r for r in self.results}

    @property
    def by_block_layer(self) -> Dict[tuple, BaselineHandoverResultEntry]:
        return {(int(r.block), int(r.layer)): r for r in self.results}


class BaselineHandoverError(RuntimeError):
    """读响应失败 / 校验失败时抛出。"""


def _parse_entry(
        record: Mapping[str, Any],
        index: int,
        ) -> BaselineHandoverResultEntry:
    try:
        return BaselineHandoverResultEntry(
            config_name=str(record["config_name"]),
            graph_key=str(record["graph_key"]),
            block=int(record["block"]),
            layer=int(record["layer"]),
            success=bool(record.get("success", False)),
            skeleton=[int(x) for x in (record.get("skeleton") or [])],
            t_baseline=[int(x) for x in (record.get("t_baseline") or [])],
            q_bits_baseline=[int(x) for x in (record.get("q_bits_baseline") or [])],
            modulus_chain=(
                dict(record["modulus_chain"])
                if isinstance(record.get("modulus_chain"), Mapping) else None
            ),
            fusion_count=int(record.get("fusion_count", 0)),
            invalid_chain=(
                dict(record["invalid_chain"])
                if isinstance(record.get("invalid_chain"), Mapping) else None
            ),
            cut_point_sf=[dict(x) for x in (record.get("cut_point_sf") or [])],
            effective_rotations=[
                str(x) for x in (record.get("effective_rotations") or [])
            ],
            error_message=str(record.get("error_message", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BaselineHandoverError(
            f"results[{index}] 解析失败：{exc}（record={record!r}）"
        ) from exc


def read_baseline_response(
        repo_root: str,
        dataset: str,
        *,
        expected_request_id: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        ) -> BaselineHandoverResult:
    """读取响应 JSON 并 strict-校验。

    Args:
        repo_root:           项目根目录
        dataset:             GLUE 任务名
        expected_request_id: 校验响应的 request_id 必须匹配；None ⇒ 跳过该校验。
        max_age_seconds:     响应文件 mtime 必须在 ``now - max_age_seconds`` 之后；
                             None ⇒ 不限。用于过期响应保护。

    Raises:
        BaselineHandoverError: 文件不存在 / schema 不匹配 / 字段缺失 /
                                aggregate 计数与 results 不一致。
    """
    paths = handover_paths(repo_root, dataset)
    resp_path = paths["response_path"]
    if not os.path.isfile(resp_path):
        raise BaselineHandoverError(
            f"baseline 响应文件不存在: {resp_path}"
            f"（dataset={dataset!r}；先把 baseline_request 发给 RO 一侧）"
        )

    if max_age_seconds is not None:
        mtime = os.path.getmtime(resp_path)
        age = float(time.time() - mtime)
        if age > float(max_age_seconds):
            raise BaselineHandoverError(
                f"baseline 响应过期（age={age:.0f}s > {max_age_seconds:.0f}s）：{resp_path}"
            )

    with open(resp_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, Mapping):
        raise BaselineHandoverError(
            f"baseline 响应顶层不是 dict：{type(raw).__name__}"
        )

    schema = str(raw.get("schema", ""))
    if schema != RESPONSE_SCHEMA_V1:
        raise BaselineHandoverError(
            f"baseline 响应 schema 不匹配：期望 {RESPONSE_SCHEMA_V1}，实际 {schema!r}"
        )

    request_id = str(raw.get("request_id", ""))
    if expected_request_id is not None and request_id != str(expected_request_id):
        raise BaselineHandoverError(
            f"baseline 响应 request_id 不匹配：期望 {expected_request_id!r}，"
            f"实际 {request_id!r}（说明 RO 写的是上一轮的响应）"
        )

    if str(raw.get("dataset", "")) != str(dataset):
        raise BaselineHandoverError(
            f"baseline 响应 dataset 不匹配：期望 {dataset!r}，"
            f"实际 {raw.get('dataset')!r}"
        )

    num_layers = int(raw.get("num_layers", 0))
    if num_layers <= 0:
        raise BaselineHandoverError(
            f"baseline 响应 num_layers 不合法：{num_layers!r}"
        )

    results_raw = raw.get("results")
    if not isinstance(results_raw, list):
        raise BaselineHandoverError("baseline 响应 results 字段不是 list")

    results: List[BaselineHandoverResultEntry] = []
    for idx, rec in enumerate(results_raw):
        if not isinstance(rec, Mapping):
            raise BaselineHandoverError(
                f"baseline 响应 results[{idx}] 不是 dict：{type(rec).__name__}"
            )
        results.append(_parse_entry(rec, idx))

    # 语义更新（2026-05）：(block=1, layer=0) 不再发给 RO（layer-0 block1
    # 噪声整体不安装；第一个 HE 配置视为无损）。期望 results 长度 = 5L - 1。
    # 若 RO 仍然返回 (1, 0) 这条记录，不算错误——会被接受但忽略。
    expected_n = 5 * num_layers - 1
    if len(results) not in (expected_n, expected_n + 1):
        raise BaselineHandoverError(
            f"baseline 响应 results 长度 {len(results)} != 期望 "
            f"{expected_n}（5*num_layers - 1，已排除 layer-0 block1）"
        )

    seen = set()
    for r in results:
        key = (int(r.block), int(r.layer))
        if key in seen:
            raise BaselineHandoverError(
                f"baseline 响应 results 中 (block={r.block}, layer={r.layer}) 重复"
            )
        seen.add(key)
    for layer in range(num_layers):
        for block in (1, 2, 3, 4, 5):
            # layer-0 block1 是允许缺失的（语义对齐）；其余必须有。
            if int(layer) == 0 and int(block) == 1:
                continue
            if (block, layer) not in seen:
                raise BaselineHandoverError(
                    f"baseline 响应 results 缺少 (block={block}, layer={layer})"
                )

    aggregate = raw.get("aggregate") or {}
    if not isinstance(aggregate, Mapping):
        raise BaselineHandoverError("baseline 响应 aggregate 不是 dict")
    valid_n = int(aggregate.get("valid_block_count", -1))
    invalid_n = int(aggregate.get("invalid_block_count", -1))
    if valid_n < 0 or invalid_n < 0:
        raise BaselineHandoverError(
            f"baseline 响应 aggregate 缺少 valid/invalid_block_count: {dict(aggregate)!r}"
        )
    # 允许 RO 在 aggregate 里把可选的 (block=1, layer=0) 也计进去，即多 1。
    if (valid_n + invalid_n) not in (expected_n, expected_n + 1):
        raise BaselineHandoverError(
            f"baseline 响应 aggregate {valid_n}+{invalid_n} != {expected_n}"
            f"（也不等于 {expected_n + 1} 的 tolerant-mode 值）"
        )
    actual_valid = sum(1 for r in results if r.success)
    actual_invalid = sum(1 for r in results if not r.success)
    if actual_valid != valid_n or actual_invalid != invalid_n:
        raise BaselineHandoverError(
            f"baseline 响应 aggregate ({valid_n}/{invalid_n}) 与 results "
            f"实际计数 ({actual_valid}/{actual_invalid}) 不一致"
        )

    total_bits_sum = int(aggregate.get("total_bits_sum", 0))
    total_fusion_count = int(aggregate.get("total_fusion_count", 0))

    actual_total_bits = 0
    for r in results:
        if r.success and isinstance(r.modulus_chain, Mapping):
            actual_total_bits += int(r.modulus_chain.get("total_bits", 0))
    if actual_total_bits != total_bits_sum:
        # 不抛异常 —— 只警告，因为 aggregate 是 RO 一侧自报的辅助字段
        # 但不一致是 RO 端 bug 信号，应该被人看到
        warnings_list = list(raw.get("warnings") or [])
        warnings_list.append(
            f"aggregate.total_bits_sum={total_bits_sum} 与 results 实算 "
            f"{actual_total_bits} 不一致；以 results 为准"
        )
    else:
        warnings_list = list(raw.get("warnings") or [])

    return BaselineHandoverResult(
        schema=schema,
        request_id=request_id,
        dataset=str(raw.get("dataset", dataset)),
        model=str(raw.get("model", "")),
        num_layers=num_layers,
        ok=bool(raw.get("ok", False)),
        error=(str(raw["error"]) if raw.get("error") is not None else None),
        results=results,
        aggregate_total_bits_sum=int(actual_total_bits),    # 用实算
        aggregate_total_fusion_count=int(total_fusion_count),
        aggregate_valid_block_count=actual_valid,
        aggregate_invalid_block_count=actual_invalid,
        warnings=[str(x) for x in warnings_list],
        raw=dict(raw),
    )


# ---------------------------------------------------------------------------
# Response → BaselineCostStats
# ---------------------------------------------------------------------------
def baseline_response_to_cost_stats(
        result: BaselineHandoverResult,
        *,
        baseline_avg_k: float = 13.0,
        ) -> BaselineCostStats:
    """把响应转换成 ``BaselineCostStats``，用于 ``BLBStage2Env`` reward 校准。

    转换规则：
      * ``total_bits_sum``    = result.aggregate_total_bits_sum（已重算）
      * ``total_fusion_count``= result.aggregate_total_fusion_count
      * ``avg_k``             = baseline_avg_k（K 不在 RO 计算范围；用 RL 的 max_K）
      * ``loss_*`` / ``metric*_*``：留 0；上层会用 probe 重算

    若 ``result.ok=False``，仍然返回 stats（以 valid 子集计算），但 caller 应该
    检查 ``result.ok`` 决定是否启动训练。
    """
    return BaselineCostStats(
        total_bits_sum=int(result.aggregate_total_bits_sum),
        total_fusion_count=int(result.aggregate_total_fusion_count),
        avg_k=float(baseline_avg_k),
        # 占位；上层 estimate_baseline_cost_stats / runner 会实际跑一次填充
        loss_mean=0.0,
        loss_std=0.0,
        metric1_mean=0.0,
        metric2_mean=0.0,
        metric1_std=0.0,
        metric2_std=0.0,
        typical_bits_drop=1.0,
        typical_fusion_count=1.0,
        typical_k_drop=1.0,
    )


# ---------------------------------------------------------------------------
# 自检 helpers（可被 tests 调用）
# ---------------------------------------------------------------------------
def validate_response_against_request(
        request_path: str,
        response: BaselineHandoverResult,
        ) -> List[str]:
    """对照 request 文件检查 response 一致性，返回问题列表（空 = 全部通过）。"""
    problems: List[str] = []
    try:
        with open(request_path, "r", encoding="utf-8") as f:
            req = json.load(f)
    except Exception as exc:
        return [f"无法读取 request: {exc}"]

    if str(req.get("schema", "")) != REQUEST_SCHEMA_V1:
        problems.append(f"request.schema 不是 {REQUEST_SCHEMA_V1}")
    if str(req.get("request_id", "")) != str(response.request_id):
        problems.append("request_id 不匹配 (request vs response)")
    if int(req.get("num_layers", 0)) != int(response.num_layers):
        problems.append("num_layers 不匹配")
    if str(req.get("dataset", "")) != str(response.dataset):
        problems.append("dataset 不匹配")

    s1 = req.get("stage1_config") or {}
    gelu = list(s1.get("gelu_degree_per_layer") or [])
    softmax = list(s1.get("softmax_degree_per_layer") or [])
    for r in response.results:
        if not r.success:
            continue
        layer = int(r.layer)
        block = int(r.block)
        if block == 3:
            if 0 <= layer < len(softmax):
                expected = f"block3_exp_n{int(softmax[layer])}"
                if r.graph_key != expected:
                    problems.append(
                        f"results (block=3, layer={layer}).graph_key={r.graph_key!r} "
                        f"不符 stage1.softmax[{layer}]={softmax[layer]} → 期望 {expected!r}"
                    )
        elif block == 5:
            if 0 <= layer < len(gelu):
                expected = f"block5_n{int(gelu[layer])}"
                if r.graph_key != expected:
                    problems.append(
                        f"results (block=5, layer={layer}).graph_key={r.graph_key!r} "
                        f"不符 stage1.gelu[{layer}]={gelu[layer]} → 期望 {expected!r}"
                    )
    return problems


# ===========================================================================
# 路径 1：直接读 static_skeletons_<dataset>.json 抽 baseline
# ===========================================================================
#
# 这是当前实际使用的 baseline 获取方式：RO 团队人工生成 baseline 归档，
# RL 这边直接读取，逐层根据 stage-1 选择的 (gelu_deg, softmax_deg) 拼出
# 每层 5 个 (block, layer) 的 baseline entry。Layer-0 block1 跳过
# （语义对齐：第一个 HE 配置无损）。
# ---------------------------------------------------------------------------


def static_skeletons_archive_path(rescale_optimizer_root: str, dataset: str) -> str:
    """返回 ``<root>/configs/<dataset>/static_skeletons_<dataset>.json`` 绝对路径。"""
    return os.path.join(
        os.path.abspath(str(rescale_optimizer_root)),
        "configs",
        str(dataset),
        f"static_skeletons_{dataset}.json",
    )


def load_static_skeletons_archive(path: str) -> Dict[str, Dict[str, Any]]:
    """读 ``static_skeletons_<dataset>.json``，返回 ``{config_name: entry}`` 索引。

    只保留 ``success=True`` 的 entry。
    """
    with open(str(path), "r", encoding="utf-8") as f:
        archive = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for entry in archive.get("results", []) or []:
        if not entry.get("success"):
            continue
        cname = str(entry.get("config_name") or "").strip()
        if cname:
            out[cname] = dict(entry)
    return out


def static_skeletons_graph_key(
        block_idx: int,
        dataset: str,
        gelu_degree: int,
        softmax_degree: int,
        ) -> str:
    """对照 ``action_space.make_config_name`` 的命名规则给 (block, dataset, deg) → graph_key。"""
    block_idx = int(block_idx)
    if block_idx == 1:
        return f"block1_{dataset}"
    if block_idx == 2:
        return f"block2_{dataset}"
    if block_idx == 3:
        return f"block3_exp_n{int(softmax_degree)}"
    if block_idx == 4:
        return "block4"
    if block_idx == 5:
        return f"block5_n{int(gelu_degree)}"
    raise ValueError(f"invalid block_idx {block_idx}")


# ---------------------------------------------------------------------------
# RO 节点名 ↔ RL 字段名映射
# ---------------------------------------------------------------------------
# The SOURCE / encode / rescale node→RL-field tables that used to live here are
# now the single source of truth in ``skeleton_stage_map`` (the complete-chain
# SSOT). Extraction above reads them via ``_ssm.source_rl_field`` /
# ``_ssm.encode_rl_fields`` / ``_ssm.rescale_rl_fields`` so a skeleton regen
# auto-propagates. ``_RO_X2_AUX_FRESH_FIELD`` below is the one piece the SSOT
# does not model (the "x2" aux fresh inferred from SOURCE.sf), so it stays.


# 2026-05-21 user spec：``"x2"`` CTCT_MUL 旁节点的语义是"两个 fresh 操作数 SF
# 相等才能让结果 SF 是 2 倍"。block 2/5 各有一个这样的"辅助 fresh × SOURCE"
# 旁节点，其 RL 字段没有自己的 SOURCE，baseline 应当从公式推出：
# ``aux_fresh.sf = SOURCE.sf`` （只在 SOURCE 已经被 cut_point_sf[0] 抽取出来
# 之后才推算）。下表只列出该模式 —— 真正的 squaring 节点 (ctct_square_*,
# ctct_gelu_x*) 是同一个 ciphertext 自乘，不需要辅助 fresh。
_RO_X2_AUX_FRESH_FIELD: Dict[int, Dict[str, str]] = {
    2: {"ctct_x_mean_over_std": "x_centered_fresh_sf"},
    5: {"ctct_xmean_over_std":  "inv_std_fresh_sf"},
}


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------
@dataclass
class StaticSkeletonsLayerBlock:
    """单个 (block, layer) 的 RO baseline 抽取结果。"""
    block_idx: int
    layer_idx: int
    graph_key: str
    # field_name (RL 字段) → baseline SF
    field_baseline_sfs: Dict[str, int] = field(default_factory=dict)
    # field_name → "fresh" / "encode" / "rescale"（来自 JSON 的哪个区块）
    field_kind_in_ro: Dict[str, str] = field(default_factory=dict)
    # 模数链 cost
    total_bits: int = 0
    fusion_count: int = 0
    drop_order: List[int] = field(default_factory=list)
    # effective_rotations 原样保留（RL 端不直接用，但写到 cfg 反映给训练）
    effective_rotations: List[Dict[str, Any]] = field(default_factory=list)
    # 调试用：JSON 里无法映射回 RL 字段的节点
    unmapped_propagation_nodes: List[str] = field(default_factory=list)
    unmapped_rescale_nodes: List[str] = field(default_factory=list)


@dataclass
class StaticSkeletonsBaseline:
    """整个模型按 Stage-1 配置抽出的 BLB baseline。"""
    dataset: str
    num_layers: int
    gelu_per_layer: List[int]
    softmax_per_layer: List[int]
    archive_path: str
    per_block_layer: Dict[Tuple[int, int], StaticSkeletonsLayerBlock] = field(default_factory=dict)
    aggregate_total_bits: int = 0
    aggregate_fusion_count: int = 0
    aggregate_valid_block_count: int = 0
    aggregate_invalid_block_count: int = 0
    # 缺失 graph_key（archive 里没 success entry）的 (block, layer) 列表
    missing_block_layer: List[Tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 主抽取函数
# ---------------------------------------------------------------------------
def _extract_one_block_layer(
        entry: Mapping[str, Any],
        block_idx: int,
        layer_idx: int,
        graph_key: str,
        *,
        gelu_degree: int,
        ) -> StaticSkeletonsLayerBlock:
    """从一条 archive entry 抽出该 (block, layer) 的 RL 字段 baseline。"""
    out = StaticSkeletonsLayerBlock(
        block_idx=int(block_idx),
        layer_idx=int(layer_idx),
        graph_key=str(graph_key),
    )

    # --- cut_point_sf：第一项是 SOURCE (fresh)，其余带 sf_post 的是 rescale ---
    cps = entry.get("cut_point_sf") or []
    if not isinstance(cps, list) or not cps:
        return out

    # SOURCE → fresh
    source_entry = cps[0] if isinstance(cps[0], Mapping) else {}
    source_sf: Optional[int] = None
    if str(source_entry.get("type", "")) == "SOURCE":
        # SOURCE → block fresh field, name-agnostic (block5's source is named
        # x_mean or inv_std depending on the graph; both map to x_centered_fresh).
        rl_field = _ssm.source_rl_field(int(block_idx))
        sf = source_entry.get("sf")
        if sf is not None:
            try:
                source_sf = int(sf)
            except (TypeError, ValueError):
                source_sf = None
        if rl_field and source_sf is not None:
            out.field_baseline_sfs[rl_field] = source_sf
            out.field_kind_in_ro[rl_field] = "fresh"

    # 非第一项里带 sf_post 的 = rescale 动作；一个 RO 节点的 sf_post 可能同时
    # 映射多个 RL 字段（block 2 q/k 共享段的 ctpt_rotKT_mask2 sf_post 同时是
    # kt_mask2_r 和 q_mask2_r 的 baseline），迭代写入。
    for cp in cps[1:]:
        if not isinstance(cp, Mapping):
            continue
        if "sf_post" not in cp:
            continue
        name = str(cp.get("name", ""))
        sf_post = cp.get("sf_post")
        if sf_post is None:
            continue
        # node → RL rescale field(s) via the complete-chain SSOT (auto-adapts
        # when a skeleton regen moves which cut-points carry sf_post).
        rl_fields = _ssm.rescale_rl_fields(int(block_idx), name)
        if not rl_fields:
            out.unmapped_rescale_nodes.append(name)
            continue
        for rl_field in rl_fields:
            out.field_baseline_sfs[rl_field] = int(sf_post)
            out.field_kind_in_ro[rl_field] = "rescale"

    # --- propagation_deltas：numeric delta = encode 动作 ---
    pd_delta_by_name: Dict[str, Any] = {}
    for pd in entry.get("propagation_deltas") or []:
        if not isinstance(pd, Mapping):
            continue
        name = str(pd.get("name", ""))
        delta = pd.get("delta")
        if name:
            pd_delta_by_name[name] = delta
        if not isinstance(delta, (int, float)):
            continue   # "x2" 是平方乘 2，不是 encode 动作
        # node → RL encode field(s) via the complete-chain SSOT.
        rl_fields = _ssm.encode_rl_fields(int(block_idx), name)
        if not rl_fields:
            out.unmapped_propagation_nodes.append(name)
            continue
        for rl_field in rl_fields:
            out.field_baseline_sfs[rl_field] = int(delta)
            out.field_kind_in_ro[rl_field] = "encode"

    # --- block-4 derived: v_fresh.sf = SF(v*mask2) - SF(mask2) ---
    # v 在 Rescale_optimizer 的 block4 计算图里没有自己的 SOURCE 节点（V 经
    # softmax×V 才进入主链）。但 baseline 的 v_fresh SF 是可以推算出来的：
    # ``ctct_rot_softmax_mul_v`` 是 v*mask2 这次密文-密文乘法的结果，CKKS 里
    # ``SF(a*b) = SF(a) + SF(b)``，所以 ``SF(v) = SF(v*mask2) - SF(mask2)``，
    # 即 ``propagation_delta(ctct_rot_softmax_mul_v) - propagation_delta(ctpt_mask2)``。
    # 例如 mrpc baseline：39 - 14 = 25。
    if int(block_idx) == 4:
        mulv_delta = pd_delta_by_name.get("ctct_rot_softmax_mul_v")
        mask2_delta = pd_delta_by_name.get("ctpt_mask2")
        if isinstance(mulv_delta, (int, float)) and isinstance(mask2_delta, (int, float)):
            v_fresh_sf = int(mulv_delta) - int(mask2_delta)
            out.field_baseline_sfs["v_fresh_sf"] = int(v_fresh_sf)
            out.field_kind_in_ro["v_fresh_sf"] = "fresh"

    # --- block-2/5 "x2" 旁节点 → 辅助 fresh 字段（baseline = SOURCE.sf） ---
    # block 2 的 ``ctct_x_mean_over_std`` 旁节点 delta="x2"：x_centered_fresh
    # 的 SF 必须 = inv_std_fresh（SOURCE）的 SF。
    # block 5 的 ``ctct_xmean_over_std`` 旁节点 delta="x2"：inv_std_fresh 的
    # SF 必须 = x_centered_fresh（SOURCE）的 SF。
    # 之前没有这条推算 → x_centered/inv_std 默认 30，与真实 baseline 31/30 偶
    # 然碰得上时碰对，碰不上时（如 block 2 inv_std=31）就错。
    if source_sf is not None:
        aux_map = _RO_X2_AUX_FRESH_FIELD.get(int(block_idx), {})
        for side_name, aux_field in aux_map.items():
            if str(pd_delta_by_name.get(side_name)) == "x2":
                out.field_baseline_sfs[aux_field] = int(source_sf)
                out.field_kind_in_ro[aux_field] = "fresh"

    # --- modulus_chain cost ---
    mc = entry.get("modulus_chain") or {}
    if isinstance(mc, Mapping):
        out.total_bits = int(mc.get("total_bits", 0))
        out.drop_order = [int(x) for x in (mc.get("drop_order") or [])]

    # propagation_deltas 里 "x2" 的总数粗略代表 fusion 候选；这里跟随
    # RO 的 fusion_count 字段（如有）；否则 0
    out.fusion_count = int(entry.get("fusion_count", 0))

    # effective_rotations
    er = entry.get("effective_rotations") or []
    if isinstance(er, list):
        out.effective_rotations = [dict(x) for x in er if isinstance(x, Mapping)]

    return out


def load_static_skeletons_baseline(
        rescale_optimizer_root: str,
        dataset: str,
        num_layers: int,
        gelu_per_layer: Sequence[int],
        softmax_per_layer: Sequence[int],
        *,
        archive_path: Optional[str] = None,
        ) -> StaticSkeletonsBaseline:
    """从 ``static_skeletons_<dataset>.json`` 抽出 BLB Stage-2 RL baseline。

    Args:
        rescale_optimizer_root: ``Rescale_optimizer`` 仓库根目录
        dataset:                GLUE 任务名（mrpc / cola / ...）
        num_layers:             模型 encoder 层数
        gelu_per_layer:         长度 num_layers，元素 ∈ {1, 2, 4}（degree 0/ReLU 已于 2026-06-06 关闭）
        softmax_per_layer:      长度 num_layers，元素 ∈ {2, 3, 4, 5, 6}
        archive_path:           手动指定 archive 路径；缺省自动拼

    Returns:
        ``StaticSkeletonsBaseline``。``per_block_layer`` 不包含 (1, 0)
        —— layer-0 block1 噪声整体不安装。

    Raises:
        FileNotFoundError: archive 路径不存在
        BaselineHandoverError: archive schema 不对，或某层的 graph_key 不在 archive
    """
    _validate_stage1(gelu_per_layer, softmax_per_layer, int(num_layers))
    path = archive_path or static_skeletons_archive_path(rescale_optimizer_root, dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"static_skeletons archive not found: {path} "
            f"(确认 Rescale_optimizer 仓库已克隆并包含 configs/{dataset}/)"
        )
    archive = load_static_skeletons_archive(path)

    out = StaticSkeletonsBaseline(
        dataset=str(dataset),
        num_layers=int(num_layers),
        gelu_per_layer=[int(d) for d in gelu_per_layer],
        softmax_per_layer=[int(d) for d in softmax_per_layer],
        archive_path=str(path),
    )

    for layer_idx in range(int(num_layers)):
        gelu_deg = int(gelu_per_layer[layer_idx])
        softmax_deg = int(softmax_per_layer[layer_idx])
        # Block3 SF/fusion is baseline-owned: it is read from the same RO archive
        # as every other block even though the policy only chooses its truncation K.
        for block_idx in (1, 2, 3, 4, 5):
            # 语义对齐：layer-0 block 1 不安装噪声，跳过抽取
            if int(layer_idx) == 0 and int(block_idx) == 1:
                continue
            graph_key = static_skeletons_graph_key(
                block_idx, str(dataset), gelu_deg, softmax_deg,
            )
            entry = archive.get(graph_key)
            if entry is None:
                out.missing_block_layer.append((int(block_idx), int(layer_idx)))
                continue
            lb = _extract_one_block_layer(
                entry, block_idx, layer_idx, graph_key, gelu_degree=gelu_deg,
            )
            out.per_block_layer[(int(block_idx), int(layer_idx))] = lb
            out.aggregate_total_bits += int(lb.total_bits)
            out.aggregate_fusion_count += int(lb.fusion_count)
            out.aggregate_valid_block_count += 1

    if out.missing_block_layer:
        # archive 缺失关键 graph 是硬错（caller 必须处理）
        raise BaselineHandoverError(
            f"static_skeletons archive {path} 缺少以下 graph_key："
            + ", ".join(
                static_skeletons_graph_key(
                    b, dataset, gelu_per_layer[l], softmax_per_layer[l],
                ) + f"@layer={l}"
                for b, l in out.missing_block_layer
            )
        )

    return out


# ---------------------------------------------------------------------------
# Baseline → RL action_vec + calibrated MaxSFsTable
# ---------------------------------------------------------------------------
def static_skeletons_baseline_to_action(
        baseline: StaticSkeletonsBaseline,
        *,
        base_max_sfs: Optional["MaxSFsTable"] = None,
        snap_sf_to_noise_table: bool = True,
        ) -> Tuple[np.ndarray, "MaxSFsTable", BaselineCostStats, Dict[str, Any]]:
    """把 ``StaticSkeletonsBaseline`` 转换成 RL 可直接消费的三元组：

      * ``action_vec``: ``np.ndarray[int]``，长度 ``sum(action_dims_for_config(num_layers))``。
                         所有 slot 都取 max idx；对于 baseline 里有 JSON SF 的 slot，
                         max_sf 被校准为 baseline SF —— 即 max-idx ↔ baseline。
      * ``max_sfs``:    校准过的 ``MaxSFsTable``。可直接喂给 ``BLBStage2Env``/``action_vector_to_cfgs``。
      * ``cost_stats``: ``BaselineCostStats``（total_bits / fusion / avg_k）。
      * ``diagnostics``: 报告每层每 block 的 active / inactive slot、unmapped 节点等。

    Args:
        baseline:                  ``load_static_skeletons_baseline`` 输出
        base_max_sfs:              基础 max_sfs（缺省时用 ``load_max_sfs(dataset)``）。
                                   未被 JSON 覆盖的 slot 保留 base_max_sfs 取值。
        snap_sf_to_noise_table:    True ⇒ 把每个 calibrated max_sf 钳到 noise table
                                   允许的最近合法值（保证 RL 动作落到 noise table 里）。
                                   False ⇒ 原样使用 JSON SF。
    """
    # Lazy import：避免循环 / 测试装载顺序问题
    from .action_space import (
        K_LEVELS, MaxSFsTable, NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N,
        _BLOCK_NODE_NAME_BY_FIELD, _BLOCK_SPECS, NUM_LEVELS_PER_DIM_BY_BLOCK_KIND,
        _block_default_N, action_dims_for_config, layer_dims, load_max_sfs,
        make_all_max_action_vector, per_layer_field_offsets, sf_from,
    )

    L = int(baseline.num_layers)
    base = base_max_sfs if base_max_sfs is not None else load_max_sfs(baseline.dataset)
    # deepcopy by reconstructing the dict
    calibrated = MaxSFsTable(
        by_block_node=dict(base.by_block_node),
        by_layer_block_node=dict(getattr(base, "by_layer_block_node", {}) or {}),
    )

    # 计算每个 RL 字段的（block_idx, ro_node_name）—— 用 _BLOCK_NODE_NAME_BY_FIELD 反查
    # 但我们抽出的 baseline 字段名可能不在 _BLOCK_NODE_NAME_BY_FIELD（例如 block 3
    # 的 square_rescale_sf_<k>）。所以这里需要做一遍 field_name → ro_node_name 的反向构建。

    # 反向 map：(block, field) → ro_node
    field_to_node: Dict[Tuple[int, str], str] = {}
    for b, dct in _BLOCK_NODE_NAME_BY_FIELD.items():
        for fname, node in dct.items():
            field_to_node[(int(b), str(fname))] = str(node)

    diagnostics: Dict[str, Any] = {
        "active_slot_count": 0,
        "fresh_slot_count": 0,
        "encode_slot_count": 0,
        "rescale_slot_count": 0,
        "inactive_rescale_slots": [],   # 在 JSON 里 NOT 是 cut point 的 RL rescale 字段
        "unmapped_nodes": {"propagation": [], "rescale": []},
        "calibrated_max_sfs": {},        # (block, field) → max_sf (for caller 审计)
    }

    # 把 baseline 抽出的 SF 校准到 MaxSFsTable
    for (block_idx, layer_idx), lb in baseline.per_block_layer.items():
        # 不同 layer 同 block 的 baseline SF 应当一致（同一个 graph）；如果不一致以最后一次为准
        for field_name, sf in lb.field_baseline_sfs.items():
            target_node = field_to_node.get((int(block_idx), str(field_name)))
            if target_node is None:
                # 字段在 RL 里但没有 RO node 注册（极少；通常表示 field_name 自身是 RO node）
                # 直接用 field_name 作为 node key
                target_node = str(field_name)
            calibrated_sf = int(sf)
            if snap_sf_to_noise_table:
                # block 1/2/4 N 由 _block_default_N 决定（与 gelu/softmax 无关）；
                # block 3/5 与该层 degree 相关，但 RL 一侧 ladder 是 per-block-per-field
                # 共享的 max，所以我们用该 block 的"通用 N"做 snap：
                #   block 1 N=8192；block 3 用 layer 的 attn_degree；block 5 用 gelu。
                N = _block_default_N(
                    int(block_idx),
                    gelu_degree=baseline.gelu_per_layer[layer_idx],
                    attn_degree=baseline.softmax_per_layer[layer_idx],
                )
                allowed = list(NOISE_TABLE_ALLOWED_SCALING_FACTORS_BY_N.get(int(N), ()))
                if allowed and calibrated_sf not in allowed:
                    le = [v for v in allowed if v <= calibrated_sf]
                    calibrated_sf = max(le) if le else min(allowed)
            calibrated.by_layer_block_node[
                (int(layer_idx), int(block_idx), target_node)
            ] = int(calibrated_sf)
            diagnostics["calibrated_max_sfs"][
                f"L{layer_idx}.block{block_idx}.{field_name}"
            ] = int(calibrated_sf)
            diagnostics["active_slot_count"] += 1
            kind = lb.field_kind_in_ro.get(field_name, "")
            if kind == "fresh":
                diagnostics["fresh_slot_count"] += 1
            elif kind == "encode":
                diagnostics["encode_slot_count"] += 1
            elif kind == "rescale":
                diagnostics["rescale_slot_count"] += 1
        # 收集 unmapped
        for nd in lb.unmapped_propagation_nodes:
            diagnostics["unmapped_nodes"]["propagation"].append(
                f"block{block_idx}.{nd}@L{layer_idx}"
            )
        for nd in lb.unmapped_rescale_nodes:
            diagnostics["unmapped_nodes"]["rescale"].append(
                f"block{block_idx}.{nd}@L{layer_idx}"
            )

    # 构造 RL action_vec。非 rescale slot 取 max idx；rescale slot 只有在
    # JSON sf_post/drop 中出现时才取 max idx，未出现时保持 index 0 (= off)。
    action_vec = make_all_max_action_vector(L)
    fields = per_layer_field_offsets()
    layer_dim = len(fields)
    active_rescale_slots = {
        (int(layer_idx), int(block_idx), str(field_name))
        for (block_idx, layer_idx), lb in baseline.per_block_layer.items()
        for field_name, kind in lb.field_kind_in_ro.items()
        if str(kind) == "rescale"
    }

    # 找出 RL 里有但 JSON 里没的 rescale 字段（=> "off at RO baseline"），仅做诊断报告
    all_rescale_fields_per_block: Dict[int, List[str]] = {b: [] for b in (1, 2, 3, 4, 5)}
    for b, spec in _BLOCK_SPECS.items():
        for fname, kind, _max in spec.fields:
            if str(kind) == "R":
                all_rescale_fields_per_block[int(b)].append(fname)
    for (block_idx, layer_idx), lb in baseline.per_block_layer.items():
        for fname in all_rescale_fields_per_block.get(int(block_idx), []):
            if fname not in lb.field_baseline_sfs:
                diagnostics["inactive_rescale_slots"].append(
                    f"block{block_idx}.{fname}@L{layer_idx}"
                )
    for li in range(L):
        for field_offset, (block_idx, field_name, kind) in enumerate(fields):
            if str(kind) != "R":
                continue
            if (li, int(block_idx), str(field_name)) not in active_rescale_slots:
                action_vec[int(li * layer_dim + field_offset)] = 0

    # BaselineCostStats（avg_k 反映 per-block baseline K：B1=13, B2=10, B3=13, B4=10, B5=13）。
    # Layer 0 没有 block 1（首层无前置 FFN2），所以 K 槽位总数 = 4·L + (L-1) ≈ 5L-1。
    # RO 不参与 K，所以 baseline avg_k 仅由 BASELINE_K_BY_BLOCK 决定。
    from .action_space import BASELINE_K_BY_BLOCK
    k_sum = 0.0
    k_count = 0
    for li in range(L):
        for b in (1, 2, 3, 4, 5):
            if li == 0 and b == 1:
                continue  # layer 0 block 1 K is forced None
            k_sum += float(BASELINE_K_BY_BLOCK.get(int(b), max(K_LEVELS)))
            k_count += 1
    baseline_avg_k = (k_sum / max(k_count, 1)) if k_count else float(max(K_LEVELS))
    cost_stats = BaselineCostStats(
        total_bits_sum=int(baseline.aggregate_total_bits),
        total_fusion_count=int(baseline.aggregate_fusion_count),
        avg_k=float(baseline_avg_k),
        typical_bits_drop=1.0,
        typical_fusion_count=1.0,
        typical_k_drop=1.0,
    )

    return action_vec, calibrated, cost_stats, diagnostics
