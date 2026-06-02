# Stage-2 fusion-count 动作 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **项目约定（覆盖 skill 默认）**：① 测试用 `unittest`（`python -m unittest ...`），不用 pytest。② Commit 只在用户要求时执行——计划里的 commit 步是**建议检查点**，执行时先征得同意。③ map 全量构建 + 需 torch 的验证在**服务器**经 `SERVER_COMMAND.md` 跑；源码只在本地改。④ 严守 `[[stage2-skeleton-driven-ssot]]`：所有槽/baseline/t_new 仍从 skeleton 派生，本计划不写死。

**Goal:** 把 Stage-2 sequential RL 每个 block 的动作从「24 个 per-slot SF 头」换成「`(fusion_option, K)` 两选项」，`fusion_option` 经离线 fusion-count 映射表展开成现有 full SF vec，搜索空间大幅缩小。

**Architecture:** 离线 builder 对 7 种 block-type 枚举 effective chain 槽 → replan → 按 realized `fusion_count` 分组 → 取 post-override 实际安装方差最小集（option0 强制=baseline）→ 缓存 JSON。运行期同一 `BLBStage2SequentialPolicy` 用 `max_step_dim=2` 实例化，env 用映射表展开动作，复用全部下游管线与 reward。

**Tech Stack:** Python / numpy / PyTorch（仅 policy/env 运行期）/ torch-free `rescale_optimizer` + `action_space` + `rescale_optimizer_bridge` / `unittest`。

参考 spec：`docs/superpowers/specs/2026-06-03-stage2-fusion-count-action-design.md`（§编号下文引用）。

---

## 文件结构与职责

| 文件 | 职责 | 新/改 |
|------|------|-------|
| `blb_stage2_rl/fusion_count_map.py` | `NoiseOrder` 接口 + 默认实现；`FusionOption`/`BlockTypeFusionMap`/`FusionCountMap` 数据类与加载器；`options/num_options/expand/baseline_option_id/max_num_options`。纯数据+查表，torch-free。 | 新 |
| `blb_stage2_rl/fusion_enum.py` | builder 的纯逻辑核：每 block-type 的 effective-chain-槽 发现、枚举、replan 评估、安装方差、分组+去重+排序。torch-free（依赖 action_space + bridge + rescale_optimizer）。 | 新 |
| `scripts/blb_build_fusion_count_map.py` | CLI 封装 `fusion_enum`：多核并行、写 JSON 缓存、K 独立性自检、HTML/JSON 报告。 | 新 |
| `blb_stage2_rl/fusion_maps/mrpc/*.json` | 7 个 block-type 的映射（服务器产物，入 git）。 | 新（服务器） |
| `blb_stage2_rl/action_space.py` | fusion-mode step schedule + `expand_fusion_step_action(...)`；不动现有 per-slot schedule。 | 改 |
| `blb_stage2_rl/sequential_env.py` | fusion 分支：`evaluate_step` 前用 map 展开 `(option,K)`→ 块 slot 向量。 | 改 |
| `blb_stage2_rl/sequential_runner.py` | 载入 map；fusion 分支 schedule/掩码/`preferred=[0,k_idx]`；`max_step_dim=2` 实例化 policy；停用 mask/radius2；ckpt variant；flag 分发。 | 改 |
| `rl_tune.py` / `layer_importance_evaluator.py` / `llama_7B_LayerImportance.sh` | flag `--blb-v3-fusion-count-action` 穿透 + 互斥检查 + preset。 | 改 |
| `tests/test_blb_fusion_count_map.py` | torch-free 单测：NoiseOrder、enum 小 block、去重、option0=baseline、expand 往返、block3=1-option。 | 新 |
| `SERVER_COMMAND.md` | F0 全量构建 + F1 短 smoke 命令。 | 改 |

---

## Task 1: torch-free 噪声表 + `NoiseOrder` 接口 + 默认安装方差实现

实现 spec §3.3 的可插拔偏序。**本地无 torch**，故噪声表必须 torch-free 取得：新建**仓根** `noise_tables.py`（无相对导入、无 torch、无 package `__init__`），AST 抽取 `function_handler.py` 的 `_NOISE_STD_RAW`（口径同 `blb_verify_noise_install.load_noise_variance_table`：`enc²/fresh²/rescale²/rotation=rescale²`）。`fusion_count_map` 仅 `import noise_tables`（仓根恒在 sys.path），不碰 `function_handler`/`action_space`，从而 torch-free 顶层可导入。

**Files:**
- Create: `noise_tables.py`（仓根，torch-free）
- Create: `blb_stage2_rl/fusion_count_map.py`
- Test: `tests/test_blb_fusion_count_map.py`（顶部 `sys.path.insert` 仓根 + `blb_stage2_rl`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_blb_fusion_count_map.py  (顶部 sys.path 注入 blb_stage2_rl 目录，绕过 package __init__ 的 torch，
#  仿 tests/test_blb_skeleton_stage_map.py 的写法)
import unittest
import fusion_count_map as fcm  # via sys.path insert of blb_stage2_rl/

class NoiseOrderTest(unittest.TestCase):
    def test_summed_installed_variance_sums_table_values(self):
        order = fcm.SummedInstalledVariance()
        pts = [
            fcm.InstalledNoisePoint(scaling_factor=30, distribution="fresh", N=8192),
            fcm.InstalledNoisePoint(scaling_factor=22, distribution="encoding", N=8192),
        ]
        # 期望 == NOISE_VARIANCE_TABLE_BY_N[8192][30]["fresh"] + [22]["encoding"]
        import noise_tables
        T = noise_tables.NOISE_VARIANCE_TABLE_BY_N
        expected = T[8192][30]["fresh"] + T[8192][22]["encoding"]
        self.assertAlmostEqual(order.total_variance(pts), expected, places=18)

    def test_empty_plan_is_zero(self):
        self.assertEqual(fcm.SummedInstalledVariance().total_variance([]), 0.0)

    def test_name_is_stable(self):
        self.assertEqual(fcm.SummedInstalledVariance().name, "summed_installed_variance")
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `cd <repo> && PYTHONPATH=blb_stage2_rl:Rescale_optimizer:. python -m unittest tests.test_blb_fusion_count_map.NoiseOrderTest -v`
Expected: FAIL（`ModuleNotFoundError: fusion_count_map` 或 attr 缺失）。

- [ ] **Step 3: 实现最小代码**

```python
# blb_stage2_rl/fusion_count_map.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Protocol, runtime_checkable

import noise_tables   # repo-root torch-free module (always on sys.path)
NOISE_VARIANCE_TABLE_BY_N = noise_tables.NOISE_VARIANCE_TABLE_BY_N

@dataclass(frozen=True)
class InstalledNoisePoint:
    scaling_factor: int
    distribution: str    # fresh / encoding / rescale / rotation
    N: int

@runtime_checkable
class NoiseOrder(Protocol):
    name: str
    def total_variance(self, installed_points: Sequence[InstalledNoisePoint]) -> float: ...

class SummedInstalledVariance:
    name = "summed_installed_variance"
    def total_variance(self, installed_points: Sequence[InstalledNoisePoint]) -> float:
        total = 0.0
        for p in installed_points:
            total += float(NOISE_VARIANCE_TABLE_BY_N[int(p.N)][int(p.scaling_factor)][str(p.distribution)])
        return total
```

- [ ] **Step 4: 跑测试确认 PASS**

Run: 同 Step 2。Expected: PASS（3 tests）。

- [ ] **Step 5: 建议 commit 检查点**（征得同意后）

```bash
git add blb_stage2_rl/fusion_count_map.py tests/test_blb_fusion_count_map.py
git commit -m "feat(stage2-fusion): add pluggable NoiseOrder + installed-variance default"
```

---

## Task 2: `FusionOption` / `FusionCountMap` 数据类 + 加载器 + `expand`

spec §3.7 / §4.2 的查表层。option0=baseline；`expand` 返回**整块 slot 向量**（含 K 占位），env 再覆盖 K。

**Files:**
- Modify: `blb_stage2_rl/fusion_count_map.py`
- Test: `tests/test_blb_fusion_count_map.py`

- [ ] **Step 1: 写失败测试**（用内存里构造的小 map dict，不读盘）

```python
class FusionMapLoaderTest(unittest.TestCase):
    def _toy(self):
        payload = {
            "profile": "mrpc",
            "graphs": {
                "block1_mrpc": {
                    "graph_key": "block1_mrpc",
                    "k_slot_index": 8,             # block1 K 在第 8 槽
                    "block_num_slots": 9,
                    "options": [
                        {"option_id": 0, "fusion_count": 1, "tie_index": 0,
                         "total_variance": 1.0, "total_bits": 100,
                         "slots": {"gelu_out_sf": 30},
                         "action_indices": [4,4,2,2,3,3,0,0,3]},   # 全 max = baseline
                        {"option_id": 1, "fusion_count": 2, "tie_index": 0,
                         "total_variance": 2.0, "total_bits": 90,
                         "slots": {"gelu_out_sf": 28},
                         "action_indices": [3,4,2,2,0,3,0,0,3]},
                    ],
                },
            },
            "max_num_options": 2,
        }
        return fcm.FusionCountMap.from_payload(payload)

    def test_baseline_option_is_zero(self):
        m = self._toy()
        self.assertEqual(m.baseline_option_id("block1_mrpc"), 0)

    def test_num_and_max_options(self):
        m = self._toy()
        self.assertEqual(m.num_options("block1_mrpc"), 2)
        self.assertEqual(m.max_num_options(), 2)

    def test_expand_overwrites_k_slot(self):
        m = self._toy()
        # option 1, K 槽(idx 8)用 k_index=5 覆盖；其余照 action_indices
        out = m.expand("block1_mrpc", option_id=1, k_index=5)
        self.assertEqual(list(out), [3,4,2,2,0,3,0,0,5])
```

- [ ] **Step 2: 跑测试确认 FAIL**

Run: `PYTHONPATH=blb_stage2_rl:Rescale_optimizer:. python -m unittest tests.test_blb_fusion_count_map.FusionMapLoaderTest -v`
Expected: FAIL。

- [ ] **Step 3: 实现**

```python
# 追加到 fusion_count_map.py
import json, numpy as np, pathlib

@dataclass(frozen=True)
class FusionOption:
    option_id: int
    fusion_count: int
    tie_index: int
    total_variance: float
    total_bits: int
    slots: Dict[str, int]
    action_indices: List[int]   # 整块 slot 向量（K 位为 baseline 占位）

@dataclass
class BlockTypeFusionMap:
    graph_key: str
    k_slot_index: int
    block_num_slots: int
    options: List[FusionOption]   # options[0] == baseline

@dataclass
class FusionCountMap:
    profile: str
    graphs: Dict[str, BlockTypeFusionMap]
    _max_num_options: int

    @classmethod
    def from_payload(cls, payload: dict) -> "FusionCountMap":
        graphs = {}
        for gk, g in payload["graphs"].items():
            opts = [FusionOption(**{**o, "action_indices": list(o["action_indices"])})
                    for o in g["options"]]
            assert opts and opts[0].option_id == 0, f"{gk}: option0 must be baseline"
            graphs[gk] = BlockTypeFusionMap(
                graph_key=g["graph_key"], k_slot_index=int(g["k_slot_index"]),
                block_num_slots=int(g["block_num_slots"]), options=opts)
        return cls(profile=payload["profile"], graphs=graphs,
                   _max_num_options=int(payload["max_num_options"]))

    @classmethod
    def load(cls, profile: str, root: str | None = None) -> "FusionCountMap":
        base = pathlib.Path(root or pathlib.Path(__file__).parent) / "fusion_maps" / profile
        graphs, mx = {}, 6
        merged = {"profile": profile, "graphs": {}, "max_num_options": 6}
        for p in sorted(base.glob("*.json")):
            g = json.loads(p.read_text())
            merged["graphs"][g["graph_key"]] = g
            mx = max(mx, len(g["options"]))
        merged["max_num_options"] = mx
        return cls.from_payload(merged)

    def options(self, graph_key: str) -> List[FusionOption]:
        return self.graphs[graph_key].options
    def num_options(self, graph_key: str) -> int:
        return len(self.graphs[graph_key].options)
    def baseline_option_id(self, graph_key: str) -> int:
        return 0
    def max_num_options(self) -> int:
        return self._max_num_options

    def expand(self, graph_key: str, option_id: int, k_index: int) -> np.ndarray:
        g = self.graphs[graph_key]
        opt = g.options[int(option_id)]
        vec = np.asarray(opt.action_indices, dtype=int).copy()
        vec[int(g.k_slot_index)] = int(k_index)   # K 是独立第二决策，覆盖占位
        return vec
```

- [ ] **Step 4: 跑测试确认 PASS**；**Step 5: 建议 commit**

```bash
git add blb_stage2_rl/fusion_count_map.py tests/test_blb_fusion_count_map.py
git commit -m "feat(stage2-fusion): FusionCountMap loader + expand with separate K"
```

---

## Task 3: builder 核心 `fusion_enum.py`（effective-chain 槽发现 + 枚举 + 安装方差 + 分组）

spec §3.2/§3.3/§3.4。torch-free（用 `InProcessInvoker.from_profile` + `RescaleOptimizerBridge`）。

**Files:**
- Create: `blb_stage2_rl/fusion_enum.py`
- Test: `tests/test_blb_fusion_count_map.py`（加 `BuilderSmallBlockTest`，仅 block1，几千组合，本地可跑）

**关键子函数（接口）：**
- `effective_chain_slots(block_idx, gelu_degree, attn_degree, profile) -> List[SlotEnumSpec]`：返回参与枚举的槽（field, kind, slot_index, level_count）。判定：`_is_action_field_effective(...)` 为真 **且** 该 field 进 replan——即 field 在 `default_block{n}_cfg_to_delta` 读取集合 ∪ skeleton t_new（fresh/active rescale）。其余 effective 槽（model-only，如 block2 `wv_sf`）→ 固定 max 档，不进枚举。
- `installed_noise_points(block_idx, cfg_after_override, optimizer_raw) -> List[InstalledNoisePoint]`：复用 `apply_optimizer_output_to_cfg` 之后的 cfg；遍历 cfg 的 fresh/encode/rescale 非 None 噪声字段 + 优化器 effective_rotations，按字段 kind→distribution、`_block_default_N` 取 N、cfg 的 SF 取 scaling_factor，产出点列表。（与 `scripts/blb_verify_noise_install.py` 的安装计划口径对齐——执行时核对该脚本的 per-node σ² 逻辑后实现，**单测对拍**。）
- `build_block_type(graph_key, block_idx, gelu_degree, attn_degree, profile, bridge, noise_order) -> dict`：从 all-max 块向量出发，对 effective-chain 槽做笛卡尔积；每组合 build cfg（`build_block{n}_cfg_from_action`）→ `bridge.evaluate(config_name=f"{graph_key}_L0", block_name, cfg)` → 跳过 invalid → `apply_optimizer_output_to_cfg`(+绑定同步) → 安装方差 → 收集 `(realized fusion_count, variance, total_bits, full_block_action_indices, installed_cfg_signature)`。然后：按 fusion_count 分组；每组取最小 variance（epsilon 容差）；按 installed_cfg_signature 去重；**option0 强制=baseline**（断言 baseline 组合 valid 且其 SF 切片==`make_all_max_action_vector` 对应切片）；其余按 `(fusion_count, variance, total_bits, 字典序)` 排序；输出 §3.7 JSON 结构（含 `k_slot_index`/`block_num_slots`）。

- [ ] **Step 1: 写失败测试（block1 小枚举，断言 baseline=option0 且还原 all-max）**

```python
@unittest.skipUnless(_BRIDGE_AVAILABLE, "rescale bridge / RO not importable")
class BuilderSmallBlockTest(unittest.TestCase):
    def test_block1_baseline_is_option0_and_all_max(self):
        import fusion_enum
        bridge = _make_inprocess_bridge("mrpc")     # helper: InProcessInvoker.from_profile + RescaleOptimizerBridge
        out = fusion_enum.build_block_type("block1_mrpc", 1, gelu_degree=4, attn_degree=4,
                                           profile="mrpc", bridge=bridge,
                                           noise_order=fcm.SummedInstalledVariance())
        opt0 = out["options"][0]
        self.assertEqual(opt0["option_id"], 0)
        # baseline 还原 all-max 的 block1 切片
        from action_space import make_all_max_action_vector, _full_vec_offset_for_block, block_dims
        full = make_all_max_action_vector(12)
        off = _full_vec_offset_for_block(12, 0, 1); n = len(block_dims(1))
        self.assertEqual(list(opt0["action_indices"]), list(full[off:off+n]))

    def test_every_option_valid_and_distinct_fusion_groups(self):
        ...  # 断言 options 覆盖 ≥1 个 fusion_count，且每 option total_variance 单调（组内最小）
```

- [ ] **Step 2: 跑测试确认 FAIL** → **Step 3: 实现 `fusion_enum.py`** → **Step 4: PASS**

Run: `PYTHONPATH=blb_stage2_rl:Rescale_optimizer:. python -m unittest tests.test_blb_fusion_count_map.BuilderSmallBlockTest -v`

> 本地若 `InProcessInvoker` 因缺 RO baseline 不可用，则该类 `skipUnless` 跳过；真正验证在服务器（Task 4/8）。block1 组合数 ~3.6K，本地秒级。

- [ ] **Step 5: 建议 commit**

```bash
git add blb_stage2_rl/fusion_enum.py tests/test_blb_fusion_count_map.py
git commit -m "feat(stage2-fusion): block-type enumeration + installed-variance grouping"
```

---

## Task 4: builder CLI `scripts/blb_build_fusion_count_map.py` + 服务器全量构建

spec §3.6/§3.8。多核并行；K 独立性自检；HTML/JSON 报告。

**Files:**
- Create: `scripts/blb_build_fusion_count_map.py`
- Create（服务器产物）: `blb_stage2_rl/fusion_maps/mrpc/{block1_mrpc,block2_mrpc,block4,block5_n0,block5_n1,block5_n2,block5_n4}.json`
- Modify: `SERVER_COMMAND.md`

- [ ] **Step 1:** CLI：`--profile mrpc --out-dir blb_stage2_rl/fusion_maps/mrpc --workers N --report <html>`；对 7 个 (graph_key, block_idx, gelu_degree, attn_degree) 调 `fusion_enum.build_block_type`（block5_n{0,1,2,4} 用对应 gelu_degree；block3 跳过——单独写 1-option baseline，见 §3.5），多进程并行；写每类型 JSON + 汇总 `max_num_options` + 报告。K 独立性自检：对每类型抽样 cut_point 配置扫 6 档 K，断言 fusion_count 不变，写进报告。
- [ ] **Step 2:** 本地干跑 block1+block5_n0（小）确认 CLI 正常：
  Run: `PYTHONPATH=. python scripts/blb_build_fusion_count_map.py --profile mrpc --only block1_mrpc,block5_n0 --out-dir /tmp/fm --report /tmp/fm/report.html`
  Expected: 生成 2 个 JSON + 报告；option0=baseline。
- [ ] **Step 3:** 写 `SERVER_COMMAND.md` 的 active 命令：全量 7 类型构建 + 报告，输出到 `blb_stage2_rl/fusion_maps/mrpc/` 与 `experiments/server_command_runs/fusion_map_build_<ts>/`；SUMMARY 打印每类型 option 数 / fusion_count 分布 / K 独立性 / 耗时。
- [ ] **Step 4（F0 门槛检查，spec §8）：** 服务器构建后回传报告，人工核对：每类型 #options 是否合理（非病态大）、是否 ≥2 个 fusion_count、baseline=option0。异常则停下复审。
- [ ] **Step 5: 建议 commit**（builder 源码本地 commit；map JSON 由服务器构建后 push 回，本地 pull）

```bash
git add scripts/blb_build_fusion_count_map.py SERVER_COMMAND.md
git commit -m "feat(stage2-fusion): map builder CLI + server full-build command"
```

---

## Task 5: action_space fusion-mode schedule + `expand_fusion_step_action`

spec §4.1/§4.2。新增，不动现有 per-slot schedule。

**Files:**
- Modify: `blb_stage2_rl/action_space.py`
- Test: `tests/test_blb_fusion_count_map.py`（加 `FusionScheduleTest`，torch-free）

**接口：**
- `fusion_step_schedule(num_layers, profile, attn_degree_per_layer, gelu_degree_per_layer, fusion_map) -> List[FusionStepSpec]`：与现 `step_schedule` 同样的 `(layer,block)` 顺序（L0 无 block1），每步带 `graph_key_suffix`、`block_idx`、`layer_idx`、`fusion_num_options=fusion_map.num_options(gk)`、`k_num_levels=LEVELS_K`、`k_slot_index`、`block_num_slots`、`full_vec_offset`。block3 步 `fusion_num_options=1`。
- `expand_fusion_step_action(spec, fusion_map, option_id, k_index) -> np.ndarray`：返回该块整 slot 向量（`fusion_map.expand(gk, option_id, k_index)`）。
- `fusion_step_schedule_dims(...) -> (max_step_dim=2, max_num_levels=max(fusion_map.max_num_options(), LEVELS_K))`：供 policy 实例化。

- [ ] **Step 1: 写失败测试**

```python
class FusionScheduleTest(unittest.TestCase):
    def test_dims_are_two_slots(self):
        import action_space as A
        m = _toy_full_map()   # 7 类型各 ≥1 option 的内存 map
        specs = A.fusion_step_schedule(12, "mrpc", [4]*12, [4]*12, m)
        self.assertEqual(len(specs), A.horizon_for_num_layers(12))   # 同 horizon=59
        md, mnl = A.fusion_step_schedule_dims(12, "mrpc", [4]*12, [4]*12, m)
        self.assertEqual(md, 2)
        self.assertGreaterEqual(mnl, A.LEVELS_K)
    def test_block3_step_is_single_option(self):
        ...  # 找到某 block3 步，断言 fusion_num_options == 1
    def test_expand_matches_map(self):
        ...  # expand_fusion_step_action(spec, m, 0, k_index=3) == m.expand(gk,0,3)
```

- [ ] **Step 2: FAIL** → **Step 3: 实现**（复用现有 `step_schedule` 的顺序/offset 计算，只换每步的槽语义）→ **Step 4: PASS**

Run: `PYTHONPATH=blb_stage2_rl:Rescale_optimizer:. python -m unittest tests.test_blb_fusion_count_map.FusionScheduleTest -v`

- [ ] **Step 5: 建议 commit**

```bash
git add blb_stage2_rl/action_space.py tests/test_blb_fusion_count_map.py
git commit -m "feat(stage2-fusion): fusion-mode step schedule + action expansion"
```

---

## Task 6: sequential_env fusion 分支（map 展开 → 现有 evaluate/commit）

spec §4.2。`evaluate_step` 接受 `(option_id, k_index)`，展开成块 slot 向量后复用现有 splice + replan + override + reward。

**Files:**
- Modify: `blb_stage2_rl/sequential_env.py`
- Test: 由 Task 8 的服务器 smoke 覆盖端到端（env 需 torch 模型 forward）；本地加一个**不触发终局 forward** 的 `evaluate_step` 单元（用 stub bridge）断言展开正确。

- [ ] **Step 1:** 给 `BLBStage2SequentialEnv` 加 `fusion_map` + `fusion_specs` 可选成员；`evaluate_step` 在 fusion 模式下：把入参 `(option_id, k_index)` 经 `expand_fusion_step_action` 展开成块 slot 向量，再走原有 `temp_vec` splice + `action_vector_to_cfgs` + `bridge.evaluate` 路径（其余完全不变）。非 fusion 模式保持原 per-slot 行为。
- [ ] **Step 2:** 本地单测（stub bridge）：构造 1 步，`evaluate_step((0, k_idx=3))` 后断言 `temp_vec` 的该块切片 == baseline all-max 切片。
  Run: `PYTHONPATH=blb_stage2_rl:Rescale_optimizer:. python -m unittest tests.test_blb_fusion_count_map.FusionEnvExpandTest -v`
- [ ] **Step 3: 建议 commit**

```bash
git add blb_stage2_rl/sequential_env.py tests/test_blb_fusion_count_map.py
git commit -m "feat(stage2-fusion): sequential env expands (option,K) via map"
```

---

## Task 7: sequential_runner 接线 + flag 穿透 + 互斥检查 + preset

spec §4.1/§4.3/§4.4/§6。

**Files:**
- Modify: `blb_stage2_rl/sequential_runner.py`, `rl_tune.py`, `layer_importance_evaluator.py`, `llama_7B_LayerImportance.sh`

- [ ] **Step 1:** `rl_tune.py` + `layer_importance_evaluator.py` 加 `blb_v3_fusion_count_action: bool=False` 穿透到 `BLBStage2TrainConfig.fusion_count_action`（仿现有 `blb_v3_substage_mode` 路径）。launcher 加 flag `--blb-v3-fusion-count-action` 与互斥检查（与 `--blb-v3-substage-mode` 不可同开），并入 mrpc 训练 preset。
- [ ] **Step 2:** `sequential_runner` fusion 分支：① `FusionCountMap.load(profile)`；② `fusion_step_schedule` + `fusion_step_schedule_dims` → 用 `max_step_dim=2`、`max_num_levels` 实例化 `BLBStage2SequentialPolicy`；③ 每步 `(slot_mask, num_levels, action_level_mask)`：slot0→`fusion_num_options`、slot1→`LEVELS_K`；④ warmstart `apply_preferred_per_step_bias([0, K_LEVELS.index(BASELINE_K)])`；anchor 强制 `(0, baseline_k_idx)`；⑤ **不**构建/调用 `StaticInvalidLevelMask`/`ForbiddenActionMask`/`EmpiricalInvalidLevelMask`/safe-neighbor/`GuardedRadius2Controller`；⑥ ckpt variant 设 `blb_v3_sequential_gtrxl_fusioncount_v1`，resume 时 variant 不符则要求 `--fresh`；⑦ env 注入 `fusion_map`+`fusion_specs`。diagnostics 额外记 `option_id`/nominal `fusion_count`/`k_index`。
- [ ] **Step 3:** 跑现有 torch-free 套件确认无回归（per-slot 路径不变）：
  Run: `make test`（= `BLB_STRICT=0 python -m unittest discover -s tests -p "test_blb_*.py" -v`）
  Expected: 现有用例不回归；新 fusion 用例 PASS。
- [ ] **Step 4:** `make preset-check` 验证新 flag/preset 无误。
- [ ] **Step 5: 建议 commit**

```bash
git add blb_stage2_rl/sequential_runner.py rl_tune.py layer_importance_evaluator.py llama_7B_LayerImportance.sh
git commit -m "feat(stage2-fusion): runner wiring, flag, preset, mutual-exclusion with substage"
```

---

## Task 8: 服务器 F1 短 smoke（端到端验证）

spec §8。需 torch；经 `SERVER_COMMAND.md`。

**Files:** Modify `SERVER_COMMAND.md`

- [ ] **Step 1:** 写 active 命令：先确认 `blb_stage2_rl/fusion_maps/mrpc/*.json` 已在仓内；跑 fusion-mode 200–500 episode：
  `CUDA_VISIBLE_DEVICES=0,1,2,3 bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --blb-v3-fusion-count-action 1 --stage2-k-trials 4 --blb-v3-reward-devices 0,1,2,3 --fresh`（episode 数用现有短跑参数）。
- [ ] **Step 2:** SUMMARY 校验：`(option,K)` 展开正确（diagnostics 有 option_id/fusion_count）、reward 正常非 -150 常数、anchor 期 best==baseline、entropy/clip/KL 健康、四卡 probe 激活日志、无崩溃。
- [ ] **Step 3:** 回传 artifacts + 报告；本地 pull 核对。异常→本地定位修复→push→重跑。
- [ ] **Step 4: 建议 commit**（仅 `SERVER_COMMAND.md` 源改动；smoke 产物按需入 git）。

---

## Task 9: 文档 + 记忆收尾

- [ ] **Step 1:** 更新 `CLAUDE.md` 相关段（动作表示、fusion-count map、停用 substage 的现状）与 `docs/adr/`（新增 ADR：fusion-count 动作取代逐槽 SF）。
- [ ] **Step 2:** 更新记忆：新增 `stage2-fusion-count-action.md`（指向 spec/plan + 关键决策），`MEMORY.md` 加索引；与 `[[stage2-skeleton-driven-ssot]]` 互链。
- [ ] **Step 3: 建议 commit**

```bash
git add CLAUDE.md docs/adr/ ~/.claude/.../memory  # 记忆在 home，单独处理
git commit -m "docs(stage2-fusion): ADR + CLAUDE.md + memory for fusion-count action"
```

---

## Self-Review（plan vs spec）

- **Spec 覆盖**：§3.1→T3/T4；§3.2 effective-chain/model-only→T3；§3.3 NoiseOrder→T1；§3.4 分组/去重/option0=baseline→T3；§3.5 block3→T4(1-option)/T5(schedule)；§3.6 K 自检→T4；§3.7 JSON/缓存→T2/T4；§3.8 build/sync→T4；§4.1 policy max_step_dim=2→T5/T7；§4.2 env 展开→T6；§4.3 warmstart/停用→T7；§4.4 flag/preset/ckpt→T7；§5 不变→各 Task 只加分支；§6 substage 休眠→T7 互斥；§8 验证阶梯→T4(F0)/T8(F1)；§9 风险→T4 门槛检查/T8 兜底。**无遗漏**。
- **Placeholder 扫描**：已修——Task 2 的 `expand` 给了完整函数体（无占位）。其余步骤均有具体代码/命令；T3 的 `installed_noise_points` 标注「实现前核对 `blb_verify_noise_install.py` 口径并单测对拍」是验证要求而非占位。
- **类型一致性**：`FusionOption.action_indices`（整块 slot 向量含 K 占位）在 T2/T3/T5/T6 一致；`expand(gk, option_id, k_index)` 签名 T2 定义、T5/T6 调用一致；`max_step_dim=2`/`max_num_levels` 在 T5/T7 一致；ckpt variant 字符串单一来源 T7。
