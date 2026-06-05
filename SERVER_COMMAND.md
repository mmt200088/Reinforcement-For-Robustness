# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command  (HOLD — 不要重建图，不要重跑 13.5h 枚举)

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."
# HOLD：7 张 fusion 图已建好并验证通过（commit 0122eb2，block4 跑了 12.7h）。
# 绝对不要再重建（--max-enum-combos 0 的全枚举 = 13.5h）。下一条真正的命令
# （post-anchor 坍塌修复后的 smoke）正在等设计决定。这个 active block 只做廉价
# 复核：重新跑 summary + soundness 审计，并跳过会触发上次 KeyError 的 _summary.json sidecar。
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_fusion_verify_${TS}"; mkdir -p "$OUT"
REF="experiments/server_command_runs/stage2_fusion_fullbuild_20260604_233648/maps_ref_committed"
python3 - "$REF" <<'PY' | tee "$OUT/map_verify.txt"
import json, glob, os, sys
ref = sys.argv[1]; ok = True
print("=== fusion maps (skip _*.json sidecars) + soundness superset audit ===")
for p in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    gk = os.path.basename(p)
    if gk.startswith("_"):            # <-- KeyError fix: _summary.json has no top-level "options"
        continue
    d = json.load(open(p)); opts = d["options"]; bm = d.get("build_meta", {})
    new = sorted({int(o["fusion_count"]) for o in opts})
    rp = os.path.join(ref, gk)
    old = sorted({int(o["fusion_count"]) for o in json.load(open(rp))["options"]}) if os.path.exists(rp) else []
    sup = set(old) <= set(new); ok = ok and sup
    print(f"  {gk:16s} new={new} old={old} superset={'OK' if sup else 'REGRESSION'} "
          f"#opt={len(opts)} valid={bm.get('valid_configs')} kept={bm.get('kept_configs')} wall={bm.get('wall_seconds')}")
print(f"superset_pass={ok}")
PY
echo "HOLD: maps verified; NOT rebuilding; post-anchor collapse-fix smoke pending decision." | tee "$OUT/HOLD.txt"
```

## metadata

- **状态（2026-06-05）**：fusion full-build 完成（`0122eb2`）。**7 张图全部 sound 验证通过**：
  `superset_pass=True`（无 fusion 丢失，多处变多——block1/block4 `[0]→[0,1]`，block5_n2/n4 `[0,1]→[0,1,2]`）；
  新档位约定正确（有效槽 `4/2 → 9`，非有效槽留 0，K 槽留 3）；option0==baseline 守住；OOM 修复生效
  （block4 4.53 亿 valid → kept 192，12.7h）。**图无需重建。**
- **上次 KeyError 根因**：包装脚本的 `map_summary`/`soundness_audit` heredoc 用 `glob *.json` 把
  builder 写的 `_summary.json` sidecar 也读了，而它没有顶层 `options` 键 → KeyError。修法 = 解析时
  `if name.startswith("_"): continue`（已写进上面的 active block；`FusionCountMap.load` 本来就跳过 `_*.json`，
  所以 RL 运行不受影响）。
- **新发现的真问题（待修，不是图的问题）**：600-ep fusion smoke 能跑（Fusion ENABLED、600/600、invalid 0%、
  四卡 K=4 probe），但 **anchor（前 80 ep）一释放，策略立刻冲向激进融合**（mean fusion_count 0→14），精度坍塌
  （m1 0.86→0.32），之后 **511/600 ep 卡在 P1(acc) / loss_mean=100 / reward≈-5，无恢复**。根因：fusion 模式
  **关掉了 safe-neighbor/radius 课程**，47 个 block 自由采样 → 每个 block 即使 ~86% 选 option0，47 个一复合就有
  ~6-10 个 block 被融合 → 坍塌；warmstart prior 单靠它压不住 47 个独立选择。**这会让 60k 长跑同样坍塌。**
- **下一步（等用户拍板）**：给 fusion 模式加回一个 block 粒度的 safe-neighbor 课程（每个 episode 只允许少数
  block 偏离 baseline option0，半径缓慢增长），让多数 episode 贴着 baseline（P3）从而 PPO 拿到有区分度的梯度。
  定稿后再把新 smoke 命令写进这个 active block。
- **协议**：服务器只 `git pull`、运行、产出/回传 artifacts；源码改动都在本地。
