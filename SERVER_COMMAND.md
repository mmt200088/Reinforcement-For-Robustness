# 服务器端运行控制文件（SERVER_COMMAND.md）

> **协议**：服务器 agent 监听本文件 → 提取**第一个 ```bash 代码块** → 在仓库根目录 `bash` 执行。
> 本地这边改一次 + push，远端下次同步 / 触发就会按新命令跑。下方 metadata 段只是给人看的，agent 不解析。

## ▶ active command  (其余 5 个微调模型：build + 加大精度 fusion maps + 校验，60k 之前的全部内容；CPU-only，不碰正在跑的 60k)

> 把 `bert base mrpc` 已有的「60k 之前」全部内容跑到其余 5 个微调模型上：
> **rte / sst2（bert-base）+ mrpc_large / rte_large / sst2_large（bert-large，24 层）**。
> 借此实测「加大精度 / 找 baseline / fusion-count map」等适配工作能否真的泛化到其他数据集 / bert-large。
> commit `1b26e77` 已把 build + 加大精度泛化到任意 profile（block2 图键 profile 后缀解析、build/apply 按 profile
> 派生图键、apply 加 --num-layers）。本地已 torch-free 证实：6 个 profile 的 block2/block4/block5 链结构逐字一致、
> ReplanSession.from_profile 全部可加载、target 一致（46/53/48/43/43）、block2 boost 端到端到 (60,60)、stage1
> degree 全 ∈{1,2,4}。完整 build（需 torch 在位，但 CPU/replan、不动 GPU）放服务器。
> **三次服务器 build 失败已修（源码包必须包含到最新 commit，含本轮 block2 kept-option 修复；codex 手动上传，git 无权限）：**
> 1. `5aad064` — rte **block2** 的 `option 0 != baseline` 误报：3 个 rescale 槽(gamma/kt_mask1/qkt_matmul,anchor SF 28,
>    枚举 SF 15..28 都 ≥ 表下限 10)在 fusion=0 下**不注入噪声**(SF 只管模数链合法性)→ 所有档位装同样噪声 → 去重留了
>    字典序最小 idx1,而 baseline 用 idx14 → 装的噪声一致、原始索引不同 → 旧守卫按索引比较误报。修复:group_min_noise_options
>    拿 baseline 的 installed_signature,option0 与之结果等价就改写成 baseline 索引(真不同才报错)。
> 2. `8499ba6` — rte **block5_n1** 的 `fast-path mismatch`:fast 路的 golden-派生模板(per-slot 探针)漏了一个**依赖槽位组合**
>    的 rescale 安装点(golden 装 rescale@15,fast 没装)→ verify_template 抓到。修复:verify 失败时**回退到 golden 枚举**
>    (参考真值)重跑该 block-type,不再中止;`--fast-verify-random` 调到 512 提高抓取率。
> 3. **本轮** — rte/sst2 **block2** 的 `output_sf=43 != target 46`：fast 路把一个**真实 fusion=0** 的配置(fc=1 option 的 3 个
>    SF-无关 rescale 解码到 lex-min SF 15 — 低 sf_post 让链**不再融合**,real replan 实测 fc=0)**误标成 fusion 1**,成了 fc=1
>    的留存代表;非融合 base 无法被加大精度抬到输出上限(故 43≠46)。verify_template 的**随机**探针没撞上这个确定性的 kept
>    combo(block2 过了 512 探针)。本地已 torch-free 证实:正确 t_new(含 rescale)下 (15,15,15)=fc0、所有 39 个**真** fc=1
>    配置(含 lex-min (15,28,26))加大精度**全部到 46**;问题只在 fast 误分类。**block4(rescale 在 baseline idx14,与 mrpc 逐字一致)
>    与已 golden 的 block5_n* 不受影响。** 修复:build 在分组后对 **KEPT options** 做 golden 自洽复核(声称的 fusion_count/total_bits
>    必须被 real golden replan 复现),不符 → 对该 block-type **golden 全量兜底**(参考真值;只有出问题的 block2 付 golden 时间)。
>    (`blb_stage2_rl/fusion_enum.py::verify_kept_options_golden` + `scripts/blb_build_fusion_count_map.py` 接线;
>     单测 `tests/test_blb_fusion_kept_option_verify.py`。)
>
> 本轮（CPU-only、replan/单测、不碰 GPU / 正在跑的 60k）：
> 0) profile 无关代码门禁（torch 在位 → 必须 PASS 不能 SKIP）：多 profile topology 解析 + 加大精度 phase-1/2 +
>    fused-rescale + boost-handoff 测试；
> 1) 逐 profile：build fusion maps（fast 路 + 512 随机 golden 交叉校验 + **kept-option golden 自洽复核** → block2 等会自动
>    golden 兜底）→ apply 加大精度（phase-1+2）→ 校验 maps
>    （FusionCountMap.load / option0==baseline / boosted output_sf==target / ≤q_max=60 / ADR-019 >46 装噪点）→
>    运行时安装路径校验（Q1 装入 boosted 组、Q2 融合 rescale 置空）→ 合成 stage1 record（从 approx_per_dataset.json，
>    一个 stage2 绑定一个 stage1）；
> 2) commit + push 5 个 profile 的 maps + stage1 records + 证据。
> **不跑 60k、不跑 GPU 短训**（RL 训练循环是 profile 无关代码，mrpc 已验证；本轮只产出 + 校验各 profile 的专属产物）。
> 跑完每个 profile 即可像 bert base mrpc 一样启动各自的 60k（届时单独触发）。

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export CUDA_VISIBLE_DEVICES=""   # CPU-only: build/boost/verify are replan-based; do NOT touch the running 60k's GPUs
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_other5_profiles_${TS}"; mkdir -p "$OUT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git config --local http.version HTTP/1.1 || true
  git config --local protocol.version 0 || true
  git rev-parse HEAD > "$OUT/HEAD.txt"; cat "$OUT/HEAD.txt"; git log --oneline -3 | tee "$OUT/recent_commits.txt"
fi

echo "#################### [0] profile-independent code gate (torch present -> MUST run, not skip) ####################"
python3 -m unittest -v \
  tests.test_blb_precision_boost_multiprofile \
  tests.test_blb_precision_boost tests.test_blb_precision_boost_phase2 \
  tests.test_blb_fused_rescale_install \
  tests.test_blb_fusion_enum_baseline tests.test_blb_fusion_kept_option_verify \
  tests.test_blb_fusion_fixed_action tests.test_blb_final_eval_fusion_fixed_action tests.test_blb_glue_boost_install \
  2>&1 | tee "$OUT/t_code.txt"
grep -qE "^OK" "$OUT/t_code.txt" || { echo "[FATAL] code gate failed"; exit 1; }
grep -q "skipped" "$OUT/t_code.txt" && { echo "[FATAL] code gate SKIPPED tests (torch/rescale_optimizer not importable)"; exit 1; } || true

PROFILES="rte sst2 mrpc_large rte_large sst2_large"
fail=0
for pf in $PROFILES; do
  case "$pf" in *_large) NL=24;; *) NL=12;; esac
  MAPS="blb_stage2_rl/fusion_maps/$pf"
  pout="$OUT/$pf"; mkdir -p "$pout"
  echo "######################## [$pf] (num_layers=$NL) ########################"

  echo "---- [$pf] 1. build fusion maps ----"
  # --fast-verify-random 512: more golden-vs-fast probes -> higher chance to catch a
  # fast-template miss BEFORE the full enum (the template can miss a combination-
  # dependent installed point). On a caught mismatch the build now FALLS BACK to the
  # golden cfg-path enum for that block-type (commit 8499ba6) instead of aborting.
  python3 scripts/blb_build_fusion_count_map.py --profile "$pf" --out-dir "$MAPS" \
    --rescale-optimizer-root Rescale_optimizer --num-layers "$NL" \
    --fast-verify-random 512 2>&1 | tee "$pout/build.txt"
  miss=0
  for gk in "block1_$pf" "block2_$pf" block4 block5_n1 block5_n2 block5_n4; do
    [ -f "$MAPS/$gk.json" ] || { echo "[FATAL][$pf] build missing $gk.json"; miss=1; }
  done
  [ "$miss" = 0 ] || { fail=1; continue; }

  echo "---- [$pf] 2. apply precision boost (phase-1 + phase-2) ----"
  python3 scripts/blb_apply_precision_boost.py --profile "$pf" --maps-dir "$MAPS" --num-layers "$NL" 2>&1 | tee "$pout/boost.txt"
  grep -q "precision boost applied" "$pout/boost.txt" || { echo "[FATAL][$pf] boost did not finish"; fail=1; continue; }

  echo "---- [$pf] 3. verify maps (load / option0==baseline / output_sf==target / <=q_max / ADR-019) ----"
  PROFILE="$pf" python3 - <<'PY' 2>&1 | tee "$pout/verify.txt"
import json, os, pathlib, sys
sys.path[:0] = [".", "blb_stage2_rl", "Rescale_optimizer"]
import precision_boost as pb
from fusion_count_map import FusionCountMap
pf = os.environ["PROFILE"]; RO = "Rescale_optimizer"
TARGETS = {f"block2_{pf}": 46, "block4": 53, "block5_n1": 48, "block5_n2": 43, "block5_n4": 43}
mdir = pathlib.Path(f"blb_stage2_rl/fusion_maps/{pf}")
bad = 0
FusionCountMap.load(pf)
print(f"[ok] FusionCountMap.load('{pf}') accepted all maps (option0==baseline)")
over46 = 0
for gk, want in TARGETS.items():
    p = mdir / f"{gk}.json"
    if not p.exists():
        print(f"[BAD] {gk}: map file missing"); bad += 1; continue
    payload = json.loads(p.read_text())
    topo = pb.topology_for_graph_key(gk)
    if topo is None:
        print(f"[BAD] {gk}: no topology resolved"); bad += 1; continue
    tgt = pb.effective_output_target(topo, pb.target_output_sf(gk, profile=pf, root=RO), int(topo.q_max))
    if tgt != want:
        print(f"[BAD] {gk}: effective target {tgt} != expected {want}"); bad += 1
    for o in payload["options"]:
        fc = int(o.get("fusion_count", 0))
        if fc == 0:
            if o.get("boosted"):
                print(f"[BAD] {gk} option0 (baseline) must NOT be boosted"); bad += 1
            continue
        if not o.get("boosted") or not o.get("explicit_field_values"):
            print(f"[BAD] {gk} fc={fc} not boosted"); bad += 1; continue
        if int(o.get("output_sf", -1)) != tgt:
            print(f"[BAD] {gk} fc={fc} output_sf={o.get('output_sf')} != target {tgt}"); bad += 1
        fv = o["explicit_field_values"]
        over = [(n.cfg_field, fv[n.cfg_field]) for n in topo.nodes
                if n.cfg_field and n.kind in ("fresh", "encode", "rescale")
                and n.cfg_field in fv and int(fv[n.cfg_field]) > int(topo.q_max)]
        if over:
            print(f"[BAD] {gk} fc={fc} installed SF over q_max: {over}"); bad += 1
        over46 += sum(1 for n in topo.nodes
                      if n.cfg_field and n.kind in ("fresh", "encode", "rescale")
                      and n.cfg_field in fv and 46 < int(fv[n.cfg_field]) <= int(topo.q_max))
        print(f"[OK] {gk} fc={fc} output_sf={o['output_sf']} ({o.get('boost_description', '')})")
if over46 < 1:
    print("[BAD] ADR-019 NOT realized: no boosted install point in (46, q_max]"); bad += 1
else:
    print(f"[ok] ADR-019 confirmed: {over46} boosted install point(s) in (46, q_max]")
print("VERIFY_OK" if bad == 0 else f"VERIFY_FAIL ({bad} problems)")
sys.exit(0 if bad == 0 else 1)
PY
  grep -q "VERIFY_OK" "$pout/verify.txt" || { echo "[FATAL][$pf] map verification failed"; fail=1; continue; }

  echo "---- [$pf] 4. runtime install-path verify (Q1 boosted-install / Q2 fused-rescale) ----"
  python3 scripts/blb_verify_boosted_install.py --profile "$pf" \
    --rescale-optimizer-root Rescale_optimizer --maps-dir "$MAPS" --num-layers "$NL" 2>&1 | tee "$pout/verify_install.txt"
  grep -q "VERIFY_OK" "$pout/verify_install.txt" || { echo "[FATAL][$pf] runtime install verify failed"; fail=1; continue; }

  echo "---- [$pf] 5. synth Stage-1 record (one stage2 binds one stage1) ----"
  PROFILE="$pf" python3 - <<'PY' 2>&1 | tee "$pout/stage1.txt"
import json, os, glob, datetime
pf = os.environ["PROFILE"]
model = "bert large" if pf.endswith("_large") else "bert base"
dataset = pf[:-6] if pf.endswith("_large") else pf
combo = f"{model} {dataset}"
ap = json.load(open("Model_analysis/configs/approx_per_dataset.json"))
s1 = ap[pf]["stage1"]; gelu = [int(x) for x in s1["gelu"]]; softmax = [int(x) for x in s1["softmax"]]
assert 0 not in gelu, f"degree-0 not allowed: {gelu}"
rec_root = "Parting Chapter/stage1/record"
existing = [d for d in glob.glob(os.path.join(rec_root, combo + " *")) if os.path.isdir(d)]
if existing:
    print("[skip] stage1 record exists:", [os.path.basename(d) for d in existing]); raise SystemExit(0)
date = datetime.datetime.now().strftime("%Y%m%d")
rec_dir = os.path.join(rec_root, f"{combo} 1 {date}")
os.makedirs(rec_dir, exist_ok=True)
json.dump({"gelu_degree_per_layer": gelu, "softmax_degree_per_layer": softmax,
           "_synthesized_from": "Model_analysis/configs/approx_per_dataset.json"},
          open(os.path.join(rec_dir, "final_config.json"), "w"), ensure_ascii=False, indent=2)
print("[ok] synth stage1 record:", rec_dir, "| gelu =", gelu)
PY
  echo "[$pf] DONE (built + boosted + verified)"
done

echo "#################### [2] commit + push maps + stage1 records ####################"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  for pf in $PROFILES; do git add "blb_stage2_rl/fusion_maps/$pf" 2>/dev/null || true; done
  git add "Parting Chapter/stage1/record" "$OUT" 2>/dev/null || true
  git commit -m "Build + precision-boost fusion maps for the other 5 fine-tuned profiles" || echo "[note] nothing to commit"
  git push origin HEAD 2>&1 | tee "$OUT/push.txt" || echo "[note] push failed; commit is local on the server, push manually"
fi

if [ "$fail" = 0 ]; then
  echo "[DONE] all 5 profiles built + boosted + verified; OUT=$OUT (the running 60k was NOT touched)."
  echo "[NEXT] each profile is now ready for its own 60k (boosted maps + stage1 record); launch per-profile separately."
else
  echo "[FAIL] one or more profiles failed; see $OUT/<profile>/*.txt"; exit 1
fi
```

## ✅ done — boost handoff 校验门禁（已折进上面新门禁的 [0]；下方 ```bash 块已过时，仅留存，服务器只跑最上面那个块）

> 校验 commit `187db50d` 的修复。此前 boost **只在 RL 训练期**装上（env `_boosted_overrides` SF-direct 重建）；
> 持久化的 best 是扁平网格**索引**向量，携带不了 boost（boosted SF 高于网格、只存在 option 的
> `explicit_field_values`），导致**验证集 final eval** 与 **GLUE 提交**都装 **pre-boost**（更吵）噪声 ——
> 被评估/提交的 ≠ RL 选中的。修复后三处一致（fusion→模数链融合→二阶段 boost→正确槽位装噪，训练/验证/GLUE 同一套）。
>
> 本门禁（CPU、replan/单测 only、不碰 GPU / 正在跑的 60k）：
> 1) boost-handoff torch-gated 测试（torch 在位 → **必须 PASS 且不能 SKIP**）：
>    `test_blb_fusion_fixed_action`（匹配器 + 元数据解析）、`test_blb_final_eval_fusion_fixed_action`（final eval 还原 boost）、
>    `test_blb_glue_boost_install`（GLUE 解码装入的 boosted block2 SF 和 > pre-boost）；
> 2) 在真实 committed best 上跑 `blb_make_fusion_fixed_action_config.py` 回填 fusion-fixed 配置，并校验可被
>    `FusionCountMap.load` 加载、`option_by_step` 覆盖全部步（best-effort：旧 best 若 stage1 不匹配只告警，[1] 才是权威门）；
> 3) commit/push 回传证据 + 回填的 fusion-fixed JSON。
>
> 注：**不启动 60k**。当前正在跑的 60k 建立在旧代码/旧图上（早于 fused-rescale 修复 / phase-2 boost / ADR-019 / 本 handoff 修复），
> 其结果在旧（有缺陷）配置上；要拿正确结果需在最新 commit 上**重启**一轮 60k（会自动持久化 group → final eval/GLUE 自动用 boosted）。
> 那是单独的、更重的触发（会杀掉正在跑的旧 60k），确认后再设。

```bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export CUDA_VISIBLE_DEVICES=""   # CPU-only: do NOT touch the GPUs the running 60k uses
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_boost_handoff_gate_${TS}"; mkdir -p "$OUT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git config --local http.version HTTP/1.1 || true
  git config --local protocol.version 0 || true
  git rev-parse HEAD > "$OUT/HEAD.txt"; cat "$OUT/HEAD.txt"; git log --oneline -3 | tee "$OUT/recent_commits.txt"
fi

echo "#################### [1] boost-handoff tests (torch present -> MUST run, not skip) ####################"
python3 -m unittest tests.test_blb_fusion_fixed_action tests.test_blb_final_eval_fusion_fixed_action tests.test_blb_glue_boost_install -v 2>&1 | tee "$OUT/t_handoff.txt"
grep -qE "^OK" "$OUT/t_handoff.txt" || { echo "[FATAL] handoff tests failed"; exit 1; }
grep -q "test_decode_helper_installs_boosted_block2_output" "$OUT/t_handoff.txt" || { echo "[FATAL] glue boost-install test name not seen in -v output"; exit 1; }
grep -q "skipped" "$OUT/t_handoff.txt" && { echo "[FATAL] handoff tests SKIPPED (torch/rescale_optimizer not importable -> boost replay NOT validated)"; exit 1; } || true

echo "#################### [2] reconstruct + backfill fusion-fixed config from a committed best (real artifact; best-effort) ####################"
BEST="Parting Chapter/stage2/bert base mrpc/progress/diagnostics/best_action_vec.json"
if [ -f "$BEST" ]; then
  GELU=$(python3 -c "import json;a=json.load(open('Model_analysis/configs/approx_per_dataset.json'));print(json.dumps([int(x) for x in a['mrpc']['stage1']['gelu']]))")
  SOFTMAX="[6,6,6,6,6,6,6,6,6,6,6,6]"
  echo "[2] gelu=$GELU softmax=$SOFTMAX"
  if python3 scripts/blb_make_fusion_fixed_action_config.py \
       --input "$BEST" --output "$OUT/best_action_fusion_fixed.json" \
       --profile mrpc --num-layers 12 --gelu "$GELU" --softmax "$SOFTMAX" 2>&1 | tee "$OUT/reconstruct.txt"; then
    OUTDIR="$OUT" python3 - <<'PY' 2>&1 | tee "$OUT/reconstruct_verify.txt" || echo "[2][warn] verify raised (non-fatal)"
import json, os, sys
sys.path[:0] = [".", "blb_stage2_rl", "Rescale_optimizer"]
from fusion_count_map import FusionCountMap
from action_space import step_schedule
cfg = json.load(open(os.path.join(os.environ["OUTDIR"], "best_action_fusion_fixed.json")))
assert cfg.get("schema_version") == "fusion_count_fixed_action_v1", "bad schema"
FusionCountMap.load("mrpc")
obs = cfg["group"]["option_by_step"]
sch = step_schedule(12, profile="mrpc",
                    attn_degree_per_layer=[int(x) for x in cfg["attn_degree"]],
                    gelu_degree_per_layer=[int(x) for x in cfg["gelu_degree"]])
missing = [s.step_idx for s in sch if str(s.step_idx) not in obs]
assert not missing, f"option_by_step missing steps: {missing}"
print(f"[ok] option_by_step covers all {len(sch)} fusion steps")
print(f"[info] total_fusion_count={cfg['summary']['total_fusion_count']} boosted_options={cfg['summary']['boosted_option_count']}")
print("RECONSTRUCT_OK")
PY
  else
    echo "[2][warn] reconstruction failed (likely stage1 mismatch for this old best); [1] already validated the code path"
  fi
else
  echo "[2][skip] committed best not found at: $BEST (fresh checkout); [1] already validated the code path"
fi

echo "#################### [3] commit + push proof ####################"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add "$OUT" 2>/dev/null || true
  git commit -m "Boost-handoff gate: validate RL-selected boosted config reaches final eval + GLUE" || echo "[note] nothing to commit"
  git push origin HEAD 2>&1 | tee "$OUT/push.txt" || echo "[note] push failed; commit is local on the server, push manually"
fi
echo "[DONE] boost-handoff validated; OUT=$OUT (the running 60k was NOT touched)."
echo "[NEXT] the corrected 60k (boosted maps + all fixes + auto-persisted group) is a separate heavier trigger that will KILL the stale running 60k; confirm before launching."
```

## ✅ done — 二阶段加大精度 phase-2 (ADR-019 **重新 apply**：已执行并 push；下方 ```bash 块已过时，仅留存供追溯，服务器只跑最上面那个块)

> **本轮是重新 apply（ADR-019）**：上一轮 `b0c18e3f` 已用**旧 ≤46 install cap** 把 phase-2 apply 进了
> committed maps（block4 1/d=21、block5_n1=46）。ADR-019 把那条 cap 打开了（SF>46 噪声可忽略 → 当 0 不装，
> 装噪点可到 q_max=60），所以那批 maps 现在**过期**，要用开放的 cap 重新生成。apply 是幂等的
> （`boost_options_for_block` 每次从 `action_indices` 网格基重新推导，不读旧 `explicit_field_values`），所以直接
> 重跑即覆盖为新结果。
>
> 二阶段「加大精度」：一阶段把中间短素数顶到 q_max；二阶段再把**最后一个节点的输出 SF**
> （= 最后一个 rescale 的 sf_post + 末尾 encode 的 SF）顶到上限
> `target = q_tail_bits - amplitude_budgets[-1] - h_sf`（从 `Rescale_optimizer/configs/<profile>/<graph_key>.json`
> 读，不写死）。提升量在「末尾 encode」与「最后 rescale sf_post」之间分配（末尾 encode 可降到硬下限 15），
> sf_post 上抬所需的前置尺度按一阶段方式分发到上游、最后素数尽量保持高位；**装噪点可到 q_max=60**
> （ADR-019: SF>46 噪声可忽略 → 当 0 不装；只有 >60 才是模数违例）；replan 校验后取**噪声最小**的组合。
> block2 43->46 / block4 51->53（**1/d 降到 15**、ln_mean_rescale 49 不装噪、总噪声更低）/ block5_n1 31->**48** / block5_n2,n4 31->43。
>
> 本轮 active（CPU、replan-only、不做 model forward、不碰 GPU / 正在跑的 60k）：
> 1) fused-rescale 回归（apply 依赖它把被融合 rescale 排除在装噪点外）；
> 2) phase-1+phase-2 precision-boost 回归（block2/4/5_n2/5_n4 真 replan）；
> 3) 把 phase-1+phase-2 应用到 committed fusion maps（boost_options_for_block 现含两阶段），
>    硬 guard 校验 output==target / 上游素数不变 / fusion 不变 / 全部 ≤q_max(60)；
> 4) 校验 maps 内容：可被 FusionCountMap.load 加载、option0==baseline、boosted output_sf==target、全部 ≤q_max(60)；
> 5) 校验**运行时安装路径**（真实 maps，驱动 BLBStage2Env.step 的 evaluate_action_for_cost(boosted)
>    → apply_optimizer_output_to_cfg + sync_block*）：Q1 送入模型的是 phase-2 boosted 组（装入精度高于网格解码）、
>    Q2 该组 rescale 被模数链融合置空（rescale_fused_away）—— 这正是用户要确保的两点；
> 6) git commit + push 回传更新后的 maps + 该校验脚本。
> 注：更新 maps 不会让正在跑的 60k 崩（它启动时已把 maps 读进内存）；要让 60k 用上 phase-2，需重启。

```bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export CUDA_VISIBLE_DEVICES=""   # CPU-only: do NOT touch the GPUs the running 60k uses
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_precision_boost_phase2_${TS}"; mkdir -p "$OUT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git config --local http.version HTTP/1.1 || true
  git config --local protocol.version 0 || true
  git rev-parse HEAD > "$OUT/HEAD.txt"; cat "$OUT/HEAD.txt"; git log --oneline -3 | tee "$OUT/recent_commits.txt"
fi

echo "#################### [1] fused-rescale install regression (apply depends on it) ####################"
python3 -m unittest tests.test_blb_fused_rescale_install -v 2>&1 | tee "$OUT/t_fused.txt"
grep -qE "^OK" "$OUT/t_fused.txt" || { echo "[FATAL] fused-rescale regression failed"; exit 1; }
grep -q "skipped" "$OUT/t_fused.txt" && { echo "[FATAL] fused-rescale tests SKIPPED (torch/rescale_optimizer not importable)"; exit 1; } || true

echo "#################### [2] precision-boost phase-1 + phase-2 regression (REAL replan) ####################"
python3 -m unittest tests.test_blb_precision_boost tests.test_blb_precision_boost_phase2 -v 2>&1 | tee "$OUT/t_boost.txt"
grep -qE "^OK" "$OUT/t_boost.txt" || { echo "[FATAL] precision-boost regression failed"; exit 1; }
grep -q "skipped" "$OUT/t_boost.txt" && { echo "[FATAL] precision-boost tests SKIPPED (torch/rescale_optimizer not importable on server)"; exit 1; } || true

echo "#################### [3] dry-run apply (phase-1 + phase-2) ####################"
python3 scripts/blb_apply_precision_boost.py --dry-run 2>&1 | tee "$OUT/apply_dryrun.txt"

echo "#################### [4] REAL apply to committed maps (hard guard inside) ####################"
python3 scripts/blb_apply_precision_boost.py 2>&1 | tee "$OUT/apply.txt"
grep -q "precision boost applied" "$OUT/apply.txt" || { echo "[FATAL] apply did not finish"; exit 1; }

echo "#################### [5] verify maps load + option0==baseline + boosted output_sf==target ####################"
python3 - <<'PY' 2>&1 | tee "$OUT/verify.txt"
import json, pathlib, sys
sys.path[:0] = [".", "blb_stage2_rl", "Rescale_optimizer"]
import precision_boost as pb
from fusion_count_map import FusionCountMap
RO = "Rescale_optimizer"
# achieved output: since the ADR (SF>46 = no noise), install points run up to q_max=60,
# so block5_n1 reaches its full config ceiling 48 (no longer clamped to 46). block4's
# decrease route pushes ln_mean_rescale to 49 (>46 = no noise).
TARGETS = {"block2_mrpc": 46, "block4": 53, "block5_n1": 48, "block5_n2": 43, "block5_n4": 43}
mdir = pathlib.Path("blb_stage2_rl/fusion_maps/mrpc")
bad = 0
# Runtime loader must accept the whole profile (from_payload also validates
# option0==baseline). load() takes a PROFILE NAME, not a per-file path — calling
# it per-file with str(p) raised FileNotFoundError and aborted the gate before
# commit/push.
FusionCountMap.load("mrpc")
print("[ok] FusionCountMap.load('mrpc') accepted all maps (option0==baseline)")
over46_total = 0  # ADR-019 proof: the opened cap must realize >=1 boosted point in (46, q_max]
for gk, want in TARGETS.items():
    p = mdir / f"{gk}.json"
    payload = json.loads(p.read_text())
    topo = pb.TOPOLOGIES[gk]
    tgt = pb.effective_output_target(topo, pb.target_output_sf(gk, profile="mrpc", root=RO))
    assert tgt == want, f"{gk}: effective target {tgt} != expected {want}"
    for o in payload["options"]:
        fc = int(o.get("fusion_count", 0))
        if fc == 0:
            if o.get("boosted"):
                print(f"[BAD] {gk} option0 (baseline) must NOT be boosted"); bad += 1
            continue
        if not o.get("boosted") or not o.get("explicit_field_values"):
            print(f"[BAD] {gk} fc={fc} not boosted"); bad += 1; continue
        if int(o.get("output_sf", -1)) != tgt:
            print(f"[BAD] {gk} fc={fc} output_sf={o.get('output_sf')} != target {tgt}"); bad += 1
        fv = o["explicit_field_values"]
        # install points may run up to q_max=60 now (>46 = no noise); only >60 is a
        # modulus violation. topo.q_max is the per-block ceiling.
        over = [(n.cfg_field, fv[n.cfg_field]) for n in topo.nodes
                if n.cfg_field and n.kind in ("fresh","encode","rescale")
                and n.cfg_field in fv and int(fv[n.cfg_field]) > int(topo.q_max)]
        if over:
            print(f"[BAD] {gk} fc={fc} installed SF over q_max: {over}"); bad += 1
        over46_total += sum(1 for n in topo.nodes
                            if n.cfg_field and n.kind in ("fresh", "encode", "rescale")
                            and n.cfg_field in fv and 46 < int(fv[n.cfg_field]) <= int(topo.q_max))
        print(f"[OK] {gk} fc={fc} output_sf={o['output_sf']} ({o.get('boost_description','')})")
# ADR-019 took effect proof: if NO boosted point installs above 46, the maps are the
# stale <=46 phase-2 (re-apply did not pick up the opened cap) -> fail loudly.
if over46_total < 1:
    print("[BAD] ADR-019 NOT realized: no boosted install point in (46, q_max] — maps look "
          "like the stale <=46 phase-2 (e.g. block4 1/d still 21); re-apply did not take effect")
    bad += 1
else:
    print(f"[ok] ADR-019 confirmed: {over46_total} boosted install point(s) in (46, q_max] (e.g. block4 ln_mean_rescale=49)")
print("VERIFY_OK" if bad == 0 else f"VERIFY_FAIL ({bad} problems)")
sys.exit(0 if bad == 0 else 1)
PY
grep -q "VERIFY_OK" "$OUT/verify.txt" || { echo "[FATAL] map verification failed"; exit 1; }

echo "#################### [5b] verify RUNTIME install path on the REAL maps: model gets the boosted group (Q1) + its rescale is fused-away (Q2) ####################"
# step [5] checks MAP CONTENTS (output_sf==target, <=46). This drives the exact
# BLBStage2Env.step path (evaluate_action_for_cost(boosted) -> apply_optimizer_output_to_cfg
# + sync_block*) on each committed boosted option and asserts: Q1 the installed cfg
# carries the boosted (phase-2, higher-precision) group not the in-grid decode; Q2
# a rescale is nulled (rescale_fused_away) for every fusing option.
python3 scripts/blb_verify_boosted_install.py \
  --profile mrpc --rescale-optimizer-root Rescale_optimizer \
  --maps-dir blb_stage2_rl/fusion_maps/mrpc 2>&1 | tee "$OUT/verify_install.txt"
grep -q "VERIFY_OK" "$OUT/verify_install.txt" || { echo "[FATAL] runtime install-path verification failed (Q1 boosted-install / Q2 fused-rescale)"; exit 1; }

echo "#################### [6] commit + push updated maps ####################"
git add blb_stage2_rl/fusion_maps/mrpc/*.json scripts/blb_verify_boosted_install.py
git commit -m "Apply phase-2 precision boost to fusion maps (output SF -> ceiling)" || echo "[note] nothing to commit"
git push origin HEAD 2>&1 | tee "$OUT/push.txt" || echo "[note] push failed; maps are committed locally on the server, push manually"

echo "[DONE] phase-2 precision boost applied + verified; OUT=$OUT (the running 60k was NOT touched)."
echo "[NOTE] phase-2 raises FUSION options' output SF -> changes installed noise for fusion options."
echo "       The in-flight 60k keeps its OLD maps in memory; RESTART it to pick up phase-2."
```

## ⏸ on-deck — KV-cache 端到端吞吐 A/B（**已解决/跳过** — 前向基准已判 NOT EFFECTIVE）

> **跳过理由**：KV-cache 的前向基准（上轮 [A1]）已实测 ON 比 OFF **慢**（0.60x，OFF 298ms vs ON 495ms/
> episode）。端到端 A/B 只会再确认这层"真模型探针在环时仍更慢"，无收益、白耗 GPU 工时。
> KV-cache 保持默认 OFF，**不进 60k**。下方脚本仅留作参考；若未来想重启 KV-cache 路（如换更长 horizon /
> CUDA graphs 摊薄 launch 开销）再人工移到第一个 ```bash 块。
>
> （原说明）真实 fusion 训练短跑（会 `--fresh` 各自的 tagged persistent 工作目录，不碰正式 60k canonical slug），同 seed 跑 OFF 与 ON 各 ~1500 episode，
> 比 episodes.jsonl 的 reward/优先级/fusion 分布（质量应一致，路径浮点等价非逐位）+ per-episode rollout 墙钟。

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data
TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_kvcache_ab_${TS}"; mkdir -p "$OUT"
EPISODES_AB=1500
KTRIALS=5; ANCHOR_EPISODES=80; FUSION_PROBE_INTERVAL=200
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; [ -z "$NGPU" ] && NGPU=1
DEVS="$(seq -s, 0 $((NGPU-1)))"
CANON_STAGE2_GROUP="Parting Chapter/persistent/rl/bert-base/mrpc"
CANON_STAGE2="${CANON_STAGE2_GROUP}/s1t0.001_s2t0.001_s2st3.0"

run_ab () {   # $1 = tag (off/on), $2 = kv flag (0/1)
  local tag="$1"; local kv="$2"; local run_tag="ab_${tag}_${TS}"
  local rundir="${CANON_STAGE2_GROUP}/s1t0.001_s2t0.001_s2st3.0__${run_tag}"
  echo "==================== [A/B] $tag : --blb-v3-kv-cache-rollout $kv ===================="
  CUDA_VISIBLE_DEVICES=$DEVS bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --run-tag "ab_${tag}_${TS}" \
    --blb-v3-fusion-count-action 1 \
    --blb-v3-fusion-neighbor-curriculum 1 \
    --stage2-search-episodes "$EPISODES_AB" \
    --stage2-k-trials "$KTRIALS" --stage2-probe-size 256 --batch-size 512 \
    --stage2-rl-devices "$DEVS" \
    --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
    --stage2-fixed-config-source json --stage2-fixed-config glue_final_configs_best_ppo.json \
    --stage2-stability-tolerance 3.0 --stage2-limit-tolerance 0.001 \
    --blb-v3-fusion-probe-interval "$FUSION_PROBE_INTERVAL" \
    --blb-v3-fusion-exploration-epsilon 0.05 \
    --stage2-workers-per-device 1 \
    --blb-v3-kv-cache-rollout "$kv" \
    --fresh 2>&1 | tee "$OUT/${tag}_launch.log"
  sleep 12
  local pid; pid="$(cat "${rundir}/run.pid" 2>/dev/null || cat "${rundir}/rl.pid" 2>/dev/null || true)"
  echo "[A/B] $tag pid=$pid — waiting for completion…"
  if [ -n "$pid" ]; then while kill -0 "$pid" 2>/dev/null; do sleep 30; done; fi
  local ep; ep="$(ls "$rundir"/stage2_noise/progress/diagnostics/episodes.jsonl 2>/dev/null | tail -1)"
  [ -n "$ep" ] || { echo "[A/B][FATAL] $tag episodes.jsonl missing under persistent dir: $rundir"; return 1; }
  cp "$ep" "$OUT/${tag}_episodes.jsonl"
  echo "[A/B] $tag episodes.jsonl -> $OUT/${tag}_episodes.jsonl ($(wc -l < "$OUT/${tag}_episodes.jsonl") lines)"
  python3 scripts/verify_stage2_persistent_outputs.py \
    --run-dir "$rundir" \
    --min-episodes "$EPISODES_AB" \
    --min-ppo-updates 1 \
    --require-png 2>&1 | tee "$OUT/${tag}_persistent_verify.txt"
}

run_ab off 0
run_ab on  1
echo "==================== [A/B] 对比 ===================="
python3 scripts/blb_kvcache_ab_compare.py \
  --off "$OUT/off_episodes.jsonl" --on "$OUT/on_episodes.jsonl" 2>&1 | tee "$OUT/ab_verdict.txt"
echo "证据目录：$OUT （off/on episodes + ab_verdict.txt）。请 git add/commit/push 回传。"
echo "结论看 ab_verdict.txt：质量须 MATCHED；speedup 给出端到端 rollout 墙钟加速。"
```

## ⏸ on-deck — ADR-016 reward 60k（KV-cache 验证完，把下面这个 ```bash 块移回上面成为第一个 ```bash 块）  (ADR-015 连续有界 reward（移植 Stage-1）+ 严格稳定性刹车 + Stage-1 cosine 探索 + 严格可行性选择 → 门禁 → 60k)

```bash
set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

# ============================================================================
# 本轮（2026-06-14，ADR-014）：第 4 次 60k（artifacts stage2_grid_gate_60k_20260613_175503，
# ADR-013 log-barrier 全开，commit 4e3aec0）**依然热崩溃**——watchdog 在 40320/60000 杀停
# （P3<2% 连续 12 窗）。fusion 单调 1→10(健康)→13(P1 飙)→35(全 P1 冻结)，从不稳定；熵未归零
# = 过度融合塑形，非探索死亡。根因（实证）：最佳点 terminal_metric1_std≈0.0155 > 可行余量
# (baseline 0.871−阈值 0.858 = 0.013)，是 ADR-013 校准 MARGIN_REF=0.25 所用 0.0018 基线 σ 的
# ~8.6× → barrier headroom 被噪声淹没、形不成可测吸引子，而 LINEAR fusion 成本是确定性单调激励 →
# 压倒 barrier → 失控。+ ADR-013 的 barrier/margin（mu、acc_barrier_*）算了却从未落进 episodes.jsonl
# = 黑箱 = 用户「只能靠最终结果猜」。修复（用户选：结构性反失控 + barrier 调参；**保持成本权重 +
# 探针 size/K 不变** → 必须在现有噪声下生效；详见 docs/adr/ADR-014-*.md）：
#   ① fusion 成本饱和化（fusion_cost.saturate_fusion，FUSION_SATURATION_TAU=0.15）：边际 fusion 奖励
#     在拐点 ~fusion8（低于噪声边界 10-13）后→~0；陡起(不冷崩)+ 平尾(不热崩)；tau≤0=恒等=ADR-013。
#     env 用 fusion_norm_saturated 缩放预算；raw 保留作诊断。
#   ② MARGIN_REF 0.25→0.5。①+② ⇒ cost(fusion)+barrier 在中等正余量出现内点峰值（max fusion 不最优）。
#     稳定最优会比刀刃 27 少（留可分辨 headroom，是稳定的 WIN 而非退步）。旋钮：FUSION_SATURATION_TAU /
#     acc_barrier_margin_ref。
#   ③ 崩溃调试落盘：mu/acc_barrier_sat/vio/near_miss/margin_m1/m2/fusion_norm_raw/saturated +
#     fusion_count_b2/b4/b5 入 episodes.jsonl；blb_stage2_health.log（repo 内 rolling600 P1/P2/P3+
#     fusion+per-block+margin，把原服务器 bash 轨迹搬进 repo）；blb_stage2_diagnostics_curve.png；
#     attribute_collapse(HOT/COLD+起点) 入 search_log；离线再生器复盘。
#   ④ 不变量：饱和/barrier 只改 PPO 标量 + 纯确定函数 → priority/rank/选择逐位不变 + item7 + 1==N。
#   reward 跨 ADR-014 不可比。判 60k 成功：曲线不崩 + fusion 稳在中等正余量(非 0 非 35) + P3 持续>0 +
#   best ≥ 无融合上限——**现在可直接从 health 日志 + 诊断曲线 + 落盘的 mu 读出（不用猜）**。
# ----------------------------------------------------------------------------
# 【历史】上轮（2026-06-13，ADR-013）：第 3 次 60k（artifacts stage2_grid_gate_60k_20260612_191530，
# ADR-012 全开）翻转成「热崩溃」——与前两次「冷崩溃 fusion=0」完全相反：
#   - fusion 单调爬升 1.4→35，metric1 单调跌 0.866→0.690（分箱：ep0 fus1.4/P3 → ep12k
#     fus11.7/P1 1240 → ep21k fus19.8/P1 2694 → ep30k fus31.8/P1 3000 reward-6.94 →
#     ep33k+ fus35.1/m0.690 冻结平在 -6.95 直到 ep60000）；后 30k 回合(50% 的跑)纯浪费；
#   - 但 hard-priority 选择仍救回 ep20880 = fusion 22 / P3 / reward 40.8 > 无融合上限 39.5
#     （余量仅 0.0003 = 刀刃、亚 σ）；搜索能找到好配置，只是不在最优点稳定。
#   - 根因：ADR-012 的近界渐变档把"越界"几乎变免费(-7→15-35)，叠加单调 fusion 成本激励
#     (block4 @130 = 最毁精度却付第二高) + ~1.3% 精度预算被 fusion 和深 K 共同吃掉 → 策略
#     沿 fusion 轴一路上滑、无回正力，最后掉进平坦无梯度盆地(全 P1、reward 平、ε 太弱爬不回)。
# 本轮修复（ADR-013；用户选 log-barrier 方向 + 保持成本权重 80:150:130:40 → barrier 独自
# 对抗 block4 的 130；详见 docs/adr/ADR-013-*.md；图无需重建——解码/档位未变）：
#   ① Stage-1 式两段 log-barrier (reward.accuracy_margin_barrier，移植自
#     layer_importance_evaluator.py:log_barrier_reward)。取代近界档(P1)+线性 P3 margin(P3)。
#     mu=最差通道带符号余量(|baseline−阈值| 单位)。满足侧 0≤mu<MARGIN_REF: SAT·(log(mu+eps)
#     −log(REF+eps)) ≤0，mu→0 斜率→∞ ⇒ cost(fusion)+barrier 在正余量处出现内点峰值(不冲过)；
#     mu≥REF ⇒ 0(成本说了算)。违反侧 mu<0: b0−VIO·(−mu) 连续+线性(整个崩溃深度无平台)
#     ⇒ 恢复梯度(3rd-60k 缺的就是这个)。下限 −10。
#   ② MARGIN_REF=0.25 (≈1.8 探针 σ) = 激进度旋钮(代码默认；过保守/过激可在
#     RewardWeights 调，sweep {0.15,0.25,0.35})；acc_barrier_enabled 开关(False=回 ADR-012)。
#   ③ 不变量：barrier 只改 PPO 标量 → priority/rank-key/选择逐位不变；违反侧 < P3 下限 40
#     且 P1 不吃 cost ⇒ item7；纯 metrics 函数 ⇒ 1==N；invalid 仍走 legacy invalid_term。
#   ④ 保留 ADR-012 的边缘复测(给 barrier 读的余量降噪，更重要了)、ε 下限、policy-K 探针。
#   ⑤ 新增 per-block-type fusion(b2/b4/b5) 入 episodes.jsonl + 健康日志；崩溃 watchdog(持续
#     P3≈0 → 杀进程；best 已周期 checkpoint，安全网，barrier 正常时永不触发)。
#   ⑥ 提速（2026-06-12，画像驱动，沿用）：上轮 60k 实测每 episode 墙钟 = 探针 2.69s
#     (78%) + rollout 0.74s (21%) + replan 0.009s；窗口负载不均仅 2.8%；PPO
#     更新 ≈1.4h/13.85h。最大安全收益 = 每卡 2 个 worker（episode 结果只依赖
#     全局序号→worker 指派无关，正是 1==N 的根基）：一个 worker 的 CPU 侧
#     rollout/簿记与同卡兄弟的 GPU 探针重叠。两个 RNG 原子单元加同卡锁：
#     (manual_seed→sample) 与 (reseed_noise→单个 trial forward)，交错顺序不
#     影响各 trial 噪声流。默认 1（与旧行为逐位一致）；gN 门禁与 60k 用 2，
#     g1 保持 1 worker 作参照——同一条 byte-diff 同时验证卡数与 worker 数两个
#     不变量。预期 ~1.3-1.4×（13.5h → ~10h）。
# 容忍度沿用用户当前 spec：stability 300% + 指标 0.1%。
# 顺序：phase0 自检(ADR-013 barrier 端到端峰值/恢复梯度/item7 断言 + ADR-012 legacy 断言
# + 四件测试含 test_blb_log_barrier_reward) → phase2 图门禁(REBUILD_MAPS=0) →
# phaseG 1卡vsN卡确定性门禁(探针出现性动态检测) → PASS 自动接 60k（带崩溃 watchdog）。
# ----------------------------------------------------------------------------
REBUILD_MAPS=0       # 图在 2026-06-11 已按 step-1×15 重建并 push（本两轮只改 RL 代码，动作→SF 解码未变）
GATE_EPISODES=300    # 门禁短跑规模：anchor80 + 220 post，5 个 PPO 窗口
LONG_EPISODES=60000  # 门禁通过后的里程碑长跑
KTRIALS=5            # K 固定为 5（与卡数解耦——这是确定性要求的一部分，勿改回 K=NGPU）
ANCHOR_EPISODES=80   # 与 mrpc-blb-stage2-rl preset 保持一致；探针从 anchor 后第一个 episode 开始
FUSION_PROBE_INTERVAL=200
WORKERS="$( n=$(nproc 2>/dev/null || echo 8); m=$(( n - 2 )); [ "$m" -lt 1 ] && m=1; [ "$m" -gt 128 ] && m=128; echo "$m" )"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; [ -z "$NGPU" ] && NGPU=1; [ "$NGPU" -lt 1 ] && NGPU=1
DEVS="$(seq -s, 0 $((NGPU-1)))"
echo "[gpu] 探测到 $NGPU 张卡 -> DEVS=$DEVS, K=$KTRIALS(固定); 枚举 WORKERS=$WORKERS (nproc=$(nproc 2>/dev/null || echo '?'))"
# ============================================================================

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_grid_gate_60k_${TS}"
mkdir -p "$OUT"
SKEL="Rescale_optimizer/configs/mrpc/static_skeletons_mrpc.json"
MAPS_DIR="blb_stage2_rl/fusion_maps/mrpc"
CANON_STAGE2_GROUP="Parting Chapter/persistent/rl/bert-base/mrpc"
CANON_STAGE2="${CANON_STAGE2_GROUP}/s1t0.001_s2t0.001_s2st3.0"

echo "==================== [phase0] 同步自检 ===================="
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$OUT/HEAD.txt"
  cat "$OUT/HEAD.txt"
  git log --oneline -5
else
  if [ -f SOURCE_SYNC_COMMIT.txt ]; then
    cat SOURCE_SYNC_COMMIT.txt > "$OUT/HEAD.txt"
  else
    echo "unknown-source-snapshot" > "$OUT/HEAD.txt"
  fi
  cat "$OUT/HEAD.txt"
  echo "[info] non-git source snapshot; using SOURCE_SYNC_COMMIT.txt"
fi
python3 - <<'PY' 2>&1 | tee "$OUT/selfcheck.txt" || { echo "[FATAL] 自检失败"; exit 1; }
import rescale_optimizer as r
print("RO 导入 OK；DEFAULT_FUSION_POLICY =", r.DEFAULT_FUSION_POLICY)
import sys; sys.path.insert(0, "blb_stage2_rl")
import action_space as asp
assert not hasattr(asp, "MIN_SF_FLOOR"), "无下限（user spec）"
assert asp.LEVELS_F == asp.LEVELS_W == asp.LEVELS_MS == asp.LEVELS_R == 15
assert [asp.sf_from(i, 30, 15) for i in range(14, -1, -1)] == list(range(30, 15, -1))
assert asp.distinct_sf_level_indices(kind="F", levels=15, max_sf=20, N=16384) == [0] + list(range(5, 15))
assert asp.distinct_sf_level_indices(kind="R", levels=15, max_sf=30, N=16384) == list(range(1, 15))
print("step-1×15 档位规则 OK（间隔1/最多15档/无下限/同值档预去重）")
import fusion_enum_fast as fef
import itertools
lens = [3, 4, 5]
full = list(itertools.product(*[range(n) for n in lens]))
assert [tuple(c) for c in fef.iter_combo_range(lens, 7, 31)] == full[7:31]
print("fusion_enum_fast unranking OK")
from seed_utils import derive_probe_seed, derive_policy_step_seed, PREFLIGHT_EPISODE
print("stage2 seed_utils OK; preflight episode =", PREFLIGHT_EPISODE)
# ---- ADR-011/012 断言：预算拆分 + 探针轮换(2,5) + 近界渐变 + 复测 + ε ----
import fusion_curriculum as fcur
assert fcur.FUSION_PROBE_BLOCK_ROTATION == (2, 5), "ADR-012: b4 必须已剔除"
assert fcur.fusion_probe_target_block(80, anchor_episodes=80, interval=200) == 2
assert fcur.fusion_probe_target_block(280, anchor_episodes=80, interval=200) == 5
assert fcur.fusion_probe_target_block(480, anchor_episodes=80, interval=200) == 2
assert fcur.fusion_probe_target_block(81, anchor_episodes=80, interval=200) is None
import fusion_cost, reward as rwd
assert abs(rwd.FUSION_COST_BUDGET_FRACTION - 2.0/3.0) < 1e-12
ch = [fusion_cost.BlockChoice(2, "block2_mrpc", 1, 1, 13)]
r = fusion_cost.compute_fusion_cost_saving(ch, fusion_w=rwd.FUSION_COST_W, trunc_w=rwd.TRUNC_COST_W)
assert abs(r.fusion_norm - 1.0) < 1e-12 and r.trunc_norm == 0.0
w = rwd.RewardWeights()
assert (w.near_miss_tier_cap, w.near_miss_tier_floor, w.near_miss_band) == (35.0, 15.0, 1.0)
base = rwd.BaselineCostStats(total_bits_sum=1000, total_fusion_count=0, avg_k=13.0,
                             loss_mean=0.34, loss_std=0.002, metric1_mean=0.8672,
                             metric2_mean=0.8672, metric1_std=0.001, metric2_std=0.001)
# ADR-013: barrier 现在默认开启并取代近界档；此处用 acc_barrier_enabled=False 验证 legacy 路径仍在。
w2 = rwd.RewardWeights(baseline_metric1=0.8672, baseline_metric2=0.8672, acc_barrier_enabled=False, reward_design="tiered")
class _O:
    any_invalid = False; total_bits_sum = 1000; total_fusion_count = 0
def _r(m1):
    m = rwd.EpisodeMetrics(loss_mean=0.34, loss_std=0.002, metric1_mean=m1,
                           metric2_mean=m1, metric1_std=0.001, metric2_std=0.001)
    return rwd.compute_reward(m, _O(), action_avg_k=13.0, baseline=base, weights=w2,
                              acc_threshold=0.858, acc_threshold_m2=0.858,
                              stab_threshold=0.05, external_cost_score=0.0,
                              external_cost_rank=0.0)
bnm = _r(0.8540)   # legacy 近界 P1
assert bnm.priority == 1 and bnm.near_miss and 20.0 < bnm.reward < 35.0
assert _r(0.8672).reward >= 40.0 > bnm.reward                  # P3 永远压住近界
assert _r(0.3200).reward < -4.0 and not _r(0.3200).near_miss   # 灾难型保留悬崖
print("ADR-012 近界渐变档(legacy, barrier off)断言 OK（边缘P1 -7 -> %.1f）" % bnm.reward)
# ---- ADR-013 断言：Stage-1 式 log-barrier（默认开启，取代近界档+线性 P3 margin）----
# ADR-014：MARGIN_REF 0.25→0.5（4th-60k 融合区探针 σ≈0.0155 是基线 σ 的 ~8.6×，0.25 亚 σ）。
# ADR-015：默认 reward_design 翻成 "continuous"；ADR-013/014 断言走 tiered 回滚路径，故 pin。
wb = rwd.RewardWeights(baseline_metric1=0.871, baseline_metric2=0.871, reward_design="tiered")
assert wb.acc_barrier_enabled and abs(wb.acc_barrier_margin_ref - 0.5) < 1e-12
assert rwd.accuracy_margin_barrier(0.5, wb) == 0.0                        # headroom 外归零
assert rwd.accuracy_margin_barrier(0.05, wb) < rwd.accuracy_margin_barrier(0.20, wb) < 0.0  # 回正力
assert rwd.accuracy_margin_barrier(-5.0, wb) < rwd.accuracy_margin_barrier(-0.1, wb)        # 违反区有梯度
assert rwd.accuracy_margin_barrier(-1e4, wb) == wb.acc_barrier_floor      # 下限
baseB = rwd.BaselineCostStats(total_bits_sum=11285, total_fusion_count=0, avg_k=13.0,
                              loss_mean=0.37, loss_std=0.01, metric1_mean=0.871, metric2_mean=0.871,
                              metric1_std=0.002, metric2_std=0.002, typical_bits_drop=1000,
                              typical_fusion_count=24, typical_k_drop=5)
def _rb(f):
    m1 = 0.871 - 0.0052 * f * (0.6 + 0.4 * f / 35.0)
    m = rwd.EpisodeMetrics(loss_mean=0.37, loss_std=0.01, metric1_mean=m1, metric2_mean=m1,
                           metric1_std=0.002, metric2_std=0.002)
    class _OB:
        any_invalid = False; total_bits_sum = 11285 - 30 * f; total_fusion_count = f
    return rwd.compute_reward(m, _OB(), action_avg_k=13.0, baseline=baseB, weights=wb,
                              acc_threshold=0.858, acc_threshold_m2=0.858,
                              external_cost_score=min(3.0, 3.0 * f / 36.0), external_cost_rank=float(f))
sweep = [(f,) + (lambda rb: (rb.reward, rb.priority, rb.worst_signed_margin))(_rb(f)) for f in range(36)]
peak = max(sweep, key=lambda r: r[1])
assert 0 < peak[0] < 35 and peak[2] == 3 and peak[3] > 0.0, f"峰值必须是正余量内点 P3: {peak}"
assert peak[1] > sweep[0][1], "fusion 必须被采纳（峰值 > 零融合基线）"
viol = [r[1] for r in sweep if r[2] == 1]
assert viol and all(viol[i] > viol[i+1] for i in range(len(viol)-1)), "违反区必须单调（恢复梯度）"
assert _rb(0).priority == 3
worst_p3 = _rb(0).reward
hi_cost_p1 = rwd.compute_reward(
    rwd.EpisodeMetrics(loss_mean=0.37, loss_std=0.01, metric1_mean=0.8579, metric2_mean=0.8579,
                       metric1_std=0.002, metric2_std=0.002),
    type("OB", (), {"any_invalid": False, "total_bits_sum": 9000, "total_fusion_count": 24})(),
    action_avg_k=8.0, baseline=baseB, weights=wb, acc_threshold=0.858, acc_threshold_m2=0.858,
    external_cost_score=4.5, external_cost_rank=24.0).reward
assert hi_cost_p1 < worst_p3, "item7：高 cost 的 P1 仍 < 任意 P3"
print("ADR-013 log-barrier 断言 OK：内点峰值 fusion=%d reward=%.2f margin=%.2f；违反区有恢复梯度" % (peak[0], peak[1], peak[3]))
# ---- ADR-014 断言：结构性反失控 fusion 成本（饱和）+ 黑箱落盘 ----
import blb_stage2_rl.fusion_cost as _fc
assert abs(rwd.FUSION_SATURATION_TAU - 0.15) < 1e-12   # ADR-014 reference constant kept
# ADR-015: RewardWeights default fusion_saturation_tau retired to 0.0 (saturation off
# under continuous reward). The function still works with an explicit tau (below).
assert abs(wb.fusion_saturation_tau - 0.0) < 1e-12
assert _fc.saturate_fusion(0.5, 0.0) == 0.5 and _fc.saturate_fusion(1.0, 0.15) == 1.0  # 关=恒等; 端点
# 凹性：边际递减（反失控核心）；过拐点边际 ≪ 起点边际
_sat = [_fc.saturate_fusion(i / 40.0, 0.15) for i in range(41)]
_slp = [_sat[i + 1] - _sat[i] for i in range(40)]
assert all(_slp[i] >= _slp[i + 1] - 1e-12 for i in range(39)) and _slp[-1] < 0.1 * _slp[0]
assert _fc.saturate_fusion(0.23, 0.15) > 0.7   # ~fusion8 已达 ~80%
# 饱和后 cost(fusion)+barrier 仍在中等正余量出现内点峰值（max fusion 不最优）
def _rbs(f):  # 用真实 fusion_norm_saturated（凹）替换线性 cost
    fn = f / 35.0
    fns = _fc.saturate_fusion(fn, 0.15)
    m1 = 0.871 - 0.0052 * f * (0.6 + 0.4 * f / 35.0)
    m = rwd.EpisodeMetrics(loss_mean=0.37, loss_std=0.01, metric1_mean=m1, metric2_mean=m1,
                           metric1_std=0.002, metric2_std=0.002)
    class _OB:
        any_invalid = False; total_bits_sum = 11285 - 30 * f; total_fusion_count = f
    return rwd.compute_reward(m, _OB(), action_avg_k=13.0, baseline=baseB, weights=wb,
                              acc_threshold=0.858, acc_threshold_m2=0.858,
                              external_cost_score=fns * (wb.p3_cost_budget * rwd.FUSION_COST_BUDGET_FRACTION),
                              external_cost_rank=float(f))
_sw = [(f, _rbs(f).reward, _rbs(f).priority, _rbs(f).worst_signed_margin) for f in range(36)]
_pk = max(_sw, key=lambda r: r[1])
assert 0 < _pk[0] < 35 and _pk[2] == 3 and _pk[3] > 0.0, f"饱和后峰值仍须正余量内点 P3: {_pk}"
assert _sw[-1][1] < _pk[1], "max fusion 不能最优"
# B1：barrier/margin/per-block fusion 已落进 episodes.jsonl 的 EpisodeStats schema
import dataclasses as _dc
from blb_stage2_rl.diagnostics import EpisodeStats as _ES
_esf = {f.name for f in _dc.fields(_ES)}
for _k in ("terminal_worst_signed_margin", "terminal_acc_barrier_sat", "terminal_acc_barrier_vio",
           "terminal_near_miss", "terminal_margin_m1", "terminal_margin_m2",
           "terminal_fusion_norm_raw", "terminal_fusion_norm_saturated"):
    assert _k in _esf, f"EpisodeStats 缺 ADR-014 调试字段 {_k}"
print("ADR-014 断言 OK：饱和凹性 + 内点峰值 fusion=%d reward=%.2f margin=%.2f + episodes.jsonl 黑箱字段齐全" % (_pk[0], _pk[1], _pk[3]))
# ---- ADR-015 断言：连续有界 reward（默认）+ 边界连续 + item7 + 严格稳定性刹车 ----
assert rwd.RewardWeights().reward_design == "continuous"   # 重建是默认
wc = rwd.RewardWeights(baseline_metric1=0.871, baseline_metric2=0.871, stab_tolerance=1.2)  # continuous（默认）；stab_tolerance = baseline std 的倍率
THRc = 0.858
def _rc(m1, std=0.002, fusion=0, cost=0.0, invalid=False):
    met = rwd.EpisodeMetrics(loss_mean=0.37, loss_std=std, metric1_mean=m1, metric2_mean=m1, metric1_std=std, metric2_std=std)
    class _OB: any_invalid=invalid; total_bits_sum=11285-30*fusion; total_fusion_count=fusion
    return rwd.compute_reward(met, _OB(), action_avg_k=13.0, baseline=baseB, weights=wc,
                             acc_threshold=THRc, acc_threshold_m2=THRc, external_cost_score=cost, external_cost_rank=float(fusion))
# 有界 [-5,5]
_cont = [_rc(0.871,fusion=8,cost=2.0), _rc(THRc-0.001,fusion=8,cost=2.0), _rc(0.70,fusion=24,cost=4.5),
         _rc(0.871,std=0.05,fusion=8,cost=2.0), _rc(0.871,invalid=True)]
assert all(-5.0001 <= b.reward <= 5.0001 for b in _cont), "continuous reward 必须有界 [-5,5]"
# 跨可行边界连续（无 ±40 跳）
_gap = abs(_rc(THRc+0.0005,fusion=6,cost=1.5).reward - _rc(THRc-0.0005,fusion=6,cost=1.5).reward)
assert _gap < 8.0, f"continuous 边界 gap 应远小于 tiered ±40：{_gap}"
# item7：高 cost 的 P1 < P3
assert _rc(0.70,fusion=24,cost=4.5).reward < _rc(0.871,fusion=4,cost=1.0).reward
# 严格稳定性刹车：高 std → P2（非 P3），拿不到 cost
_hi = _rc(0.871, std=0.05, fusion=8, cost=2.0)
assert _hi.priority == 2 and not _hi.stab_ok and _hi.metric_ok, "高 std 必须落 P2（稳定性刹车）"
# 严格稳定性 = 倍率门（2026-06-15 user spec）：阈值 = baseline.X_std × tol（非 fractional slack）。
# 用 baseline std=0.01 让倍率而非 floor(0.01) 决定门限，证明 5.0(=5×) 宽松但仍是真门、1.2(=1.2×) 严格。
_baseS = rwd.BaselineCostStats(total_bits_sum=11285, total_fusion_count=0, avg_k=13.0,
                               loss_mean=0.37, loss_std=0.01, metric1_mean=0.871, metric2_mean=0.871,
                               metric1_std=0.01, metric2_std=0.01, typical_bits_drop=1000,
                               typical_fusion_count=24, typical_k_drop=5)
def _stab_at(tol, obs):
    w = rwd.RewardWeights(baseline_metric1=0.871, baseline_metric2=0.871, stab_tolerance=tol, reward_design="continuous")
    met = rwd.EpisodeMetrics(loss_mean=0.37, loss_std=obs, metric1_mean=0.871, metric2_mean=0.871, metric1_std=obs, metric2_std=obs)
    class _OB: any_invalid=False; total_bits_sum=11285; total_fusion_count=8
    return rwd.compute_reward(met, _OB(), action_avg_k=13.0, baseline=_baseS, weights=w, acc_threshold=THRc, acc_threshold_m2=THRc)
# std=0.03：5.0×(thr=0.05) 通过、1.2×(thr=0.012) 拒绝 → 5.0 宽松但真门（非空门）
assert _stab_at(5.0, 0.03).stab_ok, "stab_tolerance=5.0 → thr=baseline_std×5=0.05，std=0.03 应通过（宽松但真门，非空门）"
assert not _stab_at(1.2, 0.03).stab_ok, "stab_tolerance=1.2 → thr=baseline_std×1.2=0.012，std=0.03 应被拒（严格门）"
# 证明 5.0 非空门：std 抬到 0.06 仍被 5.0× 拒
assert not _stab_at(5.0, 0.06).stab_ok, "stab_tolerance=5.0 仍是真门：std=0.06 > thr 0.05 应被拒"
# loss_mean 硬约束（2026-06-15 user spec "loss 也是" 越低越好；允许上浮 limit_tol；
# continuous 门控、tiered 不门控）。baseB.loss_mean=0.37，loss_threshold=0.37×1.005=0.37185。
def _rc_loss(design, loss_mean):
    w = rwd.RewardWeights(baseline_metric1=0.871, baseline_metric2=0.871, stab_tolerance=5.0, reward_design=design)
    met = rwd.EpisodeMetrics(loss_mean=loss_mean, loss_std=0.002, metric1_mean=0.871, metric2_mean=0.871, metric1_std=0.002, metric2_std=0.002)
    class _OB: any_invalid=False; total_bits_sum=11285; total_fusion_count=8
    return rwd.compute_reward(met, _OB(), action_avg_k=13.0, baseline=baseB, weights=w, acc_threshold=THRc, acc_threshold_m2=THRc)
assert _rc_loss("continuous", 0.371).loss_ok, "continuous: loss 在 +0.5% 容忍内 loss_ok"
# 2026-06-15 确定性修复：loss_mean 不进逐 episode priority/metric_ok（带噪 loss 跨卡数不确定，会破 1==N）；
# loss_ok 仅是对 CLEAN 基线的确定性诊断；硬 loss 约束在严格选择处用 noisy 阈值执行。
_lbad = _rc_loss("continuous", 0.40)
assert (not _lbad.loss_ok) and _lbad.metric_ok and _lbad.priority == 3, "continuous: loss 超界仅 loss_ok=False（不改 metric_ok/priority，确定性）"
assert _rc_loss("continuous", 0.10).loss_ok, "loss 越低越好：loss 下降 loss_ok 必 True"
_ltier = _rc_loss("tiered", 5.0)
assert _ltier.loss_ok and _ltier.metric_ok and _ltier.priority == 3, "tiered: loss_mean 不门控（逐位不变）"
print("ADR-015 断言 OK：连续有界（gap=%.2f<8） + item7 + 严格稳定性=倍率门（5.0× 宽松但非空门 / 1.2× 严格）+ loss_ok 确定性诊断（不进 priority）+ 默认 continuous" % _gap)
# ---- ADR-016 断言：headroom cost（消刀刃+内点峰）+ 线性违反恢复梯度（止冻死，修第5次60k fusion 失控）----
def _cont016(margin, cost_frac):
    eff = cost_frac * wc.p3_cost_budget if margin >= 0.0 else 0.0
    s, _ab, _sb = rwd._continuous_reward(acc_margins=[margin], std_margins=[10.0], effective_cost_score=eff, invalid=False, weights=wc)
    return s
# (1) 违反区恢复梯度：轻微违反 > 灾难违反（旧 −VIO·exp 把两者都压平 −5 → 冻死）
assert _cont016(-2.0, 0.0) > _cont016(-20.0, 0.0) + 0.5, "ADR-016: 违反区需有恢复梯度（mild > deep，非平 −5）"
# (2) 内点峰：合成 fusion 扫描（cost↑ / margin↓ 过零）→ 峰在中等 fusion、非 max、非 0
_sw = [(f, _cont016(3.0 - 0.18 * f, min(1.0, f / 30.0))) for f in range(0, 37)]
_pf, _pr = max(_sw, key=lambda x: x[1])
assert 2 < _pf < 30, "ADR-016: reward 峰必为内点（非 max-fusion 失控、非 0-fusion 冷崩）"
assert _sw[-1][1] < _pr - 1.0, "ADR-016: max fusion 必明显劣于峰（回正力，旧 reward 反而单调上爬）"
# (3) 无刀刃：headroom 让 cost 在边界平滑→0（旧 P3 门 cliff ~9）
assert abs(_cont016(0.05, 1.0) - _cont016(-0.05, 1.0)) < 1.0, "ADR-016: 跨边界无刀刃 cliff"
# (4) 有界
assert all(-5.0001 <= _cont016(m, c) <= 5.0001 for m in (-20, -2, 0.0, 1.0, 3.0) for c in (0.0, 1.0)), "ADR-016: 仍有界[−5,5]"
print("ADR-016 断言 OK：内点峰@f=%d（R=%.2f）+ 恢复梯度（mild %.2f > deep %.2f）+ 无刀刃 + 有界" % (_pf, _pr, _cont016(-2.0, 0.0), _cont016(-20.0, 0.0)))
import sys as _sys
_sys.path.insert(0, ".")
from blb_stage2_rl.env import BLBStage2EnvConfig
ec = BLBStage2EnvConfig()
assert ec.borderline_retest_enabled and ec.borderline_retest_trials_multiplier == 2
from blb_stage2_rl.sequential_runner import SequentialTrainConfig as _STC
stc = _STC()
assert abs(stc.fusion_exploration_epsilon - 0.05) < 1e-12
assert abs(stc.fusion_exploration_epsilon_k - 0.02) < 1e-12
print("ADR-012 复测/ε 默认值断言 OK")
# ---- workers-per-device（2026-06-12 提速）断言 ----
from blb_stage2_rl.parallel_runner import expand_device_ids_for_workers
assert expand_device_ids_for_workers([0, 1, 2, 3, 4], 2) == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
assert expand_device_ids_for_workers([0, 1], 1) == [0, 1]   # wpd=1 == 旧行为
from blb_stage2_rl.runner import BLBStage2TrainConfig as _BTC
assert _BTC().stage2_workers_per_device == 1
print("workers-per-device 断言 OK（默认 1；gN/60k 用 worker policy + dynamic assignment）")
PY
echo "==================== [phase0b] ADR-012/013 单元测试（torch 在位：log-barrier/ε混合/复测/近界档/轮换） ===================="
for f in test_blb_fusion_curriculum test_blb_fusion_reward test_blb_fusion_exploration test_blb_log_barrier_reward test_blb_fusion_saturation test_blb_stage2_outputs test_blb_continuous_reward; do
  python3 "tests/${f}.py" > "$OUT/unittest_${f}.log" 2>&1 || { echo "[FATAL] ${f} 失败"; tail -20 "$OUT/unittest_${f}.log"; exit 1; }
  tail -1 "$OUT/unittest_${f}.log"
done
# 前置 Stage-1 record（缺失则从已提交 degrees 合成；幂等）
python3 - <<'PY' 2>&1 | tee "$OUT/stage1_record_synth.txt" || { echo "[FATAL] Stage-1 record 处理失败"; exit 1; }
import json, os, glob, datetime
rec_root = "Parting Chapter/stage1/record"; combo = "bert base mrpc"
existing = [d for d in glob.glob(os.path.join(rec_root, combo + " *")) if os.path.isdir(d)]
if existing:
    print("[skip] 已存在 Stage-1 record：", [os.path.basename(d) for d in existing]); raise SystemExit(0)
ap = json.load(open("Model_analysis/configs/approx_per_dataset.json"))
s1 = ap["mrpc"]["stage1"]; gelu = [int(x) for x in s1["gelu"]]; softmax = [int(x) for x in s1["softmax"]]
assert 0 not in gelu, f"degree-0 不应出现: {gelu}"
date = datetime.datetime.now().strftime("%Y%m%d")
rec_dir = os.path.join(rec_root, f"{combo} 1 {date}")
os.makedirs(rec_dir, exist_ok=True)
json.dump({"gelu_degree_per_layer": gelu, "softmax_degree_per_layer": softmax,
           "_synthesized_from": "Model_analysis/configs/approx_per_dataset.json"},
          open(os.path.join(rec_dir, "final_config.json"), "w"), ensure_ascii=False, indent=2)
print("[ok] 合成 Stage-1 record:", rec_dir, "| gelu =", gelu)
PY

echo "==================== [phase1] step-1×15 全量重建 fusion 图（直连 replan 快路径 + 等价门禁）===================="
cp -a "$MAPS_DIR" "$OUT/old_maps" 2>/dev/null || true
if [ "$REBUILD_MAPS" = 1 ]; then
  # 仅在确实重建时才清空图目录（rm -rf 必须留在守卫内，防误删有效图）。
  rm -rf "$MAPS_DIR"
  mkdir -p "$MAPS_DIR"
  # 全量枚举（--max-enum-combos 0，不走捷径）。小图 both = 金/快两路各自全量、
  # 最终选项必须逐项相等（最强交叉验证）；大图 fast + 128 随机金vs快对拍门禁
  #（不一致即 FATAL）。小图在前：交叉验证先趟雷，大图不白跑。
  for gk in block1_mrpc block5_n1 block5_n2 block2_mrpc block5_n4 block4; do
    EPATH="fast"
    case "$gk" in block1_mrpc|block5_n1) EPATH="both" ;; esac
    echo "[maps] building $gk (enum-path=$EPATH, workers=$WORKERS) ..."
    python scripts/blb_build_fusion_count_map.py --profile mrpc --only "$gk" \
      --out-dir "$MAPS_DIR" --rescale-optimizer-root Rescale_optimizer \
      --num-layers 12 --workers "$WORKERS" --max-enum-combos 0 \
      --enum-path "$EPATH" --fast-verify-random 128 \
      > "$OUT/build_${gk}.log" 2>&1 || { echo "[FATAL] fusion 图 $gk 构建失败，见 build_${gk}.log"; tail -30 "$OUT/build_${gk}.log"; exit 1; }
    grep -E "\[fast\] template OK|\[both\] fast == golden|options=|wall=|rate=" "$OUT/build_${gk}.log" | tail -4
  done
fi

echo "==================== [phase2] 图门禁（option0==baseline）+ 新旧对比 ===================="
OUTDIR="$OUT" python3 - <<'PY' 2>&1 | tee "$OUT/map_gate.txt" || { echo "[FATAL] 图门禁失败"; exit 1; }
import glob, json, os
from blb_stage2_rl.fusion_count_map import FusionCountMap
FusionCountMap.load("mrpc")
print("FusionCountMap.load('mrpc') OK — 所有图 option0==baseline。")
def summarize(p):
    d = json.load(open(p)); o = d["options"]
    return len(o), sorted({x["fusion_count"] for x in o}), d.get("build_meta", {}).get("enum_total_combos")
old = "%s/old_maps" % os.environ.get("OUTDIR", "")
print("\n图              新(n_opt, fusion, enum_total)          旧(n_opt, fusion, enum_total)")
for f in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    b = os.path.basename(f)
    if b.startswith("_"): continue
    nn = summarize(f)
    op = os.path.join(old, b)
    oo = summarize(op) if os.path.exists(op) else "—"
    print(f"  {b:16s} {str(nn):36s} {oo}")
PY
cp -a "$MAPS_DIR" "$OUT/new_maps"

echo "==================== [phaseG] Stage-2 episode 并行确定性门禁：1卡 vs ${NGPU}卡 ===================="
GOUT="$OUT/stage2_ngpu_gate"; mkdir -p "$GOUT"
run_gate () {   # tag, visible devs, --stage2-rl-devices 值, workers-per-device
  local tag="$1" vis="$2" devspec="$3" wpd="${4:-1}" pid rundir t0 t1 run_tag
  run_tag="gate_${tag}_${TS}"
  rundir="${CANON_STAGE2_GROUP}/s1t0.001_s2t0.001_s2st3.0__${run_tag}"
  echo "-------- [gate] $tag CUDA_VISIBLE_DEVICES=$vis stage2-rl-devices=$devspec wpd=$wpd episodes=$GATE_EPISODES --------"
  BLB_STAGE2_POLICY_DEVICE=worker BLB_STAGE2_DYNAMIC_ASSIGNMENT=1 \
  CUDA_VISIBLE_DEVICES="$vis" bash llama_7B_LayerImportance.sh run rl \
    --preset mrpc-blb-stage2-rl \
    --run-tag "gate_${tag}_${TS}" \
    --blb-v3-fusion-count-action 1 \
    --blb-v3-fusion-neighbor-curriculum 1 \
    --stage2-workers-per-device "$wpd" \
    --stage2-search-episodes "$GATE_EPISODES" \
    --stage2-k-trials "$KTRIALS" \
    --stage2-probe-size 256 \
    --batch-size 512 \
    --stage2-rl-devices "$devspec" \
    --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
    --stage2-fixed-config-source json \
    --stage2-fixed-config glue_final_configs_best_ppo.json \
    --stage2-stability-tolerance 3.0 \
    --stage2-limit-tolerance 0.001 \
    --blb-v3-fusion-probe-interval "$FUSION_PROBE_INTERVAL" \
    --blb-v3-fusion-exploration-epsilon 0.05 \
    --fresh 2>&1 | tee "$GOUT/${tag}_launch.log"
  sleep 12
  pid="$(cat "${rundir}/run.pid" 2>/dev/null || cat "${rundir}/rl.pid" 2>/dev/null || true)"
  [ -z "$pid" ] && { echo "[gate][FATAL] $tag 没拿到 PID"; return 1; }
  t0=$(date +%s); while kill -0 "$pid" 2>/dev/null; do sleep 20; done; t1=$(date +%s)
  echo "$((t1 - t0))" > "$GOUT/${tag}_walltime_s.txt"
  # 逐窗签名：workers= 在行尾，截取与卡数无关的前缀做 byte-diff
  grep -rhoE "window_start=[0-9]+ episodes=[0-9]+ rollout_sig=[0-9a-f]+" "$rundir" 2>/dev/null \
    | sort -u > "$GOUT/${tag}_sigs.txt" || true
  grep -rh "\[ANOMALY\]" "$rundir" 2>/dev/null > "$GOUT/${tag}_anomaly.txt" || true
  local diag; diag=$(find "$rundir" -type f -name episodes.jsonl 2>/dev/null | head -1)
  [ -n "$diag" ] && cp "$diag" "$GOUT/${tag}_episodes.jsonl"
  python3 scripts/verify_stage2_persistent_outputs.py \
    --run-dir "$rundir" \
    --min-episodes "$GATE_EPISODES" \
    --min-ppo-updates 1 \
    --require-png 2>&1 | tee "$GOUT/${tag}_persistent_verify.txt"
}
# g1 = 最简参照（1 worker 总量）；gN = 生产配置（N 卡 × 2 worker/卡）。
# 同一条 byte-diff 同时验证「卡数无关」与「worker 数无关」两个不变量。
run_gate g1 0       0       1  || { echo "[FATAL] 门禁 g1 失败"; exit 1; }
run_gate gN "$DEVS" "$DEVS" 1  || { echo "[FATAL] 门禁 gN 失败"; exit 1; }

echo "==== [gate] 判读 ====" | tee "$GOUT/verdict.txt"
GATE_PASS=1
if [ -s "$GOUT/g1_sigs.txt" ] && diff "$GOUT/g1_sigs.txt" "$GOUT/gN_sigs.txt" > "$GOUT/sig_diff.txt" 2>&1; then
  echo "[gate][PASS] rollout_sig 逐窗逐字相同（1卡 == ${NGPU}卡）" | tee -a "$GOUT/verdict.txt"
else
  GATE_PASS=0; echo "[gate][FAIL] rollout_sig 不一致或为空 → 看 sig_diff.txt" | tee -a "$GOUT/verdict.txt"
fi
if [ -s "$GOUT/g1_anomaly.txt" ] || [ -s "$GOUT/gN_anomaly.txt" ]; then
  GATE_PASS=0; echo "[gate][FAIL] 出现 [ANOMALY]（fusion 图存在 invalid 动作?）" | tee -a "$GOUT/verdict.txt"
fi
python3 - <<PY 2>&1 | tee -a "$GOUT/verdict.txt"
import json
fields = ["episode","total_reward","terminal_reward","terminal_priority","terminal_loss_mean",
          "terminal_loss_std","terminal_metric1_mean","terminal_metric2_mean","fusion_count","total_bits"]
def rows(p):
    return [tuple(json.loads(l).get(f) for f in fields) for l in open(p)]
try:
    a = rows("$GOUT/g1_episodes.jsonl"); b = rows("$GOUT/gN_episodes.jsonl")
    same = (len(a) == len(b)) and all(x == y for x, y in zip(a, b))
    print(f"[gate] episodes.jsonl 数值逐项对比: {'PASS（完全一致）' if same else 'FAIL（存在差异）'}  n={len(a)}/{len(b)}")
    if not same:
        for i,(x,y) in enumerate(zip(a,b)):
            if x != y: print("  first diff @", i, x, "vs", y); break
        raise SystemExit(2)
except FileNotFoundError as e:
    print("[gate][FAIL] episodes.jsonl 缺失:", e); raise SystemExit(2)
PY
[ $? -ne 0 ] && GATE_PASS=0
# 2026-06-15 (ADR-015): forced-fusion probes are an ADR-011/012 patch for the TIERED
# reward. Under reward_design="continuous" (the rebuild's default) the runner gates
# probes OFF (fusion_probe_interval=0, along with anchor/warmstart/ε/curriculum), so
# expecting probe episodes is STALE. This is now an INFORMATIONAL report only — their
# absence under continuous is correct and must NOT fail the gate. (If a future tiered
# A/B run is gated, re-enable the strict rotation/fc check below.)
python3 - <<PY 2>&1 | tee -a "$GOUT/verdict.txt"
import json
probes = []
for l in open("$GOUT/gN_episodes.jsonl"):
    d = json.loads(l)
    mode = str(d.get("exploration_mode", ""))
    if mode.startswith("forced_fusion_probe_"):
        probes.append((int(d.get("episode", -1)), mode, int(d.get("fusion_count", -1))))
probes.sort()
print(f"[gate][info] forced_fusion_probe episodes: {probes} "
      f"(expected EMPTY under ADR-015 continuous — probes gated off)")
PY
g1s=$(cat "$GOUT/g1_walltime_s.txt" 2>/dev/null || echo 0); gNs=$(cat "$GOUT/gN_walltime_s.txt" 2>/dev/null || echo 0)
python3 -c "
g1=$g1s; gN=$gNs; ep=$GATE_EPISODES; nd=$NGPU
print(f'1-GPU : {ep/(g1/3600):.0f} ep/h ({g1}s)') if g1>0 else print('1-GPU : n/a')
print(f'{nd}-GPU : {ep/(gN/3600):.0f} ep/h ({gN}s)  speedup={g1/gN:.2f}x (ideal {nd}x)') if (g1>0 and gN>0) else 0
" | tee -a "$GOUT/verdict.txt"

if [ "$GATE_PASS" != 1 ]; then
  echo "[STOP] 门禁未通过——不启动 60k。请回传 $OUT 全部产物供本地诊断。"; exit 1
fi

echo "==================== [phase60k] 门禁 PASS → 启动 ${LONG_EPISODES}-episode curriculum-ON fusion 长跑 ===================="
BLB_STAGE2_POLICY_DEVICE=worker BLB_STAGE2_DYNAMIC_ASSIGNMENT=1 \
CUDA_VISIBLE_DEVICES=$DEVS bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --blb-v3-fusion-count-action 1 \
  --blb-v3-fusion-neighbor-curriculum 1 \
  --stage2-search-episodes "$LONG_EPISODES" \
  --stage2-k-trials "$KTRIALS" \
  --stage2-probe-size 256 \
  --batch-size 512 \
  --stage2-rl-devices "$DEVS" \
  --blb-v3-warmstart-anchor-episodes "$ANCHOR_EPISODES" \
  --stage2-fixed-config-source json \
  --stage2-fixed-config glue_final_configs_best_ppo.json \
  --stage2-stability-tolerance 3.0 \
  --stage2-limit-tolerance 0.001 \
  --blb-v3-fusion-probe-interval "$FUSION_PROBE_INTERVAL" \
  --blb-v3-fusion-exploration-epsilon 0.05 \
  --stage2-workers-per-device 1 \
  --fresh 2>&1 | tee "$OUT/long60k_launch.log"
sleep 12
PID60="$(cat "${CANON_STAGE2_GROUP}/LATEST_PID" 2>/dev/null || true)"
RUN60="$(cat "${CANON_STAGE2_GROUP}/LATEST_RUN_DIR" 2>/dev/null || true)"
[ -z "$PID60" ] && { echo "[FATAL] 60k 启动失败，看 long60k_launch.log"; exit 1; }
echo "PID=$PID60  run_dir=$RUN60  started=$(date -Is)" | tee "$OUT/long60k_RUNNING.txt"
for _i in 1 2 3 4 5; do
  if python3 scripts/verify_stage2_persistent_outputs.py \
    --run-dir "$RUN60" \
    --min-episodes 1 \
    --min-ppo-updates 0 \
    2>&1 | tee "$OUT/long60k_start_persistent_verify.txt"; then
    break
  fi
  sleep 60
done
# 监控循环：每 30 分钟记录健康快照（rolling reward / P1 P2 P3 / fusion + per-type b2/b4/b5 / 进度）
# 兼 ADR-013 崩溃 watchdog：barrier 本应根除热崩溃，但作为安全网——若连续 ≥COLLAPSE_PATIENCE
# 个滚动窗口 P3 占比 < COLLAPSE_P3_FLOOR（=持续无 P3 的死亡签名，3rd-60k 后 30k 回合就是如此），
# 则 KILL 训练进程（sequential 路径不检查 STOP_RL，故用 SIGTERM→SIGKILL；best/diagnostics 已周期性
# 原子写盘，杀掉只丢最近 <200 回合的可恢复状态，对已坍缩的跑毫无损失）。判据基于 P3 占比而非 P1
# 率，规避 ADR-012 "near-miss P1 不计" 放走真崩溃的漏洞。健康曲线正常（有 P3）时永不触发。
DIAG60=""; for _i in 1 2 3 4 5; do DIAG60=$(find "$RUN60" -type f -name episodes.jsonl 2>/dev/null | head -1); [ -n "$DIAG60" ] && break; sleep 60; done
COLLAPSE_P3_FLOOR=0.02      # 窗口 P3 占比低于此即记一次"坍缩窗口"
COLLAPSE_PATIENCE=12        # 连续 12 个 30min 窗口（~6h、>3000 anchor 后回合）全坍缩 → 停止
COLLAPSE_STREAK_FILE="$OUT/collapse_streak.txt"; echo 0 > "$COLLAPSE_STREAK_FILE"
while kill -0 "$PID60" 2>/dev/null; do
  sleep 1800
  WD_VERDICT=$(COLLAPSE_P3_FLOOR="$COLLAPSE_P3_FLOOR" COLLAPSE_PATIENCE="$COLLAPSE_PATIENCE" \
  STREAK_FILE="$COLLAPSE_STREAK_FILE" \
  python3 - <<PY 2>>"$OUT/long60k_health.log"
import json, datetime, collections, os
try:
    eps=[json.loads(l) for l in open("$DIAG60")][-600:]
    pr=collections.Counter(int(e.get("terminal_priority",0) or 0) for e in eps)
    n=max(1,len(eps))
    rw=sum(float(e.get("total_reward",0) or 0) for e in eps)/n
    fu=sum(float(e.get("fusion_count",0) or 0) for e in eps)/n
    b2=sum(float(e.get("fusion_count_b2",0) or 0) for e in eps)/n
    b4=sum(float(e.get("fusion_count_b4",0) or 0) for e in eps)/n
    b5=sum(float(e.get("fusion_count_b5",0) or 0) for e in eps)/n
    last=eps[-1].get("episode") if eps else -1
    p3_frac=pr.get(3,0)/n
    import sys
    sys.stderr.write(f"{datetime.datetime.now().isoformat()} ep={last} rolling600: reward={rw:.3f} "
          f"P1={pr.get(1,0)} P2={pr.get(2,0)} P3={pr.get(3,0)} fusion={fu:.2f} (b2={b2:.1f} b4={b4:.1f} b5={b5:.1f})\n")
    floor=float(os.environ["COLLAPSE_P3_FLOOR"]); patience=int(os.environ["COLLAPSE_PATIENCE"])
    sf=os.environ["STREAK_FILE"]
    streak=int(open(sf).read().strip() or 0) if os.path.exists(sf) else 0
    streak = streak + 1 if (last >= 600 and p3_frac < floor) else 0
    open(sf,"w").write(str(streak))
    print("COLLAPSE" if streak >= patience else "OK")
except Exception as e:
    import sys; sys.stderr.write(f"{datetime.datetime.now().isoformat()} health probe error: {e}\n"); print("OK")
PY
)
  if [ "$WD_VERDICT" = "COLLAPSE" ]; then
    echo "[watchdog][COLLAPSE] P3<${COLLAPSE_P3_FLOOR} 连续 ${COLLAPSE_PATIENCE} 窗口 → 终止 PID=$PID60（best 已 checkpoint）" | tee -a "$OUT/long60k_health.log" | tee -a "$OUT/long60k_RUNNING.txt"
    kill -TERM "$PID60" 2>/dev/null || true; sleep 60; kill -KILL "$PID60" 2>/dev/null || true
    break
  fi
done
echo "[60k] training process exited at $(date -Is)" | tee -a "$OUT/long60k_RUNNING.txt"
# 回收产物（不含 .pt 大件）
copy_run_artifacts () {
  local rundir="$1" dest="$2"; mkdir -p "$dest"
  local diagdir; diagdir=$(find "$rundir" -type d -name diagnostics 2>/dev/null | head -1)
  if [ -n "$diagdir" ]; then rsync -a --exclude='*.pt' --exclude='__pycache__' "$(dirname "$diagdir")/" "$dest/"
  else rsync -a --exclude='*.pt' --exclude='__pycache__' "$rundir/" "$dest/"; fi
}
[ -n "$RUN60" ] && [ -d "$RUN60" ] && copy_run_artifacts "$RUN60" "$OUT/long60k/run"
if [ -n "$RUN60" ] && [ -d "$RUN60" ]; then
  python3 scripts/verify_stage2_persistent_outputs.py \
    --run-dir "$RUN60" \
    --min-episodes "$LONG_EPISODES" \
    --min-ppo-updates 1 \
    --require-png 2>&1 | tee "$OUT/long60k_persistent_verify.txt" || true
fi
tail -5 "$OUT/long60k_health.log" 2>/dev/null || true

echo "==================== DONE ===================="
echo "[push] 请回传：(1) 新 canonical 图  git add \"$MAPS_DIR\"   (2) 全部运行产物  git add \"$OUT\""
ls -la "$OUT"
```

## metadata

### 本次目标（2026-06-13，ADR-013 Stage-1 式 log-barrier 精度边界）

1. **背景**：第 3 次 60k（artifacts `stage2_grid_gate_60k_20260612_191530`，ADR-012 全开）门禁 PASS、跑满 60000，但**翻转成「热崩溃」**（与前两次冷崩溃相反）：fusion 单调 1.4→35、metric1 0.866→0.690、后 30k 回合冻结平在 reward -6.95（全 P1、零梯度）。但 hard-priority 选择仍救回 ep20880 = fusion 22 / P3 / reward 40.8 > 无融合上限 39.5（余量 0.0003=刀刃/亚 σ）。根因：ADR-012 近界渐变档把越界变近免费(-7→15-35)，叠加单调 fusion 成本(block4 @130 最毁精度却付第二高) + ~1.3% 精度预算被 fusion+深K 共同吃掉 → 无回正力一路上滑、掉进平坦无梯度盆地。
2. **本提交修复**（ADR-013；详见 `docs/adr/ADR-013-*.md`；用户选 log-barrier + 保持成本权重）：① Stage-1 式两段 log-barrier `reward.accuracy_margin_barrier` 取代近界档(P1)+线性 P3 margin(P3)——满足侧近界陡降 ⇒ cost+barrier 在正余量出现内点峰值(不冲过)；违反侧线性单调 ⇒ 恢复梯度(根治冻结)；② MARGIN_REF=0.25(≈1.8σ headroom)=激进度旋钮；③ priority/rank/选择逐位不变 + item7 + 1==N 保持；④ 保留 ADR-012 复测/ε/policy-K 探针；⑤ per-block-type fusion 诊断 + 崩溃 watchdog。
3. **流程**：REBUILD_MAPS=0 → 自检（ADR-013 barrier 端到端断言 + ADR-012 legacy 断言 + 4 件单测含 `test_blb_log_barrier_reward`，torch 在位）→ 图门禁 → 1卡vsN卡确定性门禁（探针出现性动态检测）→ PASS 自动接 60k（容忍度沿用 5.0/0.005，wpd 2，带崩溃 watchdog）。
4. **60k 判读重点**（reward 跨 ADR-013 不可比；判形状不判绝对值）：①曲线是**正常 RL 曲线、不再 ep30000 后冻结**；②fusion 稳定在某个**正余量**水平（不是 0 也不是 35），健康日志的 `fusion=` 不再单调冲顶；③P3 占比保持 >0（watchdog 不触发）；④`fusion_count_b2/b4/b5` 看是否某块类型(尤其 b4)异常爬升；⑤best ≥ 无融合上限 39.5；⑥entropy/clip 不长期为 0。**若稳定最优只采纳少量 fusion（barrier 留了 ~1.8σ 余量、合理），可把 MARGIN_REF 调小（0.15）再跑——这是预期内的激进度权衡，不是失败。**

### 关键事实（给人看的）

- **log-barrier 为何根治两种崩溃**：satisfied 侧斜率 SAT/mu→∞ 当 mu→0 ⇒ cost(随 fusion 升)+barrier 必在正余量内点出现峰值 → 不会冲过(治热崩溃)；violated 侧线性不平台 ⇒ 任何深度都有指回可行域的梯度 → 能爬回(治"无恢复")。冷崩溃(不 fuse)也被同一峰值机制解决：峰值 > 零融合基线。
- **成本权重未动**（用户决定 80:150:130:40）：barrier 是唯一回正力，足以压住 block4 的 130（峰值在正余量处 ⇒ 不会一路 fuse 到 block4 毁精度区）。
- **barrier 不破硬优先级**：priority/rank-key/选择逐位不变；barrier 只改 PPO 标量；violated 段恒 < P3 下限 40 且 P1 不吃 cost ⇒ item7 保持。纯 metrics 函数 ⇒ 1==N 不受影响。`acc_barrier_enabled=False` 回退 ADR-012。
- **崩溃 watchdog**：sequential 路径不检查 STOP_RL，故 watchdog 用 SIGTERM→SIGKILL（best/diagnostics 已周期原子写盘，杀掉只丢最近 <200 回合可恢复态）。判据 = 连续 12 个 30min 窗口 P3 占比 <2%（持续无 P3 的死亡签名）；健康曲线有 P3 时永不触发。
- `--blb-v3-reward-devices`（K-split）不再使用；`--stage2-rl-devices`（互斥）；`KTRIALS` 固定 5；噪声/策略/更新/复测按 (seed, 全局episode, …) 键控。
- **workers-per-device / scheduling（提速，画像驱动）**：2026-06-23 profile-off A/B 结果显示最优稳定配置是 `--stage2-workers-per-device 1` + `BLB_STAGE2_POLICY_DEVICE=worker` + `BLB_STAGE2_DYNAMIC_ASSIGNMENT=1`，1GPU/5GPU 效果与 PPO updates matched，端到端约 `3.535x`。`wpd=2` 会把同卡 terminal probe 均值抬到约 `5.6s/episode`，端到端降到约 `2.804x`；CPU-policy 单卡绝对吞吐也低于 worker-policy 5GPU。因此默认 60k 与确定性门禁使用上述最优配置，CPU-policy / static assignment / wpd=2 仅作为显式 override 的诊断候选。

### 预期产物

- `$OUT/selfcheck.txt`（ADR-013 barrier + ADR-012 legacy 断言）+ `unittest_*.log` ×4（含 log_barrier）
- `$OUT/map_gate.txt`、`$OUT/stage2_ngpu_gate/`（g1/gN sigs、sig_diff、episodes.jsonl ×2、verdict.txt 含动态探针判读）
- `$OUT/long60k/run/`（diagnostics 全套，无 .pt）+ `long60k_health.log`（含 per-type fusion + watchdog 状态）+ `collapse_streak.txt`

### 幂等 / 安全

- Stage-1 record 合成幂等。门禁短跑各自 `--run-tag gate_${tag}_${TS}` + `--fresh`，保留在
  `Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.001_s2t0.001_s2st3.0__gate_*`，
  不覆盖正式 60k canonical slug。
- 60k 用 `--fresh`；若 agent 会话中断，训练进程(nohup)继续，下次触发只做产物回收。
- 门禁任何 FAIL 都不启动 60k。

### 历史（已完成，供参考）

- 2026-06-13：ADR-012 修复后第 3 次 60k 跑满（artifacts `stage2_grid_gate_60k_20260612_191530`，1==5 PASS、3.5×）翻转成**热崩溃**（fusion 1.4→35、后 30k 冻结 -6.95）→ 诊断 = 近界档拆刹车 + 单调 fusion 激励无回正力 + 平崖无恢复梯度 → 本轮 ADR-013 log-barrier。
- 2026-06-12：ADR-011 修复后第 2 次 60k 跑满（artifacts `stage2_grid_gate_60k_20260612_004130`，1==5 PASS、3.62×、4333 ep/h）仍 fusion=0 → 取证定位 P1 悬崖税/探针 K 抵消/entropy 冻结 → ADR-012（near-miss tier 被 013 取代）。
- 2026-06-11：step-1×15 全量重建 6 图 + 确定性门禁 PASS（1==5 逐字、3.62×）+ 第 1 次 60k fusion=0 坍缩（artifacts `stage2_grid_gate_60k_20260611_031751`，诊断→ADR-011）。
- 2026-06-10：5 卡 A/B 完成（artifacts `stage2_rebuild_ab_20260610_013046`，提交 0457fc0）；stageA Stage-1 1vs5 确定性 PASS、3.75×。
- 2026-06-07~09：fusion 图按新 replan 策略重建（1ad078c）；A/B 启动崩溃修复（c16d0f7：合成 MRPC Stage-1 record）；Stage-1 多卡确定性+提速（a1cf152/15c16ad）。
- 2026-06-05~06：fusion 课程上线（ed797b1）、degree-0 停用（469474d）、RO 默认融合策略（957ff7a 轮）。
