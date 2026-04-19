#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
用法：
  bash llama_7B_LayerImportance.sh [可选参数]

预设系统：
  --preset NAME              加载 presets/NAME.conf 中的参数（命令行参数优先覆盖预设）
  --list-presets             列出所有可用预设

核心参数：
  --dataset DATASET
  --search-algorithm rl|ga|general-rl|rl-and-ga-compare
  --logfile FILE
  --model-type bert-base|bert-large|gpt-2
  --batch-size N

普通 RL / 对比实验中的 RL：
  --stage1-search-episodes N
  --stage2-search-episodes N
  --ppo-update-interval N              每多少 episode 触发一次 PPO 更新（默认 120）；
                                       同时决定每个 details/txt 的回合数（= 3 × N, 默认 360）

GA / 对比实验中的 GA：
  --stage1-search-generations N
  --stage2-search-generations N

准确度约束参数（rl / ga / rl-and-ga-compare 可用）：
  --stage1-accuracy-tolerance FLOAT    Stage-1 指标约束百分比（默认 0.005 即 0.5%）
  --stage2-limit-tolerance FLOAT       Stage-2 指标约束百分比（以 baseline 为基准，默认 0.05 即 5%；loss 允许上浮 5%、metric 允许下降 5%）
  --stage2-stability-tolerance FLOAT   Stage-2 稳定性约束百分比（以 baseline 探针的 std 为基准，默认 0.05 即 5%）
  --stage2-k-trials INT                Stage-2 稳定性评测噪声试验次数 K（默认 5；每次评测在同一份探针上跑 K 个独立噪声种子）
  --stage2-probe-size INT              Stage-2 稳定性评测探针子集大小（默认 256；用分层采样从验证集中抽取 K 次 trial 共用的固定子集）

持久化与续训练（rl / ga 可用）：
  --fresh-start                        从头开始训练（首次运行必须指定）
  --fresh-stage1                       仅重置 Stage-1 数据（保留 Stage-2）
  --fresh-stage2                       仅重置 Stage-2 数据（保留 Stage-1）

普通 RL / GA 共用：
  --skip-stage1-search
  --skip-noise-search
  --skip-final-eval                       跳过 Stage-1 + Stage-2 合并的最终评估
  --final-eval-source search|json|manual  最终评估的配置来源
  --final-eval-config PATH                final-eval-source=json 时的合并 JSON 路径
  --manual-stage1-gelu JSON_ARRAY         manual 模式：Stage-1 GELU 多项式次数
  --manual-stage1-softmax JSON_ARRAY      manual 模式：Stage-1 Softmax 多项式次数
  --manual-stage2-noise JSON_OBJECT       manual 模式：Stage-2 噪声系数（x/wq/wk/wv/wo/wffn1/wffn2）
  --final-eval-repeat N                   最终评估的重复次数
  --stage2-fixed-config-source stage1_result|json|manual
  --stage2-fixed-config PATH
  --stage2-manual-gelu JSON_ARRAY
  --stage2-manual-softmax JSON_ARRAY
  --random-seed N
  --perm-trials N
  --cost-trials N
  --budget-trials N
  --resume-from PATH

对比实验专用（仅 rl-and-ga-compare 可用）：
  --compare-config-mode direct|persistent
  --stage2-compare-repeats N
  --compare-persistent-root PATH
  --rl-compare-stage1-json PATH
  --rl-compare-stage2-json PATH
  --ga-compare-stage1-json PATH
  --ga-compare-stage2-json PATH
  --rl-compare-stage1-accuracy-tolerance FLOAT
  --rl-compare-stage2-limit-tolerance FLOAT
  --rl-compare-stage2-stability-tolerance FLOAT
  --ga-compare-stage1-accuracy-tolerance FLOAT
  --ga-compare-stage2-limit-tolerance FLOAT
  --ga-compare-stage2-stability-tolerance FLOAT

普通 RL（仅 rl 可用）：
  --stage1-search-lr FLOAT
  --stage2-search-lr FLOAT

通用 RL（仅 general-rl 可用）：
  --general-rl-mode train|search          训练 或 搜索（search 等同于原 infer）
  --general-rl-tasks TASK1,TASK2,...
  --general-rl-rounds N
  --ppo-update-interval N                 同时决定"每轮每任务的 episode 数"与 PPO 更新间隔
                                          （两者恒等；默认 120）
  --general-rl-lr FLOAT
  --general-rl-num-rollouts N
  --general-rl-greedy
  --general-stage1-policy PATH            搜索模式：显式指定 Stage-1 策略文件
  --general-stage2-policy PATH            搜索模式：显式指定 Stage-2 策略文件
  --general-policy-dir PATH               搜索模式：指定训练持久化目录，自动推导策略文件
  --general-rl-skip-stage2
  --general-rl-stage1-config-json PATH
  --general-rl-accuracy-tolerances T1,T2,...
  --general-rl-accuracy-tolerance-range MIN,MAX
  --fresh-start                           通用 RL 训练模式同样支持

持久化目录结构（general-rl train 自动管理）：
  rl_results/persistent/general-rl/{model_type}/{taskset}/{accuracy_slug}/
  - 同一数据集 + 同一准确度区间 → 同一目录（自动续训练）
  - 不同区间 → 不同目录
  - accuracy_slug 示例：range_0.50pct_2.00pct / discrete_0.50pct_1.00pct / default

说明：
  1. 不再支持位置参数，统一改为可选参数。
  2. 参数 --model 已废弃，请改用 --dataset。
  3. 以下旧参数已从命令行入口移除，因为当前流程不会实际生效：
     lora_r、lora_alpha、degree
 4. --search-algorithm=rl-and-ga-compare 现在直接比较已有 JSON 或持久化目录结果，
     不再重新启动完整 RL / GA 训练。
  5. compare 的 persistent 模式会在启动前检查 RL / GA 目标持久化目录与 metadata.json，
     如果找不到会直接报错并打印对应路径。

示例：
  bash llama_7B_LayerImportance.sh --list-presets
  bash llama_7B_LayerImportance.sh --preset mrpc-rl-default --fresh-start
  bash llama_7B_LayerImportance.sh --preset mrpc-rl-default            # 续训练（无需 --fresh-start）
  bash llama_7B_LayerImportance.sh --dataset mrpc
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm rl --stage1-search-lr 3e-5 --stage2-search-lr 1e-5
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm ga --stage1-search-generations 120 --stage2-search-generations 90
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm ga --skip-stage1-search --skip-final-eval --stage2-search-generations 2500 --stage2-fixed-config-source json --stage2-fixed-config glue_final_configs_best_genetic.json
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm rl-and-ga-compare --compare-config-mode direct --rl-compare-stage1-json glue_final_configs_best_ppo.json --rl-compare-stage2-json glue_final_configs_best_ppo.json --ga-compare-stage1-json glue_final_configs_best_genetic.json --ga-compare-stage2-json glue_final_configs_best_genetic.json
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm general-rl --general-rl-mode train --general-rl-tasks mrpc,cola,rte,stsb --fresh-start
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm general-rl --general-rl-mode search --general-policy-dir rl_results/persistent/general-rl/bert-base/cola_mrpc_rte_stsb/default
EOF
}

err(){ echo "错误：$1" >&2; exit 1; }
needv(){ [ "$#" -ge 2 ] || err "选项 $1 缺少取值。"; }
is_pos_int(){ [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
is_pos_num(){ [[ "$1" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]] && awk -v x="$1" 'BEGIN { exit !((x + 0) > 0) }'; }
ga_total_layers_for_model_type(){
  case "$1" in
    bert-base|gpt-2) echo 12 ;;
    bert-large) echo 24 ;;
    *) err "无法为模型类型推断 GA 默认代数：$1" ;;
  esac
}
ga_default_generations_from_episodes(){
  local episode_budget="$1"
  local population_size="$2"
  echo $(( (episode_budget + population_size - 1) / population_size ))
}
ga_default_stage1_generations_for_model(){
  local layers pop
  layers="$(ga_total_layers_for_model_type "$1")"
  pop=$(( layers * 2 ))
  [ "$pop" -ge 32 ] || pop=32
  ga_default_generations_from_episodes 51000 "$pop"
}
ga_default_stage2_generations_for_model(){
  local layers pop
  layers="$(ga_total_layers_for_model_type "$1")"
  pop="$layers"
  [ "$pop" -ge 32 ] || pop=32
  ga_default_generations_from_episodes 40000 "$pop"
}
normalize_taskset_id(){
  local raw="$1"
  local old_ifs token out=""
  old_ifs="$IFS"
  IFS=','
  read -r -a __task_items <<< "$raw"
  IFS="$old_ifs"
  for token in "${__task_items[@]}"; do
    token="$(printf '%s' "$token" | tr '[:upper:]' '[:lower:]' | xargs)"
    token="${token//[^a-z0-9_-]/_}"
    token="${token//-/_}"
    [ -z "$token" ] && continue
    if [ -z "$out" ]; then
      out="$token"
    else
      out="${out}_${token}"
    fi
  done
  [ -n "$out" ] || err "无法从任务列表生成 taskset 标识：$raw"
  echo "$out"
}
origin(){ [ "$1" = "true" ] && echo "显式指定" || echo "使用默认值"; }
show(){ echo "  $1：$2（$(origin "$3")）"; }
boolzh(){ [ "$1" = "true" ] && echo "是" || echo "否"; }
algzh(){
  case "$1" in
    rl) echo "普通强化学习（rl）" ;;
    ga) echo "遗传算法（ga）" ;;
    general-rl) echo "通用强化学习（general-rl）" ;;
    rl-and-ga-compare) echo "普通 RL 与 GA 对比实验（rl-and-ga-compare）" ;;
    *) echo "$1" ;;
  esac
}
srczh(){
  case "$1" in
    search) echo "搜索结果（search）" ;;
    stage1_result) echo "Stage-1 搜索结果（stage1_result）" ;;
    json) echo "JSON 文件（json）" ;;
    manual) echo "手动指定（manual）" ;;
    *) echo "$1" ;;
  esac
}
modelzh(){
  case "$1" in
    bert-base) echo "BERT Base（bert-base）" ;;
    bert-large) echo "BERT Large（bert-large）" ;;
    gpt-2) echo "GPT-2（gpt-2）" ;;
    *) echo "$1" ;;
  esac
}
infer_family(){
  local x
  x="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "$x" == *genetic* || "$x" == *_ga.* || "$x" == *-ga.* || "$x" == ga_* ]]; then
    echo ga
  elif [[ "$x" == *ppo* ]]; then
    echo rl
  else
    echo unknown
  fi
}
default_final_eval_json_for_family(){
  [ "$1" = "ga" ] && echo "glue_final_configs_best_genetic.json" || echo "glue_final_configs_best_ppo.json"
}

resolve_compare_cuda_split() {
  RL_COMPARE_CUDA_VISIBLE_DEVICES=""
  GA_COMPARE_CUDA_VISIBLE_DEVICES=""
  COMPARE_CUDA_NOTE="未设置 CUDA_VISIBLE_DEVICES；两个子进程将继承当前环境。"
  local raw="${CUDA_VISIBLE_DEVICES:-}"
  if [ -z "$raw" ]; then
    return
  fi
  IFS=',' read -r -a __cuda_items <<< "$raw"
  local cleaned=()
  local item
  for item in "${__cuda_items[@]}"; do
    item="$(printf '%s' "$item" | xargs)"
    [ -n "$item" ] && cleaned+=("$item")
  done
  if [ "${#cleaned[@]}" -ge 2 ]; then
    RL_COMPARE_CUDA_VISIBLE_DEVICES="${cleaned[0]}"
    GA_COMPARE_CUDA_VISIBLE_DEVICES="${cleaned[1]}"
    COMPARE_CUDA_NOTE="已自动拆分 CUDA_VISIBLE_DEVICES：RL=${RL_COMPARE_CUDA_VISIBLE_DEVICES}，GA=${GA_COMPARE_CUDA_VISIBLE_DEVICES}"
  elif [ "${#cleaned[@]}" -eq 1 ]; then
    RL_COMPARE_CUDA_VISIBLE_DEVICES="${cleaned[0]}"
    GA_COMPARE_CUDA_VISIBLE_DEVICES="${cleaned[0]}"
    COMPARE_CUDA_NOTE="仅检测到 1 个可见设备：RL 与 GA 将共享 CUDA_VISIBLE_DEVICES=${cleaned[0]}"
  fi
}

DATASET="mrpc"; S_DATASET="false"
SEARCH_ALGORITHM="rl"; S_SEARCH_ALGORITHM="false"
LOGFILE="output.log"; S_LOGFILE="false"
MODEL_TYPE="bert-base"; S_MODEL_TYPE="false"
BATCH_SIZE="16"; S_BATCH_SIZE="false"
STAGE1_EPISODES="51000"; S_STAGE1_EPISODES="false"
STAGE2_EPISODES="40000"; S_STAGE2_EPISODES="false"
STAGE1_GENERATIONS=""; S_STAGE1_GENERATIONS="false"
STAGE2_GENERATIONS=""; S_STAGE2_GENERATIONS="false"
STAGE1_LR="1e-4"; S_STAGE1_LR="false"
STAGE2_LR="1e-4"; S_STAGE2_LR="false"
SKIP_STAGE1_SEARCH="false"; S_SKIP_STAGE1_SEARCH="false"
SKIP_NOISE_SEARCH="false"; S_SKIP_NOISE_SEARCH="false"
SKIP_FINAL_EVAL="false"; S_SKIP_FINAL_EVAL="false"
FINAL_EVAL_SOURCE="search"; S_FINAL_EVAL_SOURCE="false"
FINAL_EVAL_CONFIG=""; S_FINAL_EVAL_CONFIG="false"
MANUAL_STAGE1_GELU=""
MANUAL_STAGE1_SOFTMAX=""
MANUAL_STAGE2_NOISE=""
STAGE2_FIXED_CONFIG_SOURCE=""; S_STAGE2_FIXED_CONFIG_SOURCE="false"
STAGE2_FIXED_CONFIG=""; S_STAGE2_FIXED_CONFIG="false"
STAGE2_MANUAL_GELU=""
STAGE2_MANUAL_SOFTMAX=""
FINAL_EVAL_REPEAT="1"; S_FINAL_EVAL_REPEAT="false"
STAGE2_COMPARE_REPEATS=""; S_STAGE2_COMPARE_REPEATS="false"
RANDOM_SEED="42"; S_RANDOM_SEED="false"
PERM_TRIALS="10"; S_PERM_TRIALS="false"
COST_TRIALS="10"; S_COST_TRIALS="false"
BUDGET_TRIALS="10"; S_BUDGET_TRIALS="false"
GENERAL_MODE="infer"; S_GENERAL_MODE="false"
GENERAL_TASKS=""; S_GENERAL_TASKS="false"
GENERAL_ROUNDS="50"; S_GENERAL_ROUNDS="false"
PPO_UPDATE_INTERVAL_VAL="120"; S_PPO_UPDATE_INTERVAL="false"
GENERAL_LR="3e-5"; S_GENERAL_LR="false"
GENERAL_NUM_ROLLOUTS="500"; S_GENERAL_NUM_ROLLOUTS="false"
GENERAL_GREEDY="false"; S_GENERAL_GREEDY="false"
GENERAL_STAGE1_POLICY=""; S_GENERAL_STAGE1_POLICY="false"
GENERAL_STAGE2_POLICY=""; S_GENERAL_STAGE2_POLICY="false"
GENERAL_POLICY_DIR=""; S_GENERAL_POLICY_DIR="false"
GENERAL_SKIP_STAGE2="false"; S_GENERAL_SKIP_STAGE2="false"
GENERAL_STAGE1_CONFIG_JSON=""; S_GENERAL_STAGE1_CONFIG_JSON="false"
GENERAL_ACCURACY_TOLERANCES=""; S_GENERAL_ACCURACY_TOLERANCES="false"
GENERAL_ACCURACY_TOLERANCE_RANGE=""; S_GENERAL_ACCURACY_TOLERANCE_RANGE="false"
RL_COMPARE_SKIP_STAGE1_SEARCH="false"; S_RL_COMPARE_SKIP_STAGE1_SEARCH="false"
GA_COMPARE_SKIP_STAGE1_SEARCH="false"; S_GA_COMPARE_SKIP_STAGE1_SEARCH="false"
RL_COMPARE_FINAL_EVAL_SOURCE="search"; S_RL_COMPARE_FINAL_EVAL_SOURCE="false"
GA_COMPARE_FINAL_EVAL_SOURCE="search"; S_GA_COMPARE_FINAL_EVAL_SOURCE="false"
RL_COMPARE_FINAL_EVAL_CONFIG=""; S_RL_COMPARE_FINAL_EVAL_CONFIG="false"
GA_COMPARE_FINAL_EVAL_CONFIG=""; S_GA_COMPARE_FINAL_EVAL_CONFIG="false"
RL_COMPARE_SKIP_NOISE_SEARCH="false"; S_RL_COMPARE_SKIP_NOISE_SEARCH="false"
GA_COMPARE_SKIP_NOISE_SEARCH="false"; S_GA_COMPARE_SKIP_NOISE_SEARCH="false"
COMPARE_CONFIG_MODE="direct"; S_COMPARE_CONFIG_MODE="false"
COMPARE_PERSISTENT_ROOT="rl_results/persistent"; S_COMPARE_PERSISTENT_ROOT="false"
RL_COMPARE_STAGE1_JSON=""; S_RL_COMPARE_STAGE1_JSON="false"
RL_COMPARE_STAGE2_JSON=""; S_RL_COMPARE_STAGE2_JSON="false"
GA_COMPARE_STAGE1_JSON=""; S_GA_COMPARE_STAGE1_JSON="false"
GA_COMPARE_STAGE2_JSON=""; S_GA_COMPARE_STAGE2_JSON="false"
RL_COMPARE_STAGE1_ACCURACY_TOLERANCE=""; S_RL_COMPARE_STAGE1_ACCURACY_TOLERANCE="false"
RL_COMPARE_STAGE2_LIMIT_TOLERANCE=""; S_RL_COMPARE_STAGE2_LIMIT_TOLERANCE="false"
RL_COMPARE_STAGE2_STABILITY_TOLERANCE=""; S_RL_COMPARE_STAGE2_STABILITY_TOLERANCE="false"
GA_COMPARE_STAGE1_ACCURACY_TOLERANCE=""; S_GA_COMPARE_STAGE1_ACCURACY_TOLERANCE="false"
GA_COMPARE_STAGE2_LIMIT_TOLERANCE=""; S_GA_COMPARE_STAGE2_LIMIT_TOLERANCE="false"
GA_COMPARE_STAGE2_STABILITY_TOLERANCE=""; S_GA_COMPARE_STAGE2_STABILITY_TOLERANCE="false"
RESUME_FROM=""; S_RESUME_FROM="false"
STAGE1_ACCURACY_TOLERANCE="0.005"; S_STAGE1_ACCURACY_TOLERANCE="false"
STAGE2_LIMIT_TOLERANCE="0.05"; S_STAGE2_LIMIT_TOLERANCE="false"
STAGE2_STABILITY_TOLERANCE="0.05"; S_STAGE2_STABILITY_TOLERANCE="false"
STAGE2_K_TRIALS="5"; S_STAGE2_K_TRIALS="false"
STAGE2_PROBE_SIZE="256"; S_STAGE2_PROBE_SIZE="false"
FRESH_START="false"; S_FRESH_START="false"
FRESH_STAGE1="false"; S_FRESH_STAGE1="false"
FRESH_STAGE2="false"; S_FRESH_STAGE2="false"

# ── 预设支持（--preset）──────────────────────────────────────────────
# 用法: bash llama_7B_LayerImportance.sh --preset <预设名> [额外参数...]
# 预设文件位于 presets/<预设名>.conf，内容为每行一个命令行参数。
# 命令行中后续传入的参数会覆盖预设中的同名参数。
PRESET_ARGS=()
_raw_args=("$@")
_new_args=()
_i=0
while [ $_i -lt ${#_raw_args[@]} ]; do
  if [ "${_raw_args[$_i]}" = "--preset" ]; then
    _i=$((_i + 1))
    [ $_i -lt ${#_raw_args[@]} ] || err "选项 --preset 缺少取值。"
    _preset_name="${_raw_args[$_i]}"
    _preset_file="$(cd "$(dirname "$0")" && pwd)/presets/${_preset_name}.conf"
    [ -f "$_preset_file" ] || err "预设文件不存在：$_preset_file\n可用预设：$(ls "$(cd "$(dirname "$0")" && pwd)/presets/" 2>/dev/null | sed 's/\.conf$//' | tr '\n' ' ')"
    # 读取预设文件（忽略空行和注释）
    while IFS= read -r _pline; do
      _pline="$(printf '%s' "$_pline" | sed 's/#.*//' | xargs)"
      [ -n "$_pline" ] || continue
      PRESET_ARGS+=($_pline)
    done < "$_preset_file"
    echo "已加载预设：${_preset_name}（${_preset_file}）"
  else
    _new_args+=("${_raw_args[$_i]}")
  fi
  _i=$((_i + 1))
done
# 预设参数放在前面，命令行参数放在后面（后面的覆盖前面的）
set -- "${PRESET_ARGS[@]+"${PRESET_ARGS[@]}"}" "${_new_args[@]+"${_new_args[@]}"}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --list-presets)
      _presets_dir="$(cd "$(dirname "$0")" && pwd)/presets"
      echo "可用预设（presets/ 目录）："
      if [ -d "$_presets_dir" ]; then
        for _pf in "$_presets_dir"/*.conf; do
          [ -f "$_pf" ] || continue
          _pn="$(basename "$_pf" .conf)"
          _pd="$(head -1 "$_pf" | sed 's/^#\s*//')"
          printf "  %-30s %s\n" "$_pn" "$_pd"
        done
      else
        echo "  （目录不存在）"
      fi
      exit 0 ;;
    --preset) shift 2 ;;  # 已在预处理阶段处理，此处跳过
    --dataset) needv "$@"; DATASET="$2"; S_DATASET="true"; shift 2 ;;
    --model) err "参数 --model 已废弃，请改用 --dataset。" ;;
    --search-algorithm) needv "$@"; SEARCH_ALGORITHM="$2"; S_SEARCH_ALGORITHM="true"; shift 2 ;;
    --logfile) needv "$@"; LOGFILE="$2"; S_LOGFILE="true"; shift 2 ;;
    --model-type) needv "$@"; MODEL_TYPE="$2"; S_MODEL_TYPE="true"; shift 2 ;;
    --batch-size) needv "$@"; BATCH_SIZE="$2"; S_BATCH_SIZE="true"; shift 2 ;;
    --stage1-search-episodes) needv "$@"; STAGE1_EPISODES="$2"; S_STAGE1_EPISODES="true"; shift 2 ;;
    --stage2-search-episodes) needv "$@"; STAGE2_EPISODES="$2"; S_STAGE2_EPISODES="true"; shift 2 ;;
    --stage1-search-generations) needv "$@"; STAGE1_GENERATIONS="$2"; S_STAGE1_GENERATIONS="true"; shift 2 ;;
    --stage2-search-generations) needv "$@"; STAGE2_GENERATIONS="$2"; S_STAGE2_GENERATIONS="true"; shift 2 ;;
    --stage1-search-lr) needv "$@"; STAGE1_LR="$2"; S_STAGE1_LR="true"; shift 2 ;;
    --stage2-search-lr) needv "$@"; STAGE2_LR="$2"; S_STAGE2_LR="true"; shift 2 ;;
    --skip-stage1-search) SKIP_STAGE1_SEARCH="true"; S_SKIP_STAGE1_SEARCH="true"; shift ;;
    --skip-noise-search) SKIP_NOISE_SEARCH="true"; S_SKIP_NOISE_SEARCH="true"; shift ;;
    --skip-final-eval) SKIP_FINAL_EVAL="true"; S_SKIP_FINAL_EVAL="true"; shift ;;
    --final-eval-source) needv "$@"; FINAL_EVAL_SOURCE="$2"; S_FINAL_EVAL_SOURCE="true"; shift 2 ;;
    --final-eval-config) needv "$@"; FINAL_EVAL_CONFIG="$2"; S_FINAL_EVAL_CONFIG="true"; shift 2 ;;
    --manual-stage1-gelu) needv "$@"; MANUAL_STAGE1_GELU="$2"; shift 2 ;;
    --manual-stage1-softmax) needv "$@"; MANUAL_STAGE1_SOFTMAX="$2"; shift 2 ;;
    --manual-stage2-noise) needv "$@"; MANUAL_STAGE2_NOISE="$2"; shift 2 ;;
    --stage2-fixed-config-source) needv "$@"; STAGE2_FIXED_CONFIG_SOURCE="$2"; S_STAGE2_FIXED_CONFIG_SOURCE="true"; shift 2 ;;
    --stage2-fixed-config) needv "$@"; STAGE2_FIXED_CONFIG="$2"; S_STAGE2_FIXED_CONFIG="true"; shift 2 ;;
    --stage2-manual-gelu) needv "$@"; STAGE2_MANUAL_GELU="$2"; shift 2 ;;
    --stage2-manual-softmax) needv "$@"; STAGE2_MANUAL_SOFTMAX="$2"; shift 2 ;;
    --final-eval-repeat) needv "$@"; FINAL_EVAL_REPEAT="$2"; S_FINAL_EVAL_REPEAT="true"; shift 2 ;;
    --stage2-compare-repeats) needv "$@"; STAGE2_COMPARE_REPEATS="$2"; S_STAGE2_COMPARE_REPEATS="true"; shift 2 ;;
    --random-seed) needv "$@"; RANDOM_SEED="$2"; S_RANDOM_SEED="true"; shift 2 ;;
    --perm-trials) needv "$@"; PERM_TRIALS="$2"; S_PERM_TRIALS="true"; shift 2 ;;
    --cost-trials) needv "$@"; COST_TRIALS="$2"; S_COST_TRIALS="true"; shift 2 ;;
    --budget-trials) needv "$@"; BUDGET_TRIALS="$2"; S_BUDGET_TRIALS="true"; shift 2 ;;
    --general-rl-mode) needv "$@"; GENERAL_MODE="$2"; S_GENERAL_MODE="true"; shift 2 ;;
    --general-rl-tasks) needv "$@"; GENERAL_TASKS="$2"; S_GENERAL_TASKS="true"; shift 2 ;;
    --general-rl-rounds) needv "$@"; GENERAL_ROUNDS="$2"; S_GENERAL_ROUNDS="true"; shift 2 ;;
    --ppo-update-interval) needv "$@"; PPO_UPDATE_INTERVAL_VAL="$2"; S_PPO_UPDATE_INTERVAL="true"; shift 2 ;;
    --general-rl-lr) needv "$@"; GENERAL_LR="$2"; S_GENERAL_LR="true"; shift 2 ;;
    --general-rl-num-rollouts) needv "$@"; GENERAL_NUM_ROLLOUTS="$2"; S_GENERAL_NUM_ROLLOUTS="true"; shift 2 ;;
    --general-rl-greedy) GENERAL_GREEDY="true"; S_GENERAL_GREEDY="true"; shift ;;
    --general-stage1-policy) needv "$@"; GENERAL_STAGE1_POLICY="$2"; S_GENERAL_STAGE1_POLICY="true"; shift 2 ;;
    --general-stage2-policy) needv "$@"; GENERAL_STAGE2_POLICY="$2"; S_GENERAL_STAGE2_POLICY="true"; shift 2 ;;
    --general-policy-dir) needv "$@"; GENERAL_POLICY_DIR="$2"; S_GENERAL_POLICY_DIR="true"; shift 2 ;;
    --general-rl-skip-stage2) GENERAL_SKIP_STAGE2="true"; S_GENERAL_SKIP_STAGE2="true"; shift ;;
    --general-rl-stage1-config-json) needv "$@"; GENERAL_STAGE1_CONFIG_JSON="$2"; S_GENERAL_STAGE1_CONFIG_JSON="true"; shift 2 ;;
    --general-rl-accuracy-tolerances) needv "$@"; GENERAL_ACCURACY_TOLERANCES="$2"; S_GENERAL_ACCURACY_TOLERANCES="true"; shift 2 ;;
    --general-rl-accuracy-tolerance-range) needv "$@"; GENERAL_ACCURACY_TOLERANCE_RANGE="$2"; S_GENERAL_ACCURACY_TOLERANCE_RANGE="true"; shift 2 ;;
    --rl-skip-stage1-search) RL_COMPARE_SKIP_STAGE1_SEARCH="true"; S_RL_COMPARE_SKIP_STAGE1_SEARCH="true"; shift ;;
    --ga-skip-stage1-search) GA_COMPARE_SKIP_STAGE1_SEARCH="true"; S_GA_COMPARE_SKIP_STAGE1_SEARCH="true"; shift ;;
    --rl-final-eval-source) needv "$@"; RL_COMPARE_FINAL_EVAL_SOURCE="$2"; S_RL_COMPARE_FINAL_EVAL_SOURCE="true"; shift 2 ;;
    --ga-final-eval-source) needv "$@"; GA_COMPARE_FINAL_EVAL_SOURCE="$2"; S_GA_COMPARE_FINAL_EVAL_SOURCE="true"; shift 2 ;;
    --rl-final-eval-config) needv "$@"; RL_COMPARE_FINAL_EVAL_CONFIG="$2"; S_RL_COMPARE_FINAL_EVAL_CONFIG="true"; shift 2 ;;
    --ga-final-eval-config) needv "$@"; GA_COMPARE_FINAL_EVAL_CONFIG="$2"; S_GA_COMPARE_FINAL_EVAL_CONFIG="true"; shift 2 ;;
    --rl-skip-noise-search) RL_COMPARE_SKIP_NOISE_SEARCH="true"; S_RL_COMPARE_SKIP_NOISE_SEARCH="true"; shift ;;
    --ga-skip-noise-search) GA_COMPARE_SKIP_NOISE_SEARCH="true"; S_GA_COMPARE_SKIP_NOISE_SEARCH="true"; shift ;;
    --compare-config-mode) needv "$@"; COMPARE_CONFIG_MODE="$2"; S_COMPARE_CONFIG_MODE="true"; shift 2 ;;
    --compare-persistent-root) needv "$@"; COMPARE_PERSISTENT_ROOT="$2"; S_COMPARE_PERSISTENT_ROOT="true"; shift 2 ;;
    --rl-compare-stage1-json) needv "$@"; RL_COMPARE_STAGE1_JSON="$2"; S_RL_COMPARE_STAGE1_JSON="true"; shift 2 ;;
    --rl-compare-stage2-json) needv "$@"; RL_COMPARE_STAGE2_JSON="$2"; S_RL_COMPARE_STAGE2_JSON="true"; shift 2 ;;
    --ga-compare-stage1-json) needv "$@"; GA_COMPARE_STAGE1_JSON="$2"; S_GA_COMPARE_STAGE1_JSON="true"; shift 2 ;;
    --ga-compare-stage2-json) needv "$@"; GA_COMPARE_STAGE2_JSON="$2"; S_GA_COMPARE_STAGE2_JSON="true"; shift 2 ;;
    --rl-compare-stage1-accuracy-tolerance) needv "$@"; RL_COMPARE_STAGE1_ACCURACY_TOLERANCE="$2"; S_RL_COMPARE_STAGE1_ACCURACY_TOLERANCE="true"; shift 2 ;;
    --rl-compare-stage2-limit-tolerance) needv "$@"; RL_COMPARE_STAGE2_LIMIT_TOLERANCE="$2"; S_RL_COMPARE_STAGE2_LIMIT_TOLERANCE="true"; shift 2 ;;
    --rl-compare-stage2-stability-tolerance) needv "$@"; RL_COMPARE_STAGE2_STABILITY_TOLERANCE="$2"; S_RL_COMPARE_STAGE2_STABILITY_TOLERANCE="true"; shift 2 ;;
    --ga-compare-stage1-accuracy-tolerance) needv "$@"; GA_COMPARE_STAGE1_ACCURACY_TOLERANCE="$2"; S_GA_COMPARE_STAGE1_ACCURACY_TOLERANCE="true"; shift 2 ;;
    --ga-compare-stage2-limit-tolerance) needv "$@"; GA_COMPARE_STAGE2_LIMIT_TOLERANCE="$2"; S_GA_COMPARE_STAGE2_LIMIT_TOLERANCE="true"; shift 2 ;;
    --ga-compare-stage2-stability-tolerance) needv "$@"; GA_COMPARE_STAGE2_STABILITY_TOLERANCE="$2"; S_GA_COMPARE_STAGE2_STABILITY_TOLERANCE="true"; shift 2 ;;
    --resume-from) needv "$@"; RESUME_FROM="$2"; S_RESUME_FROM="true"; shift 2 ;;
    --stage1-accuracy-tolerance) needv "$@"; STAGE1_ACCURACY_TOLERANCE="$2"; S_STAGE1_ACCURACY_TOLERANCE="true"; shift 2 ;;
    --stage2-limit-tolerance) needv "$@"; STAGE2_LIMIT_TOLERANCE="$2"; S_STAGE2_LIMIT_TOLERANCE="true"; shift 2 ;;
    --stage2-stability-tolerance) needv "$@"; STAGE2_STABILITY_TOLERANCE="$2"; S_STAGE2_STABILITY_TOLERANCE="true"; shift 2 ;;
    --stage2-k-trials) needv "$@"; STAGE2_K_TRIALS="$2"; S_STAGE2_K_TRIALS="true"; shift 2 ;;
    --stage2-probe-size) needv "$@"; STAGE2_PROBE_SIZE="$2"; S_STAGE2_PROBE_SIZE="true"; shift 2 ;;
    --fresh-start) FRESH_START="true"; S_FRESH_START="true"; shift ;;
    --fresh-stage1) FRESH_STAGE1="true"; S_FRESH_STAGE1="true"; shift ;;
    --fresh-stage2) FRESH_STAGE2="true"; S_FRESH_STAGE2="true"; shift ;;
    --*) err "不支持的参数：$1" ;;
    *) err "不再支持位置参数：$1。请改用 --dataset mrpc 这种写法。" ;;
  esac
done

DATASET="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')"
SEARCH_ALGORITHM="$(printf '%s' "$SEARCH_ALGORITHM" | tr '[:upper:]' '[:lower:]')"
MODEL_TYPE="$(printf '%s' "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')"
FINAL_EVAL_SOURCE="$(printf '%s' "$FINAL_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
STAGE2_FIXED_CONFIG_SOURCE="$(printf '%s' "$STAGE2_FIXED_CONFIG_SOURCE" | tr '[:upper:]' '[:lower:]')"
GENERAL_MODE="$(printf '%s' "$GENERAL_MODE" | tr '[:upper:]' '[:lower:]')"
RL_COMPARE_FINAL_EVAL_SOURCE="$(printf '%s' "$RL_COMPARE_FINAL_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
GA_COMPARE_FINAL_EVAL_SOURCE="$(printf '%s' "$GA_COMPARE_FINAL_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
COMPARE_CONFIG_MODE="$(printf '%s' "$COMPARE_CONFIG_MODE" | tr '[:upper:]' '[:lower:]')"

case "$SEARCH_ALGORITHM" in
  rl|ppo) SEARCH_ALGORITHM="rl" ;;
  ga|genetic) SEARCH_ALGORITHM="ga" ;;
  general-rl|general_rl|generalrl) SEARCH_ALGORITHM="general-rl" ;;
  rl-and-ga-compare|rl_and_ga_compare|compare) SEARCH_ALGORITHM="rl-and-ga-compare" ;;
  *) err "不支持的搜索算法：$SEARCH_ALGORITHM" ;;
esac

case "$MODEL_TYPE" in
  bert-base|bert_base|bertbase) MODEL_TYPE="bert-base" ;;
  bert-large|bert_large|bertlarge) MODEL_TYPE="bert-large" ;;
  gpt-2|gpt2|gpt_2) MODEL_TYPE="gpt-2" ;;
  *) err "不支持的模型类型：$MODEL_TYPE" ;;
esac
case "$FINAL_EVAL_SOURCE" in search|json|manual) ;; *) err "不支持的最终评估来源：$FINAL_EVAL_SOURCE" ;; esac
case "$STAGE2_FIXED_CONFIG_SOURCE" in ""|stage1_result|search|json|manual) ;; *) err "不支持的 Stage-2 固定 GELU/Softmax 来源：$STAGE2_FIXED_CONFIG_SOURCE" ;; esac
[ "$STAGE2_FIXED_CONFIG_SOURCE" = "search" ] && STAGE2_FIXED_CONFIG_SOURCE="stage1_result"
# search 是 infer 的别名
case "$GENERAL_MODE" in train|infer|search) ;; *) err "general-rl 模式必须是 train、search 或 infer，当前为：$GENERAL_MODE" ;; esac
[ "$GENERAL_MODE" = "infer" ] && GENERAL_MODE="search"

if [ "$S_STAGE1_GENERATIONS" = "false" ]; then
  STAGE1_GENERATIONS="$(ga_default_stage1_generations_for_model "$MODEL_TYPE")"
fi
if [ "$S_STAGE2_GENERATIONS" = "false" ]; then
  STAGE2_GENERATIONS="$(ga_default_stage2_generations_for_model "$MODEL_TYPE")"
fi

is_pos_int "$BATCH_SIZE" || err "--batch-size 必须是正整数，当前为：$BATCH_SIZE"
is_pos_int "$FINAL_EVAL_REPEAT" || err "--final-eval-repeat 必须是正整数，当前为：$FINAL_EVAL_REPEAT"
[ -z "$RESUME_FROM" ] || [ -d "$RESUME_FROM" ] || err "--resume-from 指定的目录不存在：$RESUME_FROM"
# 准确度约束参数校验
is_pos_num "$STAGE1_ACCURACY_TOLERANCE" || err "--stage1-accuracy-tolerance 必须是正数，当前为：$STAGE1_ACCURACY_TOLERANCE"
awk -v x="$STAGE1_ACCURACY_TOLERANCE" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "--stage1-accuracy-tolerance 必须 < 1（百分比形式如 0.005 表示 0.5%），当前为：$STAGE1_ACCURACY_TOLERANCE"
is_pos_num "$STAGE2_LIMIT_TOLERANCE" || err "--stage2-limit-tolerance 必须是正数，当前为：$STAGE2_LIMIT_TOLERANCE"
awk -v x="$STAGE2_LIMIT_TOLERANCE" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "--stage2-limit-tolerance 必须 < 1（百分比形式如 0.05 表示 5%），当前为：$STAGE2_LIMIT_TOLERANCE"
is_pos_num "$STAGE2_STABILITY_TOLERANCE" || err "--stage2-stability-tolerance 必须是正数，当前为：$STAGE2_STABILITY_TOLERANCE"
awk -v x="$STAGE2_STABILITY_TOLERANCE" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "--stage2-stability-tolerance 必须 < 1（百分比形式如 0.05 表示 5%），当前为：$STAGE2_STABILITY_TOLERANCE"
is_pos_int "$STAGE2_K_TRIALS" || err "--stage2-k-trials 必须是正整数，当前为：$STAGE2_K_TRIALS"
is_pos_int "$STAGE2_PROBE_SIZE" || err "--stage2-probe-size 必须是正整数，当前为：$STAGE2_PROBE_SIZE"

case "$DATASET" in
  mrpc|sst2|stsb|cola|qnli|rte|wnli) DATA_PATH="$DATASET" ;;
  *) err "不支持的数据集：$DATASET" ;;
esac

if [ "$SEARCH_ALGORITHM" != "general-rl" ] && [ "$SEARCH_ALGORITHM" != "rl-and-ga-compare" ]; then
  [ "$S_FINAL_EVAL_CONFIG" = "true" ] || FINAL_EVAL_CONFIG="$(default_final_eval_json_for_family "$SEARCH_ALGORITHM")"
  if [ "$S_STAGE2_FIXED_CONFIG_SOURCE" = "false" ] && [ -n "$STAGE2_MANUAL_GELU" -o -n "$STAGE2_MANUAL_SOFTMAX" ]; then
    STAGE2_FIXED_CONFIG_SOURCE="manual"
    [ "$S_STAGE2_FIXED_CONFIG" = "true" ] || STAGE2_FIXED_CONFIG="$FINAL_EVAL_CONFIG"
  elif [ "$S_STAGE2_FIXED_CONFIG_SOURCE" = "false" ] && [ "$S_STAGE2_FIXED_CONFIG" = "true" ]; then
    STAGE2_FIXED_CONFIG_SOURCE="json"
  elif [ "$S_STAGE2_FIXED_CONFIG_SOURCE" = "false" ] && [ "$S_STAGE2_FIXED_CONFIG" = "false" ] && [ -z "$STAGE2_MANUAL_GELU" ] && [ -z "$STAGE2_MANUAL_SOFTMAX" ]; then
    case "$FINAL_EVAL_SOURCE" in
      search) STAGE2_FIXED_CONFIG_SOURCE="stage1_result" ;;
      *) STAGE2_FIXED_CONFIG_SOURCE="$FINAL_EVAL_SOURCE" ;;
    esac
    STAGE2_FIXED_CONFIG="$FINAL_EVAL_CONFIG"
    STAGE2_MANUAL_GELU="$MANUAL_STAGE1_GELU"
    STAGE2_MANUAL_SOFTMAX="$MANUAL_STAGE1_SOFTMAX"
  else
    [ "$S_STAGE2_FIXED_CONFIG_SOURCE" = "true" ] || STAGE2_FIXED_CONFIG_SOURCE="stage1_result"
    [ "$S_STAGE2_FIXED_CONFIG" = "true" ] || STAGE2_FIXED_CONFIG="$FINAL_EVAL_CONFIG"
  fi
fi

if [ "$SEARCH_ALGORITHM" != "rl-and-ga-compare" ]; then
  { [ "$S_STAGE2_COMPARE_REPEATS" = "false" ] && [ "$S_RL_COMPARE_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_GA_COMPARE_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_RL_COMPARE_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_GA_COMPARE_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_RL_COMPARE_FINAL_EVAL_CONFIG" = "false" ] && [ "$S_GA_COMPARE_FINAL_EVAL_CONFIG" = "false" ] && [ "$S_RL_COMPARE_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_GA_COMPARE_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_COMPARE_CONFIG_MODE" = "false" ] && [ "$S_COMPARE_PERSISTENT_ROOT" = "false" ] && [ "$S_RL_COMPARE_STAGE1_JSON" = "false" ] && [ "$S_RL_COMPARE_STAGE2_JSON" = "false" ] && [ "$S_GA_COMPARE_STAGE1_JSON" = "false" ] && [ "$S_GA_COMPARE_STAGE2_JSON" = "false" ] && [ "$S_RL_COMPARE_STAGE1_ACCURACY_TOLERANCE" = "false" ] && [ "$S_RL_COMPARE_STAGE2_LIMIT_TOLERANCE" = "false" ] && [ "$S_RL_COMPARE_STAGE2_STABILITY_TOLERANCE" = "false" ] && [ "$S_GA_COMPARE_STAGE1_ACCURACY_TOLERANCE" = "false" ] && [ "$S_GA_COMPARE_STAGE2_LIMIT_TOLERANCE" = "false" ] && [ "$S_GA_COMPARE_STAGE2_STABILITY_TOLERANCE" = "false" ]; } || err "当前模式不是 rl-and-ga-compare，请不要使用对比专用参数。"
fi

if [ "$SEARCH_ALGORITHM" = "general-rl" ] || [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  { [ "$S_STAGE2_FIXED_CONFIG_SOURCE" = "false" ] && [ "$S_STAGE2_FIXED_CONFIG" = "false" ] && [ -z "$STAGE2_MANUAL_GELU" ] && [ -z "$STAGE2_MANUAL_SOFTMAX" ]; } || err "当前模式不支持 --stage2-fixed-config-* 参数；该参数组仅普通 rl / ga 可用。"
fi

if [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
  { [ "$S_STAGE1_EPISODES" = "false" ] && [ "$S_STAGE2_EPISODES" = "false" ] && [ "$S_STAGE1_GENERATIONS" = "false" ] && [ "$S_STAGE2_GENERATIONS" = "false" ] && [ "$S_STAGE1_LR" = "false" ] && [ "$S_STAGE2_LR" = "false" ] && [ "$S_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_SKIP_FINAL_EVAL" = "false" ] && [ "$S_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_FINAL_EVAL_CONFIG" = "false" ] && [ -z "$MANUAL_STAGE1_GELU" ] && [ -z "$MANUAL_STAGE1_SOFTMAX" ] && [ -z "$MANUAL_STAGE2_NOISE" ] && [ "$S_STAGE2_FIXED_CONFIG_SOURCE" = "false" ] && [ "$S_STAGE2_FIXED_CONFIG" = "false" ] && [ -z "$STAGE2_MANUAL_GELU" ] && [ -z "$STAGE2_MANUAL_SOFTMAX" ]; } || err "general-rl 不能与普通 RL / GA 的阶段搜索或最终评估参数混用。"
  # ---- 准确度容忍参数校验 ----
  if [ -n "$GENERAL_ACCURACY_TOLERANCES" ]; then
    IFS=',' read -r -a __tol_items <<< "$GENERAL_ACCURACY_TOLERANCES"
    for __tol_val in "${__tol_items[@]}"; do
      __tol_val="$(printf '%s' "$__tol_val" | xargs)"
      [ -z "$__tol_val" ] && continue
      is_pos_num "$__tol_val" || err "--general-rl-accuracy-tolerances 中的值必须是正数：$__tol_val"
      awk -v x="$__tol_val" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "--general-rl-accuracy-tolerances 中的值必须 < 1（即百分比形式如 0.01 表示 1%），当前值：$__tol_val"
    done
  fi
  # ---- 连续准确度容忍范围校验 ----
  if [ -n "$GENERAL_ACCURACY_TOLERANCE_RANGE" ]; then
    IFS=',' read -r -a __range_items <<< "$GENERAL_ACCURACY_TOLERANCE_RANGE"
    [ "${#__range_items[@]}" -eq 2 ] || err "--general-rl-accuracy-tolerance-range 需要恰好两个值 (min,max)，如 '0.005,0.02'"
    __range_lo="$(printf '%s' "${__range_items[0]}" | xargs)"
    __range_hi="$(printf '%s' "${__range_items[1]}" | xargs)"
    is_pos_num "$__range_lo" || err "--general-rl-accuracy-tolerance-range 的下界必须是正数：$__range_lo"
    is_pos_num "$__range_hi" || err "--general-rl-accuracy-tolerance-range 的上界必须是正数：$__range_hi"
    awk -v x="$__range_lo" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "--general-rl-accuracy-tolerance-range 的下界必须 < 1，当前值：$__range_lo"
    awk -v x="$__range_hi" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "--general-rl-accuracy-tolerance-range 的上界必须 < 1，当前值：$__range_hi"
    awk -v lo="$__range_lo" -v hi="$__range_hi" 'BEGIN { if ((lo + 0) >= (hi + 0)) exit 1 }' || err "--general-rl-accuracy-tolerance-range 要求 min < max，当前 $__range_lo >= $__range_hi"
  fi

  # ---- 泛化模式推断 ----
  _HAS_MULTI_TASKS="false"
  if [ -n "$GENERAL_TASKS" ]; then
    IFS=',' read -r -a __task_arr <<< "$GENERAL_TASKS"
    [ "${#__task_arr[@]}" -gt 1 ] && _HAS_MULTI_TASKS="true"
  fi
  _HAS_MULTI_TOLS="false"
  if [ -n "$GENERAL_ACCURACY_TOLERANCE_RANGE" ]; then
    _HAS_MULTI_TOLS="true"
  elif [ -n "$GENERAL_ACCURACY_TOLERANCES" ]; then
    IFS=',' read -r -a __tol_arr <<< "$GENERAL_ACCURACY_TOLERANCES"
    __tol_cnt=0
    for __t in "${__tol_arr[@]}"; do
      __t="$(printf '%s' "$__t" | xargs)"
      [ -n "$__t" ] && __tol_cnt=$(( __tol_cnt + 1 ))
    done
    [ "$__tol_cnt" -gt 1 ] && _HAS_MULTI_TOLS="true"
  fi
  # 训练模式: 至少需要一种泛化维度（多任务或多容忍度或连续容忍范围）
  if [ "$GENERAL_MODE" = "train" ] && [ "$_HAS_MULTI_TASKS" = "false" ] && [ "$_HAS_MULTI_TOLS" = "false" ]; then
    echo "警告：general-rl train 模式下既未提供多个任务也未提供多个准确度容忍值/范围，策略可能无法学习到有效泛化。" >&2
  fi

  if [ "$GENERAL_MODE" = "train" ]; then
    is_pos_int "$GENERAL_ROUNDS" || err "--general-rl-rounds 必须是正整数"
    is_pos_int "$PPO_UPDATE_INTERVAL_VAL" || err "--ppo-update-interval 必须是正整数"
    is_pos_num "$GENERAL_LR" || err "--general-rl-lr 必须是正数"
    [ "$S_GENERAL_NUM_ROLLOUTS" = "false" ] && [ "$S_GENERAL_GREEDY" = "false" ] && [ "$S_GENERAL_STAGE1_POLICY" = "false" ] && [ "$S_GENERAL_STAGE2_POLICY" = "false" ] && [ "$S_GENERAL_POLICY_DIR" = "false" ] || err "general-rl train 模式不能使用 rollout / policy 参数。"
  else
    # search 模式：支持 --general-policy-dir 自动推导，或显式指定 policy 文件
    if [ -n "$GENERAL_POLICY_DIR" ]; then
      [ -d "$GENERAL_POLICY_DIR" ] || err "--general-policy-dir 指定的目录不存在：$GENERAL_POLICY_DIR"
      # 自动推导 Stage-1 策略
      if [ -z "$GENERAL_STAGE1_POLICY" ]; then
        _auto_s1="${GENERAL_POLICY_DIR}/general_stage1_policy.pt"
        [ -f "$_auto_s1" ] || err "在 --general-policy-dir 中未找到 Stage-1 策略文件：$_auto_s1"
        GENERAL_STAGE1_POLICY="$_auto_s1"
        echo "自动推导 Stage-1 策略：$GENERAL_STAGE1_POLICY"
      fi
      # 自动推导 Stage-2 策略（可选）
      if [ -z "$GENERAL_STAGE2_POLICY" ]; then
        _auto_s2="${GENERAL_POLICY_DIR}/general_stage2_noise_policy.pt"
        if [ -f "$_auto_s2" ]; then
          GENERAL_STAGE2_POLICY="$_auto_s2"
          echo "自动推导 Stage-2 策略：$GENERAL_STAGE2_POLICY"
        fi
      fi
    fi
    [ -n "$GENERAL_STAGE1_POLICY" ] || err "general-rl search 模式必须提供 --general-stage1-policy 或 --general-policy-dir。"
    [ -f "$GENERAL_STAGE1_POLICY" ] || err "--general-stage1-policy 指定的文件不存在：$GENERAL_STAGE1_POLICY"
    [ -z "$GENERAL_STAGE2_POLICY" ] || [ -f "$GENERAL_STAGE2_POLICY" ] || err "--general-stage2-policy 指定的文件不存在：$GENERAL_STAGE2_POLICY"
    is_pos_int "$GENERAL_NUM_ROLLOUTS" || err "--general-rl-num-rollouts 必须是正整数"
    { [ "$S_GENERAL_TASKS" = "false" ] && [ "$S_GENERAL_ROUNDS" = "false" ] && [ "$S_PPO_UPDATE_INTERVAL" = "false" ] && [ "$S_GENERAL_LR" = "false" ] && [ "$S_GENERAL_STAGE1_CONFIG_JSON" = "false" ] && [ "$S_RESUME_FROM" = "false" ]; } || err "general-rl search 模式不能使用训练专用参数。"
  fi
elif [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  { [ "$S_GENERAL_MODE" = "false" ] && [ "$S_GENERAL_TASKS" = "false" ] && [ "$S_GENERAL_ROUNDS" = "false" ] && [ "$S_PPO_UPDATE_INTERVAL" = "false" ] && [ "$S_GENERAL_LR" = "false" ] && [ "$S_GENERAL_NUM_ROLLOUTS" = "false" ] && [ "$S_GENERAL_GREEDY" = "false" ] && [ "$S_GENERAL_STAGE1_POLICY" = "false" ] && [ "$S_GENERAL_STAGE2_POLICY" = "false" ] && [ "$S_GENERAL_SKIP_STAGE2" = "false" ] && [ "$S_GENERAL_STAGE1_CONFIG_JSON" = "false" ] && [ "$S_GENERAL_ACCURACY_TOLERANCES" = "false" ]; } || err "rl-and-ga-compare 不能与 general-rl 参数混用。"
  if [ "$S_STAGE2_COMPARE_REPEATS" = "true" ] && [ "$S_FINAL_EVAL_REPEAT" = "true" ]; then
    err "rl-and-ga-compare 模式请只使用 --stage2-compare-repeats；不要再同时传入 --final-eval-repeat。"
  fi
  if [ -z "$STAGE2_COMPARE_REPEATS" ]; then
    STAGE2_COMPARE_REPEATS="1"
  fi
  is_pos_int "$STAGE2_COMPARE_REPEATS" || err "--stage2-compare-repeats 必须是正整数"
  [ "$SKIP_FINAL_EVAL" = "false" ] || err "rl-and-ga-compare 必须保留最终评估，不能使用 --skip-final-eval。"
  [ "$SKIP_STAGE1_SEARCH" = "false" ] || err "rl-and-ga-compare 不执行搜索流程，不能使用全局 --skip-stage1-search。"
  [ "$SKIP_NOISE_SEARCH" = "false" ] || err "rl-and-ga-compare 不执行搜索流程，不能使用全局 --skip-noise-search。"
  [ "$FINAL_EVAL_SOURCE" = "search" ] || err "rl-and-ga-compare 不使用全局 --final-eval-source。"
  [ -z "$MANUAL_STAGE1_GELU" ] && [ -z "$MANUAL_STAGE1_SOFTMAX" ] && [ -z "$MANUAL_STAGE2_NOISE" ] || err "rl-and-ga-compare 不支持 manual 配置输入。"
  [ "$S_FINAL_EVAL_CONFIG" = "false" ] || err "rl-and-ga-compare 不使用全局 --final-eval-config。"
  [ "$S_RESUME_FROM" = "false" ] || err "rl-and-ga-compare 不支持 --resume-from。"
  { [ "$S_STAGE1_EPISODES" = "false" ] && [ "$S_STAGE2_EPISODES" = "false" ] && [ "$S_STAGE1_GENERATIONS" = "false" ] && [ "$S_STAGE2_GENERATIONS" = "false" ] && [ "$S_STAGE1_LR" = "false" ] && [ "$S_STAGE2_LR" = "false" ] && [ "$S_RL_COMPARE_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_GA_COMPARE_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_RL_COMPARE_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_GA_COMPARE_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_RL_COMPARE_FINAL_EVAL_CONFIG" = "false" ] && [ "$S_GA_COMPARE_FINAL_EVAL_CONFIG" = "false" ] && [ "$S_RL_COMPARE_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_GA_COMPARE_SKIP_NOISE_SEARCH" = "false" ]; } || err "rl-and-ga-compare 已改为 JSON/持久化目录对比模式，不再支持旧的搜索/skip/source 参数。"
  case "$COMPARE_CONFIG_MODE" in
    direct|persistent) ;;
    *) err "--compare-config-mode 只支持 direct 或 persistent。" ;;
  esac

  if [ "$COMPARE_CONFIG_MODE" = "direct" ]; then
    [ "$S_COMPARE_PERSISTENT_ROOT" = "false" ] || err "direct 模式不使用 --compare-persistent-root。"
    { [ "$S_STAGE1_ACCURACY_TOLERANCE" = "false" ] && [ "$S_STAGE2_LIMIT_TOLERANCE" = "false" ] && [ "$S_STAGE2_STABILITY_TOLERANCE" = "false" ] && [ "$S_RL_COMPARE_STAGE1_ACCURACY_TOLERANCE" = "false" ] && [ "$S_RL_COMPARE_STAGE2_LIMIT_TOLERANCE" = "false" ] && [ "$S_RL_COMPARE_STAGE2_STABILITY_TOLERANCE" = "false" ] && [ "$S_GA_COMPARE_STAGE1_ACCURACY_TOLERANCE" = "false" ] && [ "$S_GA_COMPARE_STAGE2_LIMIT_TOLERANCE" = "false" ] && [ "$S_GA_COMPARE_STAGE2_STABILITY_TOLERANCE" = "false" ]; } || err "direct 模式不接受准确度约束参数，请直接指定四个 JSON 文件。"
    [ -n "$RL_COMPARE_STAGE1_JSON" ] || err "direct 模式必须提供 --rl-compare-stage1-json。"
    [ -n "$RL_COMPARE_STAGE2_JSON" ] || err "direct 模式必须提供 --rl-compare-stage2-json。"
    [ -n "$GA_COMPARE_STAGE1_JSON" ] || err "direct 模式必须提供 --ga-compare-stage1-json。"
    [ -n "$GA_COMPARE_STAGE2_JSON" ] || err "direct 模式必须提供 --ga-compare-stage2-json。"
    [ -f "$RL_COMPARE_STAGE1_JSON" ] || err "RL Stage-1 JSON 文件不存在：$RL_COMPARE_STAGE1_JSON"
    [ -f "$RL_COMPARE_STAGE2_JSON" ] || err "RL Stage-2 JSON 文件不存在：$RL_COMPARE_STAGE2_JSON"
    [ -f "$GA_COMPARE_STAGE1_JSON" ] || err "GA Stage-1 JSON 文件不存在：$GA_COMPARE_STAGE1_JSON"
    [ -f "$GA_COMPARE_STAGE2_JSON" ] || err "GA Stage-2 JSON 文件不存在：$GA_COMPARE_STAGE2_JSON"
    FAM="$(infer_family "$RL_COMPARE_STAGE1_JSON")"; [ "$FAM" != "ga" ] || err "RL Stage-1 JSON 看起来属于 GA 家族：$RL_COMPARE_STAGE1_JSON"
    FAM="$(infer_family "$RL_COMPARE_STAGE2_JSON")"; [ "$FAM" != "ga" ] || err "RL Stage-2 JSON 看起来属于 GA 家族：$RL_COMPARE_STAGE2_JSON"
    FAM="$(infer_family "$GA_COMPARE_STAGE1_JSON")"; [ "$FAM" != "rl" ] || err "GA Stage-1 JSON 看起来属于 RL/PPO 家族：$GA_COMPARE_STAGE1_JSON"
    FAM="$(infer_family "$GA_COMPARE_STAGE2_JSON")"; [ "$FAM" != "rl" ] || err "GA Stage-2 JSON 看起来属于 RL/PPO 家族：$GA_COMPARE_STAGE2_JSON"
  else
    { [ "$S_RL_COMPARE_STAGE1_JSON" = "false" ] && [ "$S_RL_COMPARE_STAGE2_JSON" = "false" ] && [ "$S_GA_COMPARE_STAGE1_JSON" = "false" ] && [ "$S_GA_COMPARE_STAGE2_JSON" = "false" ]; } || err "persistent 模式不接受直接 JSON 路径参数。"
    [ -d "$COMPARE_PERSISTENT_ROOT" ] || err "--compare-persistent-root 指定的目录不存在：$COMPARE_PERSISTENT_ROOT"
    [ -n "$RL_COMPARE_STAGE1_ACCURACY_TOLERANCE" ] || RL_COMPARE_STAGE1_ACCURACY_TOLERANCE="$STAGE1_ACCURACY_TOLERANCE"
    [ -n "$RL_COMPARE_STAGE2_LIMIT_TOLERANCE" ] || RL_COMPARE_STAGE2_LIMIT_TOLERANCE="$STAGE2_LIMIT_TOLERANCE"
    [ -n "$RL_COMPARE_STAGE2_STABILITY_TOLERANCE" ] || RL_COMPARE_STAGE2_STABILITY_TOLERANCE="$STAGE2_STABILITY_TOLERANCE"
    [ -n "$GA_COMPARE_STAGE1_ACCURACY_TOLERANCE" ] || GA_COMPARE_STAGE1_ACCURACY_TOLERANCE="$STAGE1_ACCURACY_TOLERANCE"
    [ -n "$GA_COMPARE_STAGE2_LIMIT_TOLERANCE" ] || GA_COMPARE_STAGE2_LIMIT_TOLERANCE="$STAGE2_LIMIT_TOLERANCE"
    [ -n "$GA_COMPARE_STAGE2_STABILITY_TOLERANCE" ] || GA_COMPARE_STAGE2_STABILITY_TOLERANCE="$STAGE2_STABILITY_TOLERANCE"
    for _cmp_val in \
      "$RL_COMPARE_STAGE1_ACCURACY_TOLERANCE" "$RL_COMPARE_STAGE2_LIMIT_TOLERANCE" "$RL_COMPARE_STAGE2_STABILITY_TOLERANCE" \
      "$GA_COMPARE_STAGE1_ACCURACY_TOLERANCE" "$GA_COMPARE_STAGE2_LIMIT_TOLERANCE" "$GA_COMPARE_STAGE2_STABILITY_TOLERANCE"; do
      is_pos_num "$_cmp_val" || err "持久化目录对比的约束参数必须是正数，当前值：$_cmp_val"
      awk -v x="$_cmp_val" 'BEGIN { if ((x + 0) >= 1) exit 1 }' || err "持久化目录对比的约束参数必须 < 1，当前值：$_cmp_val"
    done
    RL_COMPARE_CONSTRAINT_SLUG="s1t${RL_COMPARE_STAGE1_ACCURACY_TOLERANCE}_s2t${RL_COMPARE_STAGE2_LIMIT_TOLERANCE}_s2st${RL_COMPARE_STAGE2_STABILITY_TOLERANCE}"
    GA_COMPARE_CONSTRAINT_SLUG="s1t${GA_COMPARE_STAGE1_ACCURACY_TOLERANCE}_s2t${GA_COMPARE_STAGE2_LIMIT_TOLERANCE}_s2st${GA_COMPARE_STAGE2_STABILITY_TOLERANCE}"
    RL_COMPARE_PERSISTENT_DIR="${COMPARE_PERSISTENT_ROOT}/rl/${MODEL_TYPE}/${DATASET}/${RL_COMPARE_CONSTRAINT_SLUG}"
    GA_COMPARE_PERSISTENT_DIR="${COMPARE_PERSISTENT_ROOT}/ga/${MODEL_TYPE}/${DATASET}/${GA_COMPARE_CONSTRAINT_SLUG}"
    [ -d "$RL_COMPARE_PERSISTENT_DIR" ] || err "persistent 模式未找到 RL 持久化目录：$RL_COMPARE_PERSISTENT_DIR。请先运行对应 RL 实验，或检查数据集 / 模型 / 约束参数是否一致。"
    [ -f "${RL_COMPARE_PERSISTENT_DIR}/metadata.json" ] || err "persistent 模式找到 RL 目录但缺少 metadata.json：${RL_COMPARE_PERSISTENT_DIR}/metadata.json"
    [ -d "$GA_COMPARE_PERSISTENT_DIR" ] || err "persistent 模式未找到 GA 持久化目录：$GA_COMPARE_PERSISTENT_DIR。请先运行对应 GA 实验，或检查数据集 / 模型 / 约束参数是否一致。"
    [ -f "${GA_COMPARE_PERSISTENT_DIR}/metadata.json" ] || err "persistent 模式找到 GA 目录但缺少 metadata.json：${GA_COMPARE_PERSISTENT_DIR}/metadata.json"
  fi
else
  { [ "$S_GENERAL_MODE" = "false" ] && [ "$S_GENERAL_TASKS" = "false" ] && [ "$S_GENERAL_ROUNDS" = "false" ] && [ "$S_GENERAL_LR" = "false" ] && [ "$S_GENERAL_NUM_ROLLOUTS" = "false" ] && [ "$S_GENERAL_GREEDY" = "false" ] && [ "$S_GENERAL_STAGE1_POLICY" = "false" ] && [ "$S_GENERAL_STAGE2_POLICY" = "false" ] && [ "$S_GENERAL_SKIP_STAGE2" = "false" ] && [ "$S_GENERAL_STAGE1_CONFIG_JSON" = "false" ] && [ "$S_GENERAL_ACCURACY_TOLERANCES" = "false" ]; } || err "当前搜索算法不是 general-rl，请不要使用 --general-rl-* 参数。"
  # rl/ga 模式下 --resume-from 已废弃，改用持久化目录自动续训练
  [ "$S_RESUME_FROM" = "false" ] || err "rl / ga 模式已改用持久化目录自动续训练，不再支持手动 --resume-from。续训练时直接运行相同参数即可；首次运行请加 --fresh-start。"
  _EARLY_CONSTRAINT_SLUG="s1t${STAGE1_ACCURACY_TOLERANCE}_s2t${STAGE2_LIMIT_TOLERANCE}_s2st${STAGE2_STABILITY_TOLERANCE}"
  _EARLY_PERSISTENT_DIR="rl_results/persistent/${SEARCH_ALGORITHM}/${MODEL_TYPE}/${DATASET}/${_EARLY_CONSTRAINT_SLUG}"
  if [ "$SEARCH_ALGORITHM" = "rl" ]; then
    [ "$S_STAGE1_GENERATIONS" = "false" ] && [ "$S_STAGE2_GENERATIONS" = "false" ] || err "rl 模式不使用 GA 代数参数，请移除 --stage1-search-generations / --stage2-search-generations。"
    is_pos_int "$STAGE1_EPISODES" || err "--stage1-search-episodes 必须是正整数"
    is_pos_int "$STAGE2_EPISODES" || err "--stage2-search-episodes 必须是正整数"
    is_pos_num "$STAGE1_LR" || err "--stage1-search-lr 必须是正数"
    is_pos_num "$STAGE2_LR" || err "--stage2-search-lr 必须是正数"
    [ "$SKIP_STAGE1_SEARCH" = "true" ] || [ "$STAGE1_EPISODES" -ge 170 ] || err "rl 的 Stage-1 回合数至少需要 170。"
    [ "$SKIP_NOISE_SEARCH" = "true" ] || [ "$STAGE2_EPISODES" -ge 170 ] || err "rl 的 Stage-2 回合数至少需要 170。"
  else
    [ "$S_STAGE1_EPISODES" = "false" ] && [ "$S_STAGE2_EPISODES" = "false" ] || err "ga 模式不再使用 episode 作为搜索预算，请改用 --stage1-search-generations / --stage2-search-generations。"
    is_pos_int "$STAGE1_GENERATIONS" || err "--stage1-search-generations 必须是正整数"
    is_pos_int "$STAGE2_GENERATIONS" || err "--stage2-search-generations 必须是正整数"
    [ "$SKIP_STAGE1_SEARCH" = "false" ] || [ "$S_STAGE1_GENERATIONS" = "false" ] || err "已指定 --skip-stage1-search 时，不能再显式提供 --stage1-search-generations。"
    [ "$SKIP_NOISE_SEARCH" = "false" ] || [ "$S_STAGE2_GENERATIONS" = "false" ] || err "已指定 --skip-noise-search 时，不能再显式提供 --stage2-search-generations。"
    [ "$S_STAGE1_LR" = "false" ] && [ "$S_STAGE2_LR" = "false" ] || err "GA 不使用 PPO 学习率参数，请移除 --stage1-search-lr / --stage2-search-lr。"
  fi
  if [ "$FINAL_EVAL_SOURCE" = "manual" ]; then
    [ -n "$MANUAL_STAGE1_GELU" ] && [ -n "$MANUAL_STAGE1_SOFTMAX" ] || err "manual 最终评估配置必须同时提供 --manual-stage1-gelu 和 --manual-stage1-softmax。"
    [ -n "$MANUAL_STAGE2_NOISE" ] || err "manual 最终评估配置必须提供 --manual-stage2-noise。"
  else
    [ -z "$MANUAL_STAGE1_GELU" ] && [ -z "$MANUAL_STAGE1_SOFTMAX" ] && [ -z "$MANUAL_STAGE2_NOISE" ] || err "只有 --final-eval-source=manual 时才能提供 --manual-stage1-gelu / --manual-stage1-softmax / --manual-stage2-noise。"
  fi
  _NEEDS_STAGE2_FIXED_CONFIG="false"
  if [ "$SKIP_NOISE_SEARCH" = "false" ]; then
    _NEEDS_STAGE2_FIXED_CONFIG="true"
  fi
  if [ "$_NEEDS_STAGE2_FIXED_CONFIG" = "true" ]; then
    if [ "$STAGE2_FIXED_CONFIG_SOURCE" = "manual" ]; then
      [ -n "$STAGE2_MANUAL_GELU" ] && [ -n "$STAGE2_MANUAL_SOFTMAX" ] || err "stage2-fixed-config-source=manual 时必须同时提供 --stage2-manual-gelu 和 --stage2-manual-softmax。"
    else
      [ -z "$STAGE2_MANUAL_GELU" ] && [ -z "$STAGE2_MANUAL_SOFTMAX" ] || err "只有 --stage2-fixed-config-source=manual 时才能提供 --stage2-manual-gelu / --stage2-manual-softmax。"
    fi
    if [ "$SKIP_STAGE1_SEARCH" = "true" ] && [ "$STAGE2_FIXED_CONFIG_SOURCE" = "stage1_result" ]; then
      _HAS_S1_CKPT="false"
      [ -f "${_EARLY_PERSISTENT_DIR}/stage1/stage1_rl_checkpoint.pt" ] && _HAS_S1_CKPT="true"
      [ "$_HAS_S1_CKPT" = "true" ] || err "跳过 Stage-1 搜索且无历史 Stage-1 结果，Stage-2 固定 GELU/Softmax 不能使用 --stage2-fixed-config-source=stage1_result。请改用 json 或 manual，或先运行 Stage-1。"
    fi
    if [ "$STAGE2_FIXED_CONFIG_SOURCE" = "json" ]; then
      [ -f "$STAGE2_FIXED_CONFIG" ] || err "Stage-2 固定 GELU/Softmax 的 JSON 配置文件不存在：$STAGE2_FIXED_CONFIG"
      FAM="$(infer_family "$STAGE2_FIXED_CONFIG")"
      [ "$SEARCH_ALGORITHM" != "ga" ] || [ "$FAM" != "rl" ] || err "已选择 ga，但 Stage-2 固定 GELU/Softmax 的 JSON 配置看起来属于 RL/PPO 家族：$STAGE2_FIXED_CONFIG"
      [ "$SEARCH_ALGORITHM" != "rl" ] || [ "$FAM" != "ga" ] || err "已选择 rl，但 Stage-2 固定 GELU/Softmax 的 JSON 配置看起来属于 GA 家族：$STAGE2_FIXED_CONFIG"
    fi
  fi
  if [ "$FINAL_EVAL_SOURCE" = "search" ]; then
    if [ "$SKIP_STAGE1_SEARCH" = "true" ] && { [ "$SKIP_NOISE_SEARCH" = "false" ] || [ "$SKIP_FINAL_EVAL" = "false" ]; }; then
      [ -f "$FINAL_EVAL_CONFIG" ] || err "--final-eval-source=search 且跳过 Stage-1 时，需要 --final-eval-config 提供 Stage-1 回退配置。未找到文件：$FINAL_EVAL_CONFIG"
    fi
    if [ "$SKIP_NOISE_SEARCH" = "true" ] && [ "$SKIP_FINAL_EVAL" = "false" ]; then
      [ -f "$FINAL_EVAL_CONFIG" ] || err "--final-eval-source=search 且跳过 Stage-2 并保留最终评估时，需要 --final-eval-config 提供 Stage-2 回退配置。未找到文件：$FINAL_EVAL_CONFIG"
    fi
    [ "$SKIP_STAGE1_SEARCH" = "false" ] || [ "$SKIP_NOISE_SEARCH" = "false" ] || [ "$SKIP_FINAL_EVAL" = "true" ] || err "当 Stage-1 与 Stage-2 搜索都跳过且保留最终评估时，不能使用 --final-eval-source=search。"
  fi
  if [ "$S_FINAL_EVAL_SOURCE" = "true" ]; then
    [ "$SKIP_STAGE1_SEARCH" = "true" ] || [ "$SKIP_NOISE_SEARCH" = "true" ] || [ "$FINAL_EVAL_SOURCE" = "search" ] || err "执行 Stage-1 / Stage-2 搜索时，--final-eval-source 只能是 search。"
  else
    if [ "$SKIP_STAGE1_SEARCH" = "false" ] && [ "$SKIP_NOISE_SEARCH" = "false" ] && [ "$FINAL_EVAL_SOURCE" != "search" ]; then
      echo "[自动适配] 启用 Stage-1 + Stage-2 搜索，自动将 --final-eval-source 设为 search。"
      FINAL_EVAL_SOURCE="search"
    fi
  fi
  if [ "$FINAL_EVAL_SOURCE" = "json" ] && [ "$SKIP_FINAL_EVAL" = "false" ]; then
    [ -f "$FINAL_EVAL_CONFIG" ] || err "最终评估 JSON 配置文件不存在：$FINAL_EVAL_CONFIG"
    FAM="$(infer_family "$FINAL_EVAL_CONFIG")"
    [ "$SEARCH_ALGORITHM" != "ga" ] || [ "$FAM" != "rl" ] || err "已选择 ga，但最终评估 JSON 配置看起来属于 RL/PPO 家族：$FINAL_EVAL_CONFIG"
    [ "$SEARCH_ALGORITHM" != "rl" ] || [ "$FAM" != "ga" ] || err "已选择 rl，但最终评估 JSON 配置看起来属于 GA 家族：$FINAL_EVAL_CONFIG"
  fi
fi

if [ "$MODEL_TYPE" = "bert-base" ]; then
  case "$DATASET" in
    wnli) BASE_MODEL="textattack/bert-base-uncased-WNLI" ;;
    rte) BASE_MODEL="textattack/bert-base-uncased-RTE" ;;
    cola) BASE_MODEL="textattack/bert-base-uncased-CoLA" ;;
    qnli) BASE_MODEL="textattack/bert-base-uncased-QNLI" ;;
    mrpc) BASE_MODEL="textattack/bert-base-uncased-MRPC" ;;
    sst2) BASE_MODEL="textattack/bert-base-uncased-SST-2" ;;
    stsb) BASE_MODEL="textattack/bert-base-uncased-STS-B" ;;
  esac
elif [ "$MODEL_TYPE" = "bert-large" ]; then
  case "$DATASET" in
    mrpc) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-mrpc" ;;
    cola) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-cola" ;;
    stsb) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-stsb" ;;
    rte) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-rte" ;;
    sst2) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-sst2" ;;
    qnli) BASE_MODEL="yoshitomo-matsubara/bert-large-uncased-qnli" ;;
    *) err "bert-large 当前仅支持 mrpc, cola, stsb, rte, sst2, qnli" ;;
  esac
else
  case "$DATASET" in
    cola) BASE_MODEL="PavanNeerudu/gpt2-finetuned-cola" ;;
    sst2) BASE_MODEL="PavanNeerudu/gpt2-finetuned-sst2" ;;
    mrpc) BASE_MODEL="PavanNeerudu/gpt2-finetuned-mrpc" ;;
    stsb) BASE_MODEL="PavanNeerudu/gpt2-finetuned-stsb" ;;
    qnli) BASE_MODEL="PavanNeerudu/gpt2-finetuned-qnli" ;;
    rte) BASE_MODEL="PavanNeerudu/gpt2-finetuned-rte" ;;
    wnli) BASE_MODEL="PavanNeerudu/gpt2-finetuned-wnli" ;;
    *) err "gpt-2 当前不支持数据集：$DATASET" ;;
  esac
fi

export NCCL_DEBUG=INFO
LOGFILE_BASENAME="$(basename "$LOGFILE")"
[ -n "$LOGFILE_BASENAME" ] || LOGFILE_BASENAME="output.log"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_ID="${RUN_TIMESTAMP}_pid$$"
GENERAL_TASKSET_ID=""
# 持久化约束配置标识符（用于构建确定性目录）
CONSTRAINT_SLUG="s1t${STAGE1_ACCURACY_TOLERANCE}_s2t${STAGE2_LIMIT_TOLERANCE}_s2st${STAGE2_STABILITY_TOLERANCE}"
USE_PERSISTENT="false"
PERSISTENT_DIR=""

if [ "$SEARCH_ALGORITHM" = "rl" ] || [ "$SEARCH_ALGORITHM" = "ga" ]; then
  # RL / GA 使用持久化目录：基于(算法、模型、数据集、约束参数)确定性生成
  USE_PERSISTENT="true"
  PERSISTENT_DIR="rl_results/persistent/${SEARCH_ALGORITHM}/${MODEL_TYPE}/${DATASET}/${CONSTRAINT_SLUG}"

  # 判断是否跳过了所有搜索阶段（eval-only 模式）
  _ALL_SEARCH_SKIPPED="false"
  if [ "$SKIP_STAGE1_SEARCH" = "true" ] && [ "$SKIP_NOISE_SEARCH" = "true" ]; then
    _ALL_SEARCH_SKIPPED="true"
  fi

  if [ "$FRESH_START" = "true" ]; then
    # 显式从头训练：清除旧数据（如果存在）
    if [ -d "$PERSISTENT_DIR" ]; then
      # 安全检查：如果跳过了某个阶段的搜索，但持久化目录中有对应的 checkpoint，
      # 说明 --fresh-start 会销毁用户未打算重做的阶段的搜索结果。
      _HAS_STAGE1_DATA="false"
      _HAS_STAGE2_DATA="false"
      [ -d "${PERSISTENT_DIR}/stage1" ] && _HAS_STAGE1_DATA="true"
      [ -d "${PERSISTENT_DIR}/stage2_noise" ] && _HAS_STAGE2_DATA="true"
      if [ "$SKIP_STAGE1_SEARCH" = "true" ] && [ "$_HAS_STAGE1_DATA" = "true" ]; then
        echo "⚠ 警告：--fresh-start 将清除已有的 Stage-1 搜索结果，但当前 --skip-stage1-search 跳过了 Stage-1。"
        echo "  如果您只想重做 Stage-2 搜索，请不要使用 --fresh-start（直接运行即可自动续训练）。"
        echo "  5 秒后继续执行（Ctrl+C 可取消）..."
        sleep 5
      fi
      if [ "$SKIP_NOISE_SEARCH" = "true" ] && [ "$_HAS_STAGE2_DATA" = "true" ]; then
        echo "⚠ 警告：--fresh-start 将清除已有的 Stage-2 搜索结果，但当前 --skip-noise-search 跳过了 Stage-2。"
        echo "  如果您只想重做 Stage-1 搜索，请不要使用 --fresh-start（直接运行即可自动续训练）。"
        echo "  5 秒后继续执行（Ctrl+C 可取消）..."
        sleep 5
      fi
      echo "警告：--fresh-start 指定，正在清除已有持久化目录：$PERSISTENT_DIR"
      rm -rf "$PERSISTENT_DIR"
    fi
    mkdir -p "${PERSISTENT_DIR}/logs"
  elif [ -d "$PERSISTENT_DIR" ] && [ -f "${PERSISTENT_DIR}/metadata.json" ]; then
    # 目录存在且有 metadata → 自动续训练
    echo "检测到已有持久化目录：$PERSISTENT_DIR"
    echo "自动进入续训练模式（如需从头训练请加 --fresh-start）。"
    RESUME_FROM="$PERSISTENT_DIR"
    S_RESUME_FROM="true"
    # ---- 单阶段重置（--fresh-stage1 / --fresh-stage2） ----
    if [ "$FRESH_STAGE1" = "true" ] && [ -d "${PERSISTENT_DIR}/stage1" ]; then
      echo "[单阶段重置] --fresh-stage1 指定，正在清除 Stage-1 数据：${PERSISTENT_DIR}/stage1"
      rm -rf "${PERSISTENT_DIR}/stage1"
      rm -rf "${PERSISTENT_DIR}/stage1_final_eval"
      # 更新 metadata 中的阶段状态
      if command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
p = '${PERSISTENT_DIR}/metadata.json'
with open(p, 'r') as f: m = json.load(f)
ss = m.setdefault('stage_status', {})
ss['stage1_search'] = 'not_started'
ss['stage1_final_eval'] = 'not_started'
with open(p, 'w') as f: json.dump(m, f, indent=2)
" 2>/dev/null || true
      fi
    fi
    if [ "$FRESH_STAGE2" = "true" ] && [ -d "${PERSISTENT_DIR}/stage2_noise" ]; then
      echo "[单阶段重置] --fresh-stage2 指定，正在清除 Stage-2 数据：${PERSISTENT_DIR}/stage2_noise"
      rm -rf "${PERSISTENT_DIR}/stage2_noise"
      rm -rf "${PERSISTENT_DIR}/stage2_noise_final_eval"
      if command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
p = '${PERSISTENT_DIR}/metadata.json'
with open(p, 'r') as f: m = json.load(f)
ss = m.setdefault('stage_status', {})
ss['stage2_search'] = 'not_started'
ss['stage2_final_eval'] = 'not_started'
with open(p, 'w') as f: json.dump(m, f, indent=2)
" 2>/dev/null || true
      fi
    fi
  elif [ "$_ALL_SEARCH_SKIPPED" = "true" ]; then
    # 所有搜索都跳过（eval-only）：无需 --fresh-start，自动创建目录
    echo "所有搜索阶段已跳过（eval-only 模式），自动创建持久化目录：$PERSISTENT_DIR"
    mkdir -p "${PERSISTENT_DIR}/logs"
  else
    # 第一次运行且未指定 --fresh-start
    err "首次运行该参数组合（${CONSTRAINT_SLUG}），请显式指定 --fresh-start 以确认从头训练。如果是续训练但目录被删除，请重新用 --fresh-start 开始。"
  fi
  RUN_ROOT="$PERSISTENT_DIR"
  RUN_GROUP_DIR="$(dirname "$PERSISTENT_DIR")"
  LATEST_BASE_DIR="$RUN_GROUP_DIR"
elif [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
  if [ "$GENERAL_MODE" = "train" ]; then
    GENERAL_TASKSET_ID="$(normalize_taskset_id "${GENERAL_TASKS:-$DATASET}")"
    # 泛化模式决定子目录（仅用于显示）
    if [ "$_HAS_MULTI_TASKS" = "true" ] && [ "$_HAS_MULTI_TOLS" = "true" ]; then
      _GEN_SUBDIR="combined_gen"
    elif [ "$_HAS_MULTI_TOLS" = "true" ]; then
      _GEN_SUBDIR="accuracy_gen"
    elif [ "$_HAS_MULTI_TASKS" = "true" ]; then
      _GEN_SUBDIR="dataset_gen"
    else
      _GEN_SUBDIR="single"
    fi
    # ---- 构建准确度标识符（accuracy_slug）----
    if [ -n "$GENERAL_ACCURACY_TOLERANCE_RANGE" ]; then
      # 连续范围: range_0.50pct_2.00pct
      IFS=',' read -r -a __range_parts <<< "$GENERAL_ACCURACY_TOLERANCE_RANGE"
      _lo_pct="$(awk -v x="$(printf '%s' "${__range_parts[0]}" | xargs)" 'BEGIN{printf "%.2f", x*100}')"
      _hi_pct="$(awk -v x="$(printf '%s' "${__range_parts[1]}" | xargs)" 'BEGIN{printf "%.2f", x*100}')"
      GENERAL_ACCURACY_SLUG="range_${_lo_pct}pct_${_hi_pct}pct"
    elif [ -n "$GENERAL_ACCURACY_TOLERANCES" ]; then
      # 离散列表: discrete_0.50pct_1.00pct_2.00pct
      _slug="discrete"
      IFS=',' read -r -a __tol_parts <<< "$GENERAL_ACCURACY_TOLERANCES"
      for __tv in "${__tol_parts[@]}"; do
        __tv="$(printf '%s' "$__tv" | xargs)"
        [ -z "$__tv" ] && continue
        _tv_pct="$(awk -v x="$__tv" 'BEGIN{printf "%.2f", x*100}')"
        _slug="${_slug}_${_tv_pct}pct"
      done
      GENERAL_ACCURACY_SLUG="$_slug"
    else
      GENERAL_ACCURACY_SLUG="default"
    fi
    # ---- 持久化目录 ----
    USE_PERSISTENT="true"
    PERSISTENT_DIR="rl_results/persistent/general-rl/${MODEL_TYPE}/${GENERAL_TASKSET_ID}/${GENERAL_ACCURACY_SLUG}"
    if [ "$FRESH_START" = "true" ]; then
      if [ -d "$PERSISTENT_DIR" ]; then
        echo "警告：--fresh-start 指定，正在清除已有持久化目录：$PERSISTENT_DIR"
        rm -rf "$PERSISTENT_DIR"
      fi
      mkdir -p "${PERSISTENT_DIR}/logs"
    elif [ -d "$PERSISTENT_DIR" ] && [ -f "${PERSISTENT_DIR}/metadata.json" ]; then
      echo "检测到已有持久化目录：$PERSISTENT_DIR"
      echo "自动进入续训练模式（如需从头训练请加 --fresh-start）。"
      RESUME_FROM="$PERSISTENT_DIR"
      S_RESUME_FROM="true"
    else
      err "首次运行该参数组合（${GENERAL_TASKSET_ID}/${GENERAL_ACCURACY_SLUG}），请显式指定 --fresh-start 以确认从头训练。如果是续训练但目录被删除，请重新用 --fresh-start 开始。"
    fi
    RUN_ROOT="$PERSISTENT_DIR"
    RUN_GROUP_DIR="$(dirname "$PERSISTENT_DIR")"
    LATEST_BASE_DIR="$RUN_GROUP_DIR"
  else
    # search 模式：使用时间戳目录存放搜索结果
    RUN_GROUP_DIR="rl_results/runs/general_rl/search/${DATASET}"
    RUN_ROOT="${RUN_GROUP_DIR}/${RUN_ID}"
    LATEST_BASE_DIR="$RUN_GROUP_DIR"
  fi
else
  # rl-and-ga-compare：对比实验使用时间戳目录
  RUN_GROUP_DIR="rl_results/runs/compare/rl_vs_ga/${DATASET}"
  RUN_ROOT="${RUN_GROUP_DIR}/${RUN_ID}"
  LATEST_BASE_DIR="$RUN_GROUP_DIR"
fi
LOGFILE_PATH="${RUN_ROOT}/logs/${LOGFILE_BASENAME}"
ERROR_SUMMARY_PATH="${RUN_ROOT}/logs/error_summary.txt"
mkdir -p "${RUN_ROOT}/logs"

# 对于持久化目录，写入/更新 metadata.json
if [ "$USE_PERSISTENT" = "true" ]; then
  _META_FILE="${PERSISTENT_DIR}/metadata.json"
  if [ ! -f "$_META_FILE" ]; then
    if [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
      # general-rl 持久化 metadata
      _META_TASKS="${GENERAL_TASKS:-$DATASET}"
      _META_TOL_RANGE="${GENERAL_ACCURACY_TOLERANCE_RANGE:-null}"
      _META_TOL_LIST="${GENERAL_ACCURACY_TOLERANCES:-null}"
      [ "$_META_TOL_RANGE" != "null" ] && _META_TOL_RANGE="\"$_META_TOL_RANGE\""
      [ "$_META_TOL_LIST" != "null" ] && _META_TOL_LIST="\"$_META_TOL_LIST\""
      cat > "$_META_FILE" <<METAEOF
{
  "algorithm": "general-rl",
  "model_type": "$MODEL_TYPE",
  "taskset": "$GENERAL_TASKSET_ID",
  "tasks": "$_META_TASKS",
  "accuracy_slug": "$GENERAL_ACCURACY_SLUG",
  "accuracy_tolerance_range": $_META_TOL_RANGE,
  "accuracy_tolerances": $_META_TOL_LIST,
  "created_at": "$(date -Iseconds)",
  "last_updated_at": "$(date -Iseconds)",
  "run_count": 1
}
METAEOF
    else
      cat > "$_META_FILE" <<METAEOF
{
  "algorithm": "$SEARCH_ALGORITHM",
  "model_type": "$MODEL_TYPE",
  "dataset": "$DATASET",
  "stage1_accuracy_tolerance": $STAGE1_ACCURACY_TOLERANCE,
  "stage2_limit_tolerance": $STAGE2_LIMIT_TOLERANCE,
  "stage2_stability_tolerance": $STAGE2_STABILITY_TOLERANCE,
  "stage2_k_trials": $STAGE2_K_TRIALS,
  "stage2_probe_size": $STAGE2_PROBE_SIZE,
  "created_at": "$(date -Iseconds)",
  "last_updated_at": "$(date -Iseconds)",
  "run_count": 1,
  "stage_status": {
    "stage1_search": "not_started",
    "stage1_final_eval": "not_started",
    "stage2_search": "not_started",
    "stage2_final_eval": "not_started"
  }
}
METAEOF
    fi
  else
    # 更新已有 metadata 的时间戳和计数
    if command -v python3 &>/dev/null; then
      python3 -c "
import json, sys, datetime
with open('$_META_FILE', 'r') as f: m = json.load(f)
m['last_updated_at'] = datetime.datetime.now().isoformat()
m['run_count'] = m.get('run_count', 0) + 1
with open('$_META_FILE', 'w') as f: json.dump(m, f, indent=2)
" 2>/dev/null || true
    fi
  fi
fi

if [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
  GENERAL_DATA_PATH="$DATA_PATH"
  [ "$GENERAL_MODE" = "train" ] && [ -n "$GENERAL_TASKS" ] && GENERAL_DATA_PATH="$GENERAL_TASKS"
  # search 模式在 Python 端仍映射为 infer（兼容）
  _PY_MODE="$GENERAL_MODE"
  [ "$_PY_MODE" = "search" ] && _PY_MODE="search"
  CMD=(python rl_tune_general.py "$_PY_MODE" --model_type "$MODEL_TYPE" --data_path "$GENERAL_DATA_PATH" --output_dir "$RUN_ROOT" --batch_size "$BATCH_SIZE" --device cuda)
  if [ "$GENERAL_MODE" = "train" ]; then
    CMD+=(--total_rounds "$GENERAL_ROUNDS" --ppo_update_interval "$PPO_UPDATE_INTERVAL_VAL" --general_lr "$GENERAL_LR" --skip_stage2 "$GENERAL_SKIP_STAGE2")
    [ -n "$GENERAL_STAGE1_CONFIG_JSON" ] && CMD+=(--stage1_config_json "$GENERAL_STAGE1_CONFIG_JSON")
    [ -n "$GENERAL_ACCURACY_TOLERANCES" ] && CMD+=(--accuracy_tolerances "$GENERAL_ACCURACY_TOLERANCES")
    [ -n "$GENERAL_ACCURACY_TOLERANCE_RANGE" ] && CMD+=(--accuracy_tolerance_range "$GENERAL_ACCURACY_TOLERANCE_RANGE")
    [ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")
  else
    CMD+=(--general_stage1_policy "$GENERAL_STAGE1_POLICY" --num_rollouts "$GENERAL_NUM_ROLLOUTS" --greedy "$GENERAL_GREEDY" --skip_stage2 "$GENERAL_SKIP_STAGE2")
    [ -n "$GENERAL_STAGE2_POLICY" ] && CMD+=(--general_stage2_policy "$GENERAL_STAGE2_POLICY")
    [ -n "$GENERAL_POLICY_DIR" ] && CMD+=(--general_policy_dir "$GENERAL_POLICY_DIR")
    [ -n "$GENERAL_ACCURACY_TOLERANCES" ] && CMD+=(--accuracy_tolerance "$(printf '%s' "$GENERAL_ACCURACY_TOLERANCES" | cut -d, -f1 | xargs)")
  fi
elif [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  CMD=(python rl_ga_compare_runner.py --base-model "$BASE_MODEL" --data-path "$DATA_PATH" --dataset "$DATASET" --output-dir "$RUN_ROOT" --model-type "$MODEL_TYPE" --batch-size "$BATCH_SIZE" --stage1-search-episodes "$STAGE1_EPISODES" --stage2-search-episodes "$STAGE2_EPISODES" --stage1-search-generations "$STAGE1_GENERATIONS" --stage2-search-generations "$STAGE2_GENERATIONS" --stage1-search-lr "$STAGE1_LR" --stage2-search-lr "$STAGE2_LR" --random-seed "$RANDOM_SEED" --perm-trials "$PERM_TRIALS" --cost-trials "$COST_TRIALS" --budget-trials "$BUDGET_TRIALS" --stage2-compare-repeats "$STAGE2_COMPARE_REPEATS" --compare-config-mode "$COMPARE_CONFIG_MODE" --stage1-accuracy-tolerance "$STAGE1_ACCURACY_TOLERANCE" --stage2-limit-tolerance "$STAGE2_LIMIT_TOLERANCE" --stage2-stability-tolerance "$STAGE2_STABILITY_TOLERANCE" --stage2-k-trials "$STAGE2_K_TRIALS" --stage2-probe-size "$STAGE2_PROBE_SIZE")
  if [ "$COMPARE_CONFIG_MODE" = "direct" ]; then
    CMD+=(--rl-compare-stage1-json "$RL_COMPARE_STAGE1_JSON" --rl-compare-stage2-json "$RL_COMPARE_STAGE2_JSON" --ga-compare-stage1-json "$GA_COMPARE_STAGE1_JSON" --ga-compare-stage2-json "$GA_COMPARE_STAGE2_JSON")
  else
    CMD+=(--compare-persistent-root "$COMPARE_PERSISTENT_ROOT")
    CMD+=(--rl-compare-stage1-accuracy-tolerance "$RL_COMPARE_STAGE1_ACCURACY_TOLERANCE" --rl-compare-stage2-limit-tolerance "$RL_COMPARE_STAGE2_LIMIT_TOLERANCE" --rl-compare-stage2-stability-tolerance "$RL_COMPARE_STAGE2_STABILITY_TOLERANCE")
    CMD+=(--ga-compare-stage1-accuracy-tolerance "$GA_COMPARE_STAGE1_ACCURACY_TOLERANCE" --ga-compare-stage2-limit-tolerance "$GA_COMPARE_STAGE2_LIMIT_TOLERANCE" --ga-compare-stage2-stability-tolerance "$GA_COMPARE_STAGE2_STABILITY_TOLERANCE")
  fi
else
  if [ "$SEARCH_ALGORITHM" = "rl" ]; then
    RL_STAGE1_EPISODES_SPECIFIED="$S_STAGE1_EPISODES"
    RL_STAGE2_EPISODES_SPECIFIED="$S_STAGE2_EPISODES"
    [ "$SKIP_STAGE1_SEARCH" = "true" ] && RL_STAGE1_EPISODES_SPECIFIED="false"
    [ "$SKIP_NOISE_SEARCH" = "true" ] && RL_STAGE2_EPISODES_SPECIFIED="false"
    CMD=(python rl_tune.py --base_model "$BASE_MODEL" --data_path "$DATA_PATH" --output_dir "$RUN_ROOT" --batch_size "$BATCH_SIZE" --micro_batch_size "$BATCH_SIZE" --num_epochs 1 --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 --eval_step 80 --adapter_name lora --target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]" --stage1_rl_episodes "$STAGE1_EPISODES" --stage2_rl_episodes "$STAGE2_EPISODES" --stage1_rl_episodes_specified "$RL_STAGE1_EPISODES_SPECIFIED" --stage2_rl_episodes_specified "$RL_STAGE2_EPISODES_SPECIFIED" --ppo_update_interval "$PPO_UPDATE_INTERVAL_VAL" --use_ist --final_eval_config_source "$FINAL_EVAL_SOURCE" --final_eval_config_path "$FINAL_EVAL_CONFIG" --manual_stage1_gelu "$MANUAL_STAGE1_GELU" --manual_stage1_softmax "$MANUAL_STAGE1_SOFTMAX" --manual_stage2_noise "$MANUAL_STAGE2_NOISE" --stage2_fixed_config_source "$STAGE2_FIXED_CONFIG_SOURCE" --stage2_fixed_config_path "$STAGE2_FIXED_CONFIG" --stage2_manual_gelu "$STAGE2_MANUAL_GELU" --stage2_manual_softmax "$STAGE2_MANUAL_SOFTMAX" --final_eval_random_seed "$RANDOM_SEED" --final_eval_permutation_trials "$PERM_TRIALS" --final_eval_cost_equivalent_trials "$COST_TRIALS" --final_eval_budget_equivalent_trials "$BUDGET_TRIALS" --final_eval_repeat_n "$FINAL_EVAL_REPEAT" --skip_noise_rl "$SKIP_NOISE_SEARCH" --skip_stage1_rl "$SKIP_STAGE1_SEARCH" --skip_final_eval "$SKIP_FINAL_EVAL" --resume_run_dir "$RESUME_FROM" --stage1_rl_lr "$STAGE1_LR" --stage2_rl_lr "$STAGE2_LR" --stage1_accuracy_tolerance "$STAGE1_ACCURACY_TOLERANCE" --stage2_limit_tolerance "$STAGE2_LIMIT_TOLERANCE" --stage2_stability_tolerance "$STAGE2_STABILITY_TOLERANCE" --stage2_k_trials "$STAGE2_K_TRIALS" --stage2_probe_size "$STAGE2_PROBE_SIZE")
  else
    CMD=(python rl_tune_genetic.py --base_model "$BASE_MODEL" --data_path "$DATA_PATH" --output_dir "$RUN_ROOT" --batch_size "$BATCH_SIZE" --micro_batch_size "$BATCH_SIZE" --num_epochs 1 --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 --eval_step 80 --adapter_name lora --target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]" --use_ist --final_eval_config_source "$FINAL_EVAL_SOURCE" --final_eval_config_path "$FINAL_EVAL_CONFIG" --manual_stage1_gelu "$MANUAL_STAGE1_GELU" --manual_stage1_softmax "$MANUAL_STAGE1_SOFTMAX" --manual_stage2_noise "$MANUAL_STAGE2_NOISE" --stage2_fixed_config_source "$STAGE2_FIXED_CONFIG_SOURCE" --stage2_fixed_config_path "$STAGE2_FIXED_CONFIG" --stage2_manual_gelu "$STAGE2_MANUAL_GELU" --stage2_manual_softmax "$STAGE2_MANUAL_SOFTMAX" --final_eval_random_seed "$RANDOM_SEED" --final_eval_permutation_trials "$PERM_TRIALS" --final_eval_cost_equivalent_trials "$COST_TRIALS" --final_eval_budget_equivalent_trials "$BUDGET_TRIALS" --final_eval_repeat_n "$FINAL_EVAL_REPEAT" --skip_noise_rl "$SKIP_NOISE_SEARCH" --skip_stage1_rl "$SKIP_STAGE1_SEARCH" --skip_final_eval "$SKIP_FINAL_EVAL" --resume_run_dir "$RESUME_FROM" --stage1_accuracy_tolerance "$STAGE1_ACCURACY_TOLERANCE" --stage2_limit_tolerance "$STAGE2_LIMIT_TOLERANCE" --stage2_stability_tolerance "$STAGE2_STABILITY_TOLERANCE" --stage2_k_trials "$STAGE2_K_TRIALS" --stage2_probe_size "$STAGE2_PROBE_SIZE")
    [ "$S_STAGE1_GENERATIONS" = "true" ] && CMD+=(--stage1_ga_generations "$STAGE1_GENERATIONS" --stage1_ga_generations_specified "true")
    [ "$S_STAGE2_GENERATIONS" = "true" ] && CMD+=(--stage2_ga_generations "$STAGE2_GENERATIONS" --stage2_ga_generations_specified "true")
  fi
fi

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "启动配置："
show "搜索算法" "$(algzh "$SEARCH_ALGORITHM")" "$S_SEARCH_ALGORITHM"
show "数据集" "$DATASET" "$S_DATASET"
show "模型类型" "$(modelzh "$MODEL_TYPE")" "$S_MODEL_TYPE"
show "日志文件" "$LOGFILE_BASENAME" "$S_LOGFILE"
show "批大小" "$BATCH_SIZE" "$S_BATCH_SIZE"
show "模式目录" "$RUN_GROUP_DIR" "true"
show "运行目录" "$RUN_ROOT" "true"
if [ "$SEARCH_ALGORITHM" = "rl" ] || [ "$SEARCH_ALGORITHM" = "ga" ]; then
  show "Stage-1 准确度约束" "$STAGE1_ACCURACY_TOLERANCE" "$S_STAGE1_ACCURACY_TOLERANCE"
  show "Stage-2 指标约束百分比" "$STAGE2_LIMIT_TOLERANCE" "$S_STAGE2_LIMIT_TOLERANCE"
  show "Stage-2 稳定性约束百分比" "$STAGE2_STABILITY_TOLERANCE" "$S_STAGE2_STABILITY_TOLERANCE"
  show "Stage-2 K 次噪声试验" "$STAGE2_K_TRIALS" "$S_STAGE2_K_TRIALS"
  show "Stage-2 探针子集大小" "$STAGE2_PROBE_SIZE" "$S_STAGE2_PROBE_SIZE"
fi
if [ "$USE_PERSISTENT" = "true" ]; then
  show "持久化目录" "$PERSISTENT_DIR" "true"
  show "从头训练" "$(boolzh "$FRESH_START")" "$S_FRESH_START"
  if [ "$FRESH_STAGE1" = "true" ]; then show "单独重置 Stage-1" "是" "$S_FRESH_STAGE1"; fi
  if [ "$FRESH_STAGE2" = "true" ]; then show "单独重置 Stage-2" "是" "$S_FRESH_STAGE2"; fi
fi

if [ "$SEARCH_ALGORITHM" = "rl" ]; then
  show "Stage-1 回合数" "$STAGE1_EPISODES" "$S_STAGE1_EPISODES"
  show "Stage-2 回合数" "$STAGE2_EPISODES" "$S_STAGE2_EPISODES"
  show "PPO 更新间隔" "$PPO_UPDATE_INTERVAL_VAL" "$S_PPO_UPDATE_INTERVAL"
  show "Stage-1 学习率" "$STAGE1_LR" "$S_STAGE1_LR"
  show "Stage-2 学习率" "$STAGE2_LR" "$S_STAGE2_LR"
  show "最终评估来源" "$(srczh "$FINAL_EVAL_SOURCE")" "$S_FINAL_EVAL_SOURCE"
  show "Stage-2 固定 GELU/Softmax 来源" "$(srczh "$STAGE2_FIXED_CONFIG_SOURCE")" "$S_STAGE2_FIXED_CONFIG_SOURCE"
  show "跳过 Stage-1 搜索" "$(boolzh "$SKIP_STAGE1_SEARCH")" "$S_SKIP_STAGE1_SEARCH"
  show "跳过 Stage-2 搜索" "$(boolzh "$SKIP_NOISE_SEARCH")" "$S_SKIP_NOISE_SEARCH"
  show "跳过最终评估" "$(boolzh "$SKIP_FINAL_EVAL")" "$S_SKIP_FINAL_EVAL"
elif [ "$SEARCH_ALGORITHM" = "ga" ]; then
  show "Stage-1 迭代代数" "$STAGE1_GENERATIONS" "$S_STAGE1_GENERATIONS"
  show "Stage-2 迭代代数" "$STAGE2_GENERATIONS" "$S_STAGE2_GENERATIONS"
  show "最终评估来源" "$(srczh "$FINAL_EVAL_SOURCE")" "$S_FINAL_EVAL_SOURCE"
  show "Stage-2 固定 GELU/Softmax 来源" "$(srczh "$STAGE2_FIXED_CONFIG_SOURCE")" "$S_STAGE2_FIXED_CONFIG_SOURCE"
  show "跳过 Stage-1 搜索" "$(boolzh "$SKIP_STAGE1_SEARCH")" "$S_SKIP_STAGE1_SEARCH"
  show "跳过 Stage-2 搜索" "$(boolzh "$SKIP_NOISE_SEARCH")" "$S_SKIP_NOISE_SEARCH"
  show "跳过最终评估" "$(boolzh "$SKIP_FINAL_EVAL")" "$S_SKIP_FINAL_EVAL"
elif [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  show "对比配置模式" "$COMPARE_CONFIG_MODE" "$S_COMPARE_CONFIG_MODE"
  show "Stage-2 对比重复次数" "$STAGE2_COMPARE_REPEATS" "$S_STAGE2_COMPARE_REPEATS"
  if [ "$COMPARE_CONFIG_MODE" = "direct" ]; then
    echo "  RL Stage-1 JSON：$RL_COMPARE_STAGE1_JSON"
    echo "  RL Stage-2 JSON：$RL_COMPARE_STAGE2_JSON"
    echo "  GA Stage-1 JSON：$GA_COMPARE_STAGE1_JSON"
    echo "  GA Stage-2 JSON：$GA_COMPARE_STAGE2_JSON"
  else
    echo "  持久化目录根路径：$COMPARE_PERSISTENT_ROOT"
    echo "  RL 约束：s1_tol=$RL_COMPARE_STAGE1_ACCURACY_TOLERANCE, s2_limit_tol=$RL_COMPARE_STAGE2_LIMIT_TOLERANCE, s2_stability_tol=$RL_COMPARE_STAGE2_STABILITY_TOLERANCE"
    echo "  GA 约束：s1_tol=$GA_COMPARE_STAGE1_ACCURACY_TOLERANCE, s2_limit_tol=$GA_COMPARE_STAGE2_LIMIT_TOLERANCE, s2_stability_tol=$GA_COMPARE_STAGE2_STABILITY_TOLERANCE"
    echo "  RL 持久化目录：$RL_COMPARE_PERSISTENT_DIR"
    echo "  GA 持久化目录：$GA_COMPARE_PERSISTENT_DIR"
  fi
  echo "  对比结果目录：${RUN_ROOT}/reports"
  echo "  元信息目录：${RUN_ROOT}/meta"
  echo "  说明：该模式不会重新启动 RL/GA 完整训练，而是直接使用显式 JSON 或持久化目录中的已有结果生成对比报告。"
else
  show "通用强化学习模式" "$GENERAL_MODE" "$S_GENERAL_MODE"
  show "跳过 Stage-2" "$(boolzh "$GENERAL_SKIP_STAGE2")" "$S_GENERAL_SKIP_STAGE2"
  if [ "$GENERAL_MODE" = "train" ]; then
    # 泛化模式显示
    case "$_GEN_SUBDIR" in
      dataset_gen) show "泛化模式" "数据集泛化" "true" ;;
      accuracy_gen) show "泛化模式" "准确度泛化" "true" ;;
      combined_gen) show "泛化模式" "数据集 + 准确度联合泛化" "true" ;;
      *) show "泛化模式" "单任务单容忍" "true" ;;
    esac
    show "任务集合标识" "$GENERAL_TASKSET_ID" "true"
    show "准确度标识" "$GENERAL_ACCURACY_SLUG" "true"
    show "训练任务" "${GENERAL_TASKS:-$DATASET}" "$S_GENERAL_TASKS"
    [ -n "$GENERAL_ACCURACY_TOLERANCES" ] && show "准确度容忍值" "$GENERAL_ACCURACY_TOLERANCES" "$S_GENERAL_ACCURACY_TOLERANCES"
    [ -n "$GENERAL_ACCURACY_TOLERANCE_RANGE" ] && show "准确度容忍范围" "$GENERAL_ACCURACY_TOLERANCE_RANGE" "true"
    show "训练轮数" "$GENERAL_ROUNDS" "$S_GENERAL_ROUNDS"
    show "PPO 更新间隔" "$PPO_UPDATE_INTERVAL_VAL" "$S_PPO_UPDATE_INTERVAL"
    show "通用策略学习率" "$GENERAL_LR" "$S_GENERAL_LR"
    show "从头训练" "$(boolzh "$FRESH_START")" "$S_FRESH_START"
  else
    [ -n "$GENERAL_POLICY_DIR" ] && show "策略目录" "$GENERAL_POLICY_DIR" "$S_GENERAL_POLICY_DIR"
    show "Stage-1 策略文件" "$GENERAL_STAGE1_POLICY" "$S_GENERAL_STAGE1_POLICY"
    show "离线 rollout 次数" "$GENERAL_NUM_ROLLOUTS" "$S_GENERAL_NUM_ROLLOUTS"
    show "是否贪心 rollout" "$(boolzh "$GENERAL_GREEDY")" "$S_GENERAL_GREEDY"
    [ -n "$GENERAL_STAGE2_POLICY" ] && show "Stage-2 策略文件" "$GENERAL_STAGE2_POLICY" "$S_GENERAL_STAGE2_POLICY"
    [ -n "$GENERAL_ACCURACY_TOLERANCES" ] && show "推断准确度容忍" "$GENERAL_ACCURACY_TOLERANCES" "$S_GENERAL_ACCURACY_TOLERANCES"
  fi
fi

[ -n "$RESUME_FROM" ] && show "恢复目录" "$RESUME_FROM" "$S_RESUME_FROM"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "  CUDA_VISIBLE_DEVICES：${CUDA_VISIBLE_DEVICES}（显式指定）"
else
  echo "  CUDA_VISIBLE_DEVICES：未设置（使用默认值）"
fi
echo "  实际执行命令（Command）：$CMD_STR"

nohup "${CMD[@]}" > "$LOGFILE_PATH" 2>&1 &
JOB_PID=$!
if [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  mkdir -p "${RUN_ROOT}/meta"
  echo "$JOB_PID" > "${RUN_ROOT}/meta/compare_launcher.pid"
  echo "$RUN_ROOT" > "${LATEST_BASE_DIR}/LATEST_RUN_DIR"
  echo "$JOB_PID" > "${LATEST_BASE_DIR}/LATEST_PID"
  echo "$RUN_ROOT" > "${LATEST_BASE_DIR}/LATEST_COMPARE_RUN_DIR"
  echo "$JOB_PID" > "${LATEST_BASE_DIR}/LATEST_COMPARE_PID"
  rm -f "${LATEST_BASE_DIR}/LATEST_RL_PID" "${LATEST_BASE_DIR}/LATEST_GA_PID"
else
  echo "$JOB_PID" > "${RUN_ROOT}/run.pid"
  echo "$JOB_PID" > "${RUN_ROOT}/rl.pid"
  echo "$RUN_ROOT" > "${LATEST_BASE_DIR}/LATEST_RUN_DIR"
  echo "$JOB_PID" > "${LATEST_BASE_DIR}/LATEST_PID"
fi

echo
echo "已在后台启动。"
echo "  进程号（PID）：$JOB_PID"
echo "  查看日志：tail -f $LOGFILE_PATH"
echo "  错误摘要：$ERROR_SUMMARY_PATH"
echo "  LATEST_RUN_DIR：${LATEST_BASE_DIR}/LATEST_RUN_DIR"
echo "  LATEST_PID：${LATEST_BASE_DIR}/LATEST_PID"
if [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  echo "  优雅停止（Graceful Stop）：kill -INT $JOB_PID"
  echo "  Compare 进程 PID 文件：${RUN_ROOT}/meta/compare.pid"
  echo "  Compare 元信息：${RUN_ROOT}/meta/compare_metadata.json"
  echo "  Compare 运行状态：${RUN_ROOT}/meta/compare_status.json"
  echo "  Compare 最终状态：${RUN_ROOT}/meta/compare_final_status.json"
  echo "  Stage-1 对比报告：${RUN_ROOT}/reports/stage1_compare_report_${DATASET}.md"
  echo "  Stage-2 对比报告：${RUN_ROOT}/reports/stage2_compare_report_${DATASET}.md"
else
  echo "  优雅停止（Graceful Stop）：kill -INT $JOB_PID"
fi
