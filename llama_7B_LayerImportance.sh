#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
用法：
  bash llama_7B_LayerImportance.sh [可选参数]

核心参数：
  --dataset DATASET
  --search-algorithm rl|ga|general-rl|rl-and-ga-compare
  --logfile FILE
  --model-type bert-base|bert-large|gpt-2
  --batch-size N

普通 RL / 对比实验中的 RL：
  --stage1-search-episodes N
  --stage2-search-episodes N

GA / 对比实验中的 GA：
  --stage1-search-generations N
  --stage2-search-generations N

普通 RL / GA 共用：
  --skip-stage1-search
  --skip-noise-search
  --skip-stage1-final-eval
  --skip-noise-final-eval
  --final-eval-source search|json|manual
  --final-eval-config PATH
  --manual-gelu JSON_ARRAY
  --manual-softmax JSON_ARRAY
  --noise-eval-source search|json|manual
  --noise-eval-config PATH
  --manual-noise-config JSON_OBJECT
  --noise-eval-repeat N
  --random-seed N
  --perm-trials N
  --cost-trials N
  --budget-trials N
  --resume-from PATH

对比实验专用（仅 rl-and-ga-compare 可用）：
  --rl-skip-stage1-search
  --ga-skip-stage1-search
  --rl-final-eval-source search|json
  --ga-final-eval-source search|json
  --rl-final-eval-config PATH
  --ga-final-eval-config PATH
  --rl-skip-noise-search
  --ga-skip-noise-search
  --rl-noise-eval-source search|json
  --ga-noise-eval-source search|json
  --rl-noise-eval-config PATH
  --ga-noise-eval-config PATH

普通 RL（仅 rl 可用）：
  --stage1-search-lr FLOAT
  --stage2-search-lr FLOAT

通用 RL（仅 general-rl 可用）：
  --general-rl-mode train|infer
  --general-rl-tasks TASK1,TASK2,...
  --general-rl-rounds N
  --general-rl-episodes-per-round N
  --general-rl-lr FLOAT
  --general-rl-num-rollouts N
  --general-rl-greedy
  --general-stage1-policy PATH
  --general-stage2-policy PATH
  --general-rl-skip-stage2
  --general-rl-stage1-config-json PATH

说明：
  1. 不再支持位置参数，统一改为可选参数。
  2. 参数 --model 已废弃，请改用 --dataset。
  3. 以下旧参数已从命令行入口移除，因为当前流程不会实际生效：
     lora_r、lora_alpha、degree
  4. --search-algorithm=rl-and-ga-compare 会同时启动一个普通 RL 进程和一个 GA 进程，
     两者使用独立输出目录，并在 Stage-1 / Stage-2 结束后自动生成对比文本和对比图。

示例：
  bash llama_7B_LayerImportance.sh --dataset mrpc
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm rl --stage1-search-lr 3e-5 --stage2-search-lr 1e-5
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm ga --stage1-search-generations 120 --stage2-search-generations 90
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm rl-and-ga-compare --stage1-search-generations 1594 --stage2-search-generations 1250
  bash llama_7B_LayerImportance.sh --dataset mrpc --search-algorithm general-rl --general-rl-mode train --general-rl-tasks mrpc,cola,rte,stsb
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
default_stage1_json_for_family(){
  [ "$1" = "ga" ] && echo "glue_configs_best_genetic.json" || echo "glue_configs_best_ppo.json"
}
default_stage2_json_for_family(){
  [ "$1" = "ga" ] && echo "glue_noise_configs_best_genetic.json" || echo "glue_noise_configs_best_ppo.json"
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
SKIP_STAGE1_FINAL_EVAL="false"; S_SKIP_STAGE1_FINAL_EVAL="false"
SKIP_NOISE_FINAL_EVAL="false"; S_SKIP_NOISE_FINAL_EVAL="false"
FINAL_EVAL_SOURCE="search"; S_FINAL_EVAL_SOURCE="false"
FINAL_EVAL_CONFIG=""; S_FINAL_EVAL_CONFIG="false"
MANUAL_GELU=""
MANUAL_SOFTMAX=""
NOISE_EVAL_SOURCE="search"; S_NOISE_EVAL_SOURCE="false"
NOISE_EVAL_CONFIG=""; S_NOISE_EVAL_CONFIG="false"
MANUAL_NOISE_CONFIG=""
NOISE_EVAL_REPEAT="1"; S_NOISE_EVAL_REPEAT="false"
RANDOM_SEED="42"; S_RANDOM_SEED="false"
PERM_TRIALS="10"; S_PERM_TRIALS="false"
COST_TRIALS="10"; S_COST_TRIALS="false"
BUDGET_TRIALS="10"; S_BUDGET_TRIALS="false"
GENERAL_MODE="infer"; S_GENERAL_MODE="false"
GENERAL_TASKS=""; S_GENERAL_TASKS="false"
GENERAL_ROUNDS="50"; S_GENERAL_ROUNDS="false"
GENERAL_EPISODES_PER_ROUND="170"; S_GENERAL_EPISODES_PER_ROUND="false"
GENERAL_LR="3e-5"; S_GENERAL_LR="false"
GENERAL_NUM_ROLLOUTS="500"; S_GENERAL_NUM_ROLLOUTS="false"
GENERAL_GREEDY="false"; S_GENERAL_GREEDY="false"
GENERAL_STAGE1_POLICY=""; S_GENERAL_STAGE1_POLICY="false"
GENERAL_STAGE2_POLICY=""; S_GENERAL_STAGE2_POLICY="false"
GENERAL_SKIP_STAGE2="false"; S_GENERAL_SKIP_STAGE2="false"
GENERAL_STAGE1_CONFIG_JSON=""; S_GENERAL_STAGE1_CONFIG_JSON="false"
RL_COMPARE_SKIP_STAGE1_SEARCH="false"; S_RL_COMPARE_SKIP_STAGE1_SEARCH="false"
GA_COMPARE_SKIP_STAGE1_SEARCH="false"; S_GA_COMPARE_SKIP_STAGE1_SEARCH="false"
RL_COMPARE_FINAL_EVAL_SOURCE="search"; S_RL_COMPARE_FINAL_EVAL_SOURCE="false"
GA_COMPARE_FINAL_EVAL_SOURCE="search"; S_GA_COMPARE_FINAL_EVAL_SOURCE="false"
RL_COMPARE_FINAL_EVAL_CONFIG=""; S_RL_COMPARE_FINAL_EVAL_CONFIG="false"
GA_COMPARE_FINAL_EVAL_CONFIG=""; S_GA_COMPARE_FINAL_EVAL_CONFIG="false"
RL_COMPARE_SKIP_NOISE_SEARCH="false"; S_RL_COMPARE_SKIP_NOISE_SEARCH="false"
GA_COMPARE_SKIP_NOISE_SEARCH="false"; S_GA_COMPARE_SKIP_NOISE_SEARCH="false"
RL_COMPARE_NOISE_EVAL_SOURCE="search"; S_RL_COMPARE_NOISE_EVAL_SOURCE="false"
GA_COMPARE_NOISE_EVAL_SOURCE="search"; S_GA_COMPARE_NOISE_EVAL_SOURCE="false"
RL_COMPARE_NOISE_EVAL_CONFIG=""; S_RL_COMPARE_NOISE_EVAL_CONFIG="false"
GA_COMPARE_NOISE_EVAL_CONFIG=""; S_GA_COMPARE_NOISE_EVAL_CONFIG="false"
RESUME_FROM=""; S_RESUME_FROM="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
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
    --skip-stage1-final-eval) SKIP_STAGE1_FINAL_EVAL="true"; S_SKIP_STAGE1_FINAL_EVAL="true"; shift ;;
    --skip-noise-final-eval) SKIP_NOISE_FINAL_EVAL="true"; S_SKIP_NOISE_FINAL_EVAL="true"; shift ;;
    --final-eval-source) needv "$@"; FINAL_EVAL_SOURCE="$2"; S_FINAL_EVAL_SOURCE="true"; shift 2 ;;
    --final-eval-config) needv "$@"; FINAL_EVAL_CONFIG="$2"; S_FINAL_EVAL_CONFIG="true"; shift 2 ;;
    --manual-gelu) needv "$@"; MANUAL_GELU="$2"; shift 2 ;;
    --manual-softmax) needv "$@"; MANUAL_SOFTMAX="$2"; shift 2 ;;
    --noise-eval-source) needv "$@"; NOISE_EVAL_SOURCE="$2"; S_NOISE_EVAL_SOURCE="true"; shift 2 ;;
    --noise-eval-config) needv "$@"; NOISE_EVAL_CONFIG="$2"; S_NOISE_EVAL_CONFIG="true"; shift 2 ;;
    --manual-noise-config) needv "$@"; MANUAL_NOISE_CONFIG="$2"; shift 2 ;;
    --noise-eval-repeat) needv "$@"; NOISE_EVAL_REPEAT="$2"; S_NOISE_EVAL_REPEAT="true"; shift 2 ;;
    --random-seed) needv "$@"; RANDOM_SEED="$2"; S_RANDOM_SEED="true"; shift 2 ;;
    --perm-trials) needv "$@"; PERM_TRIALS="$2"; S_PERM_TRIALS="true"; shift 2 ;;
    --cost-trials) needv "$@"; COST_TRIALS="$2"; S_COST_TRIALS="true"; shift 2 ;;
    --budget-trials) needv "$@"; BUDGET_TRIALS="$2"; S_BUDGET_TRIALS="true"; shift 2 ;;
    --general-rl-mode) needv "$@"; GENERAL_MODE="$2"; S_GENERAL_MODE="true"; shift 2 ;;
    --general-rl-tasks) needv "$@"; GENERAL_TASKS="$2"; S_GENERAL_TASKS="true"; shift 2 ;;
    --general-rl-rounds) needv "$@"; GENERAL_ROUNDS="$2"; S_GENERAL_ROUNDS="true"; shift 2 ;;
    --general-rl-episodes-per-round) needv "$@"; GENERAL_EPISODES_PER_ROUND="$2"; S_GENERAL_EPISODES_PER_ROUND="true"; shift 2 ;;
    --general-rl-lr) needv "$@"; GENERAL_LR="$2"; S_GENERAL_LR="true"; shift 2 ;;
    --general-rl-num-rollouts) needv "$@"; GENERAL_NUM_ROLLOUTS="$2"; S_GENERAL_NUM_ROLLOUTS="true"; shift 2 ;;
    --general-rl-greedy) GENERAL_GREEDY="true"; S_GENERAL_GREEDY="true"; shift ;;
    --general-stage1-policy) needv "$@"; GENERAL_STAGE1_POLICY="$2"; S_GENERAL_STAGE1_POLICY="true"; shift 2 ;;
    --general-stage2-policy) needv "$@"; GENERAL_STAGE2_POLICY="$2"; S_GENERAL_STAGE2_POLICY="true"; shift 2 ;;
    --general-rl-skip-stage2) GENERAL_SKIP_STAGE2="true"; S_GENERAL_SKIP_STAGE2="true"; shift ;;
    --general-rl-stage1-config-json) needv "$@"; GENERAL_STAGE1_CONFIG_JSON="$2"; S_GENERAL_STAGE1_CONFIG_JSON="true"; shift 2 ;;
    --rl-skip-stage1-search) RL_COMPARE_SKIP_STAGE1_SEARCH="true"; S_RL_COMPARE_SKIP_STAGE1_SEARCH="true"; shift ;;
    --ga-skip-stage1-search) GA_COMPARE_SKIP_STAGE1_SEARCH="true"; S_GA_COMPARE_SKIP_STAGE1_SEARCH="true"; shift ;;
    --rl-final-eval-source) needv "$@"; RL_COMPARE_FINAL_EVAL_SOURCE="$2"; S_RL_COMPARE_FINAL_EVAL_SOURCE="true"; shift 2 ;;
    --ga-final-eval-source) needv "$@"; GA_COMPARE_FINAL_EVAL_SOURCE="$2"; S_GA_COMPARE_FINAL_EVAL_SOURCE="true"; shift 2 ;;
    --rl-final-eval-config) needv "$@"; RL_COMPARE_FINAL_EVAL_CONFIG="$2"; S_RL_COMPARE_FINAL_EVAL_CONFIG="true"; shift 2 ;;
    --ga-final-eval-config) needv "$@"; GA_COMPARE_FINAL_EVAL_CONFIG="$2"; S_GA_COMPARE_FINAL_EVAL_CONFIG="true"; shift 2 ;;
    --rl-skip-noise-search) RL_COMPARE_SKIP_NOISE_SEARCH="true"; S_RL_COMPARE_SKIP_NOISE_SEARCH="true"; shift ;;
    --ga-skip-noise-search) GA_COMPARE_SKIP_NOISE_SEARCH="true"; S_GA_COMPARE_SKIP_NOISE_SEARCH="true"; shift ;;
    --rl-noise-eval-source) needv "$@"; RL_COMPARE_NOISE_EVAL_SOURCE="$2"; S_RL_COMPARE_NOISE_EVAL_SOURCE="true"; shift 2 ;;
    --ga-noise-eval-source) needv "$@"; GA_COMPARE_NOISE_EVAL_SOURCE="$2"; S_GA_COMPARE_NOISE_EVAL_SOURCE="true"; shift 2 ;;
    --rl-noise-eval-config) needv "$@"; RL_COMPARE_NOISE_EVAL_CONFIG="$2"; S_RL_COMPARE_NOISE_EVAL_CONFIG="true"; shift 2 ;;
    --ga-noise-eval-config) needv "$@"; GA_COMPARE_NOISE_EVAL_CONFIG="$2"; S_GA_COMPARE_NOISE_EVAL_CONFIG="true"; shift 2 ;;
    --resume-from) needv "$@"; RESUME_FROM="$2"; S_RESUME_FROM="true"; shift 2 ;;
    --*) err "不支持的参数：$1" ;;
    *) err "不再支持位置参数：$1。请改用 --dataset mrpc 这种写法。" ;;
  esac
done

DATASET="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')"
SEARCH_ALGORITHM="$(printf '%s' "$SEARCH_ALGORITHM" | tr '[:upper:]' '[:lower:]')"
MODEL_TYPE="$(printf '%s' "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')"
FINAL_EVAL_SOURCE="$(printf '%s' "$FINAL_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
NOISE_EVAL_SOURCE="$(printf '%s' "$NOISE_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
GENERAL_MODE="$(printf '%s' "$GENERAL_MODE" | tr '[:upper:]' '[:lower:]')"
RL_COMPARE_FINAL_EVAL_SOURCE="$(printf '%s' "$RL_COMPARE_FINAL_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
GA_COMPARE_FINAL_EVAL_SOURCE="$(printf '%s' "$GA_COMPARE_FINAL_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
RL_COMPARE_NOISE_EVAL_SOURCE="$(printf '%s' "$RL_COMPARE_NOISE_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"
GA_COMPARE_NOISE_EVAL_SOURCE="$(printf '%s' "$GA_COMPARE_NOISE_EVAL_SOURCE" | tr '[:upper:]' '[:lower:]')"

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
case "$FINAL_EVAL_SOURCE" in search|json|manual) ;; *) err "不支持的第一阶段评估来源：$FINAL_EVAL_SOURCE" ;; esac
case "$NOISE_EVAL_SOURCE" in search|json|manual) ;; *) err "不支持的第二阶段评估来源：$NOISE_EVAL_SOURCE" ;; esac
case "$GENERAL_MODE" in train|infer) ;; *) err "general-rl 模式必须是 train 或 infer，当前为：$GENERAL_MODE" ;; esac

if [ "$S_STAGE1_GENERATIONS" = "false" ]; then
  STAGE1_GENERATIONS="$(ga_default_stage1_generations_for_model "$MODEL_TYPE")"
fi
if [ "$S_STAGE2_GENERATIONS" = "false" ]; then
  STAGE2_GENERATIONS="$(ga_default_stage2_generations_for_model "$MODEL_TYPE")"
fi

is_pos_int "$BATCH_SIZE" || err "--batch-size 必须是正整数，当前为：$BATCH_SIZE"
is_pos_int "$NOISE_EVAL_REPEAT" || err "--noise-eval-repeat 必须是正整数，当前为：$NOISE_EVAL_REPEAT"
[ -z "$RESUME_FROM" ] || [ -d "$RESUME_FROM" ] || err "--resume-from 指定的目录不存在：$RESUME_FROM"

case "$DATASET" in
  mrpc|sst2|stsb|cola|qnli|rte|wnli) DATA_PATH="$DATASET" ;;
  *) err "不支持的数据集：$DATASET" ;;
esac

if [ "$SEARCH_ALGORITHM" != "general-rl" ] && [ "$SEARCH_ALGORITHM" != "rl-and-ga-compare" ]; then
  [ "$S_FINAL_EVAL_CONFIG" = "true" ] || FINAL_EVAL_CONFIG=$([ "$SEARCH_ALGORITHM" = "ga" ] && echo glue_configs_best_genetic.json || echo glue_configs_best_ppo.json)
  [ "$S_NOISE_EVAL_CONFIG" = "true" ] || NOISE_EVAL_CONFIG=$([ "$SEARCH_ALGORITHM" = "ga" ] && echo glue_noise_configs_best_genetic.json || echo glue_noise_configs_best_ppo.json)
fi

if [ "$SEARCH_ALGORITHM" != "rl-and-ga-compare" ]; then
  { [ "$S_RL_COMPARE_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_GA_COMPARE_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_RL_COMPARE_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_GA_COMPARE_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_RL_COMPARE_FINAL_EVAL_CONFIG" = "false" ] && [ "$S_GA_COMPARE_FINAL_EVAL_CONFIG" = "false" ] && [ "$S_RL_COMPARE_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_GA_COMPARE_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_RL_COMPARE_NOISE_EVAL_SOURCE" = "false" ] && [ "$S_GA_COMPARE_NOISE_EVAL_SOURCE" = "false" ] && [ "$S_RL_COMPARE_NOISE_EVAL_CONFIG" = "false" ] && [ "$S_GA_COMPARE_NOISE_EVAL_CONFIG" = "false" ]; } || err "当前模式不是 rl-and-ga-compare，请不要使用 --rl-* / --ga-* 的对比专用参数。"
fi

if [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
  { [ "$S_STAGE1_EPISODES" = "false" ] && [ "$S_STAGE2_EPISODES" = "false" ] && [ "$S_STAGE1_GENERATIONS" = "false" ] && [ "$S_STAGE2_GENERATIONS" = "false" ] && [ "$S_STAGE1_LR" = "false" ] && [ "$S_STAGE2_LR" = "false" ] && [ "$S_SKIP_STAGE1_SEARCH" = "false" ] && [ "$S_SKIP_NOISE_SEARCH" = "false" ] && [ "$S_SKIP_STAGE1_FINAL_EVAL" = "false" ] && [ "$S_SKIP_NOISE_FINAL_EVAL" = "false" ] && [ "$S_FINAL_EVAL_SOURCE" = "false" ] && [ "$S_FINAL_EVAL_CONFIG" = "false" ] && [ -z "$MANUAL_GELU" ] && [ -z "$MANUAL_SOFTMAX" ] && [ "$S_NOISE_EVAL_SOURCE" = "false" ] && [ "$S_NOISE_EVAL_CONFIG" = "false" ] && [ -z "$MANUAL_NOISE_CONFIG" ]; } || err "general-rl 不能与普通 RL / GA 的阶段搜索或最终评估参数混用。"
  if [ "$GENERAL_MODE" = "train" ]; then
    is_pos_int "$GENERAL_ROUNDS" || err "--general-rl-rounds 必须是正整数"
    is_pos_int "$GENERAL_EPISODES_PER_ROUND" || err "--general-rl-episodes-per-round 必须是正整数"
    is_pos_num "$GENERAL_LR" || err "--general-rl-lr 必须是正数"
    [ "$S_GENERAL_NUM_ROLLOUTS" = "false" ] && [ "$S_GENERAL_GREEDY" = "false" ] && [ "$S_GENERAL_STAGE1_POLICY" = "false" ] && [ "$S_GENERAL_STAGE2_POLICY" = "false" ] || err "general-rl train 模式不能使用 rollout / policy 参数。"
  else
    [ -n "$GENERAL_STAGE1_POLICY" ] || err "general-rl infer 模式必须提供 --general-stage1-policy。"
    [ -f "$GENERAL_STAGE1_POLICY" ] || err "--general-stage1-policy 指定的文件不存在：$GENERAL_STAGE1_POLICY"
    [ -z "$GENERAL_STAGE2_POLICY" ] || [ -f "$GENERAL_STAGE2_POLICY" ] || err "--general-stage2-policy 指定的文件不存在：$GENERAL_STAGE2_POLICY"
    is_pos_int "$GENERAL_NUM_ROLLOUTS" || err "--general-rl-num-rollouts 必须是正整数"
    { [ "$S_GENERAL_TASKS" = "false" ] && [ "$S_GENERAL_ROUNDS" = "false" ] && [ "$S_GENERAL_EPISODES_PER_ROUND" = "false" ] && [ "$S_GENERAL_LR" = "false" ] && [ "$S_GENERAL_STAGE1_CONFIG_JSON" = "false" ] && [ "$S_RESUME_FROM" = "false" ]; } || err "general-rl infer 模式不能使用训练专用参数。"
  fi
elif [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  { [ "$S_GENERAL_MODE" = "false" ] && [ "$S_GENERAL_TASKS" = "false" ] && [ "$S_GENERAL_ROUNDS" = "false" ] && [ "$S_GENERAL_EPISODES_PER_ROUND" = "false" ] && [ "$S_GENERAL_LR" = "false" ] && [ "$S_GENERAL_NUM_ROLLOUTS" = "false" ] && [ "$S_GENERAL_GREEDY" = "false" ] && [ "$S_GENERAL_STAGE1_POLICY" = "false" ] && [ "$S_GENERAL_STAGE2_POLICY" = "false" ] && [ "$S_GENERAL_SKIP_STAGE2" = "false" ] && [ "$S_GENERAL_STAGE1_CONFIG_JSON" = "false" ]; } || err "rl-and-ga-compare 不能与 general-rl 参数混用。"
  is_pos_int "$STAGE1_EPISODES" || err "--stage1-search-episodes 必须是正整数"
  is_pos_int "$STAGE2_EPISODES" || err "--stage2-search-episodes 必须是正整数"
  is_pos_int "$STAGE1_GENERATIONS" || err "--stage1-search-generations 必须是正整数"
  is_pos_int "$STAGE2_GENERATIONS" || err "--stage2-search-generations 必须是正整数"
  is_pos_num "$STAGE1_LR" || err "--stage1-search-lr 必须是正数"
  is_pos_num "$STAGE2_LR" || err "--stage2-search-lr 必须是正数"
  [ "$SKIP_STAGE1_FINAL_EVAL" = "false" ] || err "rl-and-ga-compare 必须保留 Stage-1 最终评估，不能使用 --skip-stage1-final-eval。"
  [ "$SKIP_NOISE_FINAL_EVAL" = "false" ] || err "rl-and-ga-compare 必须保留 Stage-2 最终评估，不能使用 --skip-noise-final-eval。"
  [ "$SKIP_STAGE1_SEARCH" = "false" ] || err "rl-and-ga-compare 不使用全局 --skip-stage1-search；请改用 --rl-skip-stage1-search / --ga-skip-stage1-search。"
  [ "$SKIP_NOISE_SEARCH" = "false" ] || err "rl-and-ga-compare 不使用全局 --skip-noise-search；请改用 --rl-skip-noise-search / --ga-skip-noise-search。"
  [ "$FINAL_EVAL_SOURCE" = "search" ] || err "rl-and-ga-compare 不使用全局 --final-eval-source；请改用 --rl-final-eval-source / --ga-final-eval-source。"
  [ "$NOISE_EVAL_SOURCE" = "search" ] || err "rl-and-ga-compare 不使用全局 --noise-eval-source；请改用 --rl-noise-eval-source / --ga-noise-eval-source。"
  [ -z "$MANUAL_GELU" ] && [ -z "$MANUAL_SOFTMAX" ] && [ -z "$MANUAL_NOISE_CONFIG" ] || err "rl-and-ga-compare 不支持 manual 配置输入。"
  [ "$S_FINAL_EVAL_CONFIG" = "false" ] || err "rl-and-ga-compare 不使用全局 --final-eval-config；请改用 --rl-final-eval-config / --ga-final-eval-config。"
  [ "$S_NOISE_EVAL_CONFIG" = "false" ] || err "rl-and-ga-compare 不使用全局 --noise-eval-config；请改用 --rl-noise-eval-config / --ga-noise-eval-config。"
  [ "$S_RESUME_FROM" = "false" ] || err "当前版本的 rl-and-ga-compare 不支持 --resume-from，请分别续训 RL/GA run。"
  [ "$RL_COMPARE_FINAL_EVAL_SOURCE" = "search" ] || [ "$RL_COMPARE_FINAL_EVAL_SOURCE" = "json" ] || err "--rl-final-eval-source 只支持 search/json。"
  [ "$GA_COMPARE_FINAL_EVAL_SOURCE" = "search" ] || [ "$GA_COMPARE_FINAL_EVAL_SOURCE" = "json" ] || err "--ga-final-eval-source 只支持 search/json。"
  [ "$RL_COMPARE_NOISE_EVAL_SOURCE" = "search" ] || [ "$RL_COMPARE_NOISE_EVAL_SOURCE" = "json" ] || err "--rl-noise-eval-source 只支持 search/json。"
  [ "$GA_COMPARE_NOISE_EVAL_SOURCE" = "search" ] || [ "$GA_COMPARE_NOISE_EVAL_SOURCE" = "json" ] || err "--ga-noise-eval-source 只支持 search/json。"

  if [ "$RL_COMPARE_SKIP_STAGE1_SEARCH" = "true" ]; then
    [ "$RL_COMPARE_FINAL_EVAL_SOURCE" != "search" ] || RL_COMPARE_FINAL_EVAL_SOURCE="json"
    [ -n "$RL_COMPARE_FINAL_EVAL_CONFIG" ] || RL_COMPARE_FINAL_EVAL_CONFIG="$(default_stage1_json_for_family rl)"
  else
    [ "$RL_COMPARE_FINAL_EVAL_SOURCE" = "search" ] || err "RL 未跳过 Stage-1 搜索时，--rl-final-eval-source 必须为 search。"
    [ -z "$RL_COMPARE_FINAL_EVAL_CONFIG" ] || err "RL 未跳过 Stage-1 搜索时，不应提供 --rl-final-eval-config。"
    [ "$STAGE1_EPISODES" -ge 170 ] || err "RL 的 Stage-1 回合数至少需要 170。"
  fi
  if [ "$GA_COMPARE_SKIP_STAGE1_SEARCH" = "true" ]; then
    [ "$GA_COMPARE_FINAL_EVAL_SOURCE" != "search" ] || GA_COMPARE_FINAL_EVAL_SOURCE="json"
    [ -n "$GA_COMPARE_FINAL_EVAL_CONFIG" ] || GA_COMPARE_FINAL_EVAL_CONFIG="$(default_stage1_json_for_family ga)"
  else
    [ "$GA_COMPARE_FINAL_EVAL_SOURCE" = "search" ] || err "GA 未跳过 Stage-1 搜索时，--ga-final-eval-source 必须为 search。"
    [ -z "$GA_COMPARE_FINAL_EVAL_CONFIG" ] || err "GA 未跳过 Stage-1 搜索时，不应提供 --ga-final-eval-config。"
  fi
  if [ "$RL_COMPARE_SKIP_NOISE_SEARCH" = "true" ]; then
    [ "$RL_COMPARE_NOISE_EVAL_SOURCE" != "search" ] || RL_COMPARE_NOISE_EVAL_SOURCE="json"
    [ -n "$RL_COMPARE_NOISE_EVAL_CONFIG" ] || RL_COMPARE_NOISE_EVAL_CONFIG="$(default_stage2_json_for_family rl)"
  else
    [ "$RL_COMPARE_NOISE_EVAL_SOURCE" = "search" ] || err "RL 未跳过 Stage-2 搜索时，--rl-noise-eval-source 必须为 search。"
    [ -z "$RL_COMPARE_NOISE_EVAL_CONFIG" ] || err "RL 未跳过 Stage-2 搜索时，不应提供 --rl-noise-eval-config。"
    [ "$STAGE2_EPISODES" -ge 170 ] || err "RL 的 Stage-2 回合数至少需要 170。"
  fi
  if [ "$GA_COMPARE_SKIP_NOISE_SEARCH" = "true" ]; then
    [ "$GA_COMPARE_NOISE_EVAL_SOURCE" != "search" ] || GA_COMPARE_NOISE_EVAL_SOURCE="json"
    [ -n "$GA_COMPARE_NOISE_EVAL_CONFIG" ] || GA_COMPARE_NOISE_EVAL_CONFIG="$(default_stage2_json_for_family ga)"
  else
    [ "$GA_COMPARE_NOISE_EVAL_SOURCE" = "search" ] || err "GA 未跳过 Stage-2 搜索时，--ga-noise-eval-source 必须为 search。"
    [ -z "$GA_COMPARE_NOISE_EVAL_CONFIG" ] || err "GA 未跳过 Stage-2 搜索时，不应提供 --ga-noise-eval-config。"
  fi

  if [ "$RL_COMPARE_FINAL_EVAL_SOURCE" = "json" ]; then
    [ -f "$RL_COMPARE_FINAL_EVAL_CONFIG" ] || err "RL Stage-1 JSON 配置文件不存在：$RL_COMPARE_FINAL_EVAL_CONFIG"
    FAM="$(infer_family "$RL_COMPARE_FINAL_EVAL_CONFIG")"
    [ "$FAM" != "ga" ] || err "RL Stage-1 JSON 配置看起来属于 GA 家族：$RL_COMPARE_FINAL_EVAL_CONFIG"
  fi
  if [ "$GA_COMPARE_FINAL_EVAL_SOURCE" = "json" ]; then
    [ -f "$GA_COMPARE_FINAL_EVAL_CONFIG" ] || err "GA Stage-1 JSON 配置文件不存在：$GA_COMPARE_FINAL_EVAL_CONFIG"
    FAM="$(infer_family "$GA_COMPARE_FINAL_EVAL_CONFIG")"
    [ "$FAM" != "rl" ] || err "GA Stage-1 JSON 配置看起来属于 RL/PPO 家族：$GA_COMPARE_FINAL_EVAL_CONFIG"
  fi
  if [ "$RL_COMPARE_NOISE_EVAL_SOURCE" = "json" ]; then
    [ -f "$RL_COMPARE_NOISE_EVAL_CONFIG" ] || err "RL Stage-2 JSON 配置文件不存在：$RL_COMPARE_NOISE_EVAL_CONFIG"
    FAM="$(infer_family "$RL_COMPARE_NOISE_EVAL_CONFIG")"
    [ "$FAM" != "ga" ] || err "RL Stage-2 JSON 配置看起来属于 GA 家族：$RL_COMPARE_NOISE_EVAL_CONFIG"
  fi
  if [ "$GA_COMPARE_NOISE_EVAL_SOURCE" = "json" ]; then
    [ -f "$GA_COMPARE_NOISE_EVAL_CONFIG" ] || err "GA Stage-2 JSON 配置文件不存在：$GA_COMPARE_NOISE_EVAL_CONFIG"
    FAM="$(infer_family "$GA_COMPARE_NOISE_EVAL_CONFIG")"
    [ "$FAM" != "rl" ] || err "GA Stage-2 JSON 配置看起来属于 RL/PPO 家族：$GA_COMPARE_NOISE_EVAL_CONFIG"
  fi
else
  { [ "$S_GENERAL_MODE" = "false" ] && [ "$S_GENERAL_TASKS" = "false" ] && [ "$S_GENERAL_ROUNDS" = "false" ] && [ "$S_GENERAL_EPISODES_PER_ROUND" = "false" ] && [ "$S_GENERAL_LR" = "false" ] && [ "$S_GENERAL_NUM_ROLLOUTS" = "false" ] && [ "$S_GENERAL_GREEDY" = "false" ] && [ "$S_GENERAL_STAGE1_POLICY" = "false" ] && [ "$S_GENERAL_STAGE2_POLICY" = "false" ] && [ "$S_GENERAL_SKIP_STAGE2" = "false" ] && [ "$S_GENERAL_STAGE1_CONFIG_JSON" = "false" ]; } || err "当前搜索算法不是 general-rl，请不要使用 --general-rl-* 参数。"
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
    [ "$S_STAGE1_LR" = "false" ] && [ "$S_STAGE2_LR" = "false" ] || err "GA 不使用 PPO 学习率参数，请移除 --stage1-search-lr / --stage2-search-lr。"
  fi
  if [ "$FINAL_EVAL_SOURCE" = "manual" ]; then
    [ -n "$MANUAL_GELU" ] && [ -n "$MANUAL_SOFTMAX" ] || err "manual 第一阶段配置必须同时提供 --manual-gelu 和 --manual-softmax。"
  else
    [ -z "$MANUAL_GELU" ] && [ -z "$MANUAL_SOFTMAX" ] || err "只有 --final-eval-source=manual 时才能提供手动 GELU / Softmax 配置。"
  fi
  if [ "$NOISE_EVAL_SOURCE" = "manual" ]; then
    [ -n "$MANUAL_NOISE_CONFIG" ] || err "manual 第二阶段配置必须提供 --manual-noise-config。"
  else
    [ -z "$MANUAL_NOISE_CONFIG" ] || err "只有 --noise-eval-source=manual 时才能提供手动噪声配置。"
  fi
  [ "$SKIP_STAGE1_SEARCH" = "false" ] || [ "$FINAL_EVAL_SOURCE" != "search" ] || err "跳过 Stage-1 搜索后，不能再使用 --final-eval-source=search。"
  [ "$SKIP_STAGE1_SEARCH" = "true" ] || [ "$FINAL_EVAL_SOURCE" = "search" ] || err "执行 Stage-1 搜索时，--final-eval-source 只能是 search。"
  [ "$SKIP_NOISE_FINAL_EVAL" = "true" ] || [ "$SKIP_NOISE_SEARCH" = "false" ] || [ "$NOISE_EVAL_SOURCE" != "search" ] || err "跳过 Stage-2 搜索后，不能再使用 --noise-eval-source=search。"
  [ "$SKIP_NOISE_FINAL_EVAL" = "true" ] || [ "$SKIP_NOISE_SEARCH" = "true" ] || [ "$NOISE_EVAL_SOURCE" = "search" ] || err "执行 Stage-2 搜索且保留最终评估时，--noise-eval-source 只能是 search。"
  if [ "$FINAL_EVAL_SOURCE" = "json" ]; then [ -f "$FINAL_EVAL_CONFIG" ] || err "第一阶段 JSON 配置文件不存在：$FINAL_EVAL_CONFIG"; fi
  if [ "$NOISE_EVAL_SOURCE" = "json" ] && [ "$SKIP_NOISE_FINAL_EVAL" = "false" ]; then [ -f "$NOISE_EVAL_CONFIG" ] || err "第二阶段 JSON 配置文件不存在：$NOISE_EVAL_CONFIG"; fi
  if [ "$FINAL_EVAL_SOURCE" = "json" ]; then
    FAM="$(infer_family "$FINAL_EVAL_CONFIG")"
    [ "$SEARCH_ALGORITHM" != "ga" ] || [ "$FAM" != "rl" ] || err "已选择 ga，但第一阶段 JSON 配置看起来属于 RL/PPO 家族：$FINAL_EVAL_CONFIG"
    [ "$SEARCH_ALGORITHM" != "rl" ] || [ "$FAM" != "ga" ] || err "已选择 rl，但第一阶段 JSON 配置看起来属于 GA 家族：$FINAL_EVAL_CONFIG"
  fi
  if [ "$NOISE_EVAL_SOURCE" = "json" ] && [ "$SKIP_NOISE_FINAL_EVAL" = "false" ]; then
    FAM="$(infer_family "$NOISE_EVAL_CONFIG")"
    [ "$SEARCH_ALGORITHM" != "ga" ] || [ "$FAM" != "rl" ] || err "已选择 ga，但第二阶段 JSON 配置看起来属于 RL/PPO 家族：$NOISE_EVAL_CONFIG"
    [ "$SEARCH_ALGORITHM" != "rl" ] || [ "$FAM" != "ga" ] || err "已选择 rl，但第二阶段 JSON 配置看起来属于 GA 家族：$NOISE_EVAL_CONFIG"
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
if [ "$SEARCH_ALGORITHM" = "rl" ]; then
  RUN_GROUP_DIR="rl_results/runs/rl/${DATASET}"
elif [ "$SEARCH_ALGORITHM" = "ga" ]; then
  RUN_GROUP_DIR="rl_results/runs/ga/${DATASET}"
elif [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
  if [ "$GENERAL_MODE" = "train" ]; then
    GENERAL_TASKSET_ID="$(normalize_taskset_id "${GENERAL_TASKS:-$DATASET}")"
    RUN_GROUP_DIR="rl_results/runs/general_rl/train/${GENERAL_TASKSET_ID}"
  else
    RUN_GROUP_DIR="rl_results/runs/general_rl/infer/${DATASET}"
  fi
else
  RUN_GROUP_DIR="rl_results/runs/compare/rl_vs_ga/${DATASET}"
fi
RUN_ROOT="${RUN_GROUP_DIR}/${RUN_ID}"
LATEST_BASE_DIR="$RUN_GROUP_DIR"
LOGFILE_PATH="${RUN_ROOT}/logs/${LOGFILE_BASENAME}"
ERROR_SUMMARY_PATH="${RUN_ROOT}/logs/error_summary.txt"
mkdir -p "${RUN_ROOT}/logs"

if [ "$SEARCH_ALGORITHM" = "general-rl" ]; then
  GENERAL_DATA_PATH="$DATA_PATH"
  [ "$GENERAL_MODE" = "train" ] && [ -n "$GENERAL_TASKS" ] && GENERAL_DATA_PATH="$GENERAL_TASKS"
  CMD=(python rl_tune_general.py "$GENERAL_MODE" --model_type "$MODEL_TYPE" --data_path "$GENERAL_DATA_PATH" --output_dir "$RUN_ROOT" --batch_size "$BATCH_SIZE" --device cuda)
  if [ "$GENERAL_MODE" = "train" ]; then
    CMD+=(--total_rounds "$GENERAL_ROUNDS" --episodes_per_task_per_round "$GENERAL_EPISODES_PER_ROUND" --general_lr "$GENERAL_LR" --skip_stage2 "$GENERAL_SKIP_STAGE2")
    [ -n "$GENERAL_STAGE1_CONFIG_JSON" ] && CMD+=(--stage1_config_json "$GENERAL_STAGE1_CONFIG_JSON")
    [ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")
  else
    CMD+=(--general_stage1_policy "$GENERAL_STAGE1_POLICY" --num_rollouts "$GENERAL_NUM_ROLLOUTS" --greedy "$GENERAL_GREEDY" --skip_stage2 "$GENERAL_SKIP_STAGE2")
    [ -n "$GENERAL_STAGE2_POLICY" ] && CMD+=(--general_stage2_policy "$GENERAL_STAGE2_POLICY")
  fi
elif [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  resolve_compare_cuda_split
  CMD=(python rl_ga_compare_runner.py --base-model "$BASE_MODEL" --data-path "$DATA_PATH" --dataset "$DATASET" --output-dir "$RUN_ROOT" --batch-size "$BATCH_SIZE" --stage1-search-episodes "$STAGE1_EPISODES" --stage2-search-episodes "$STAGE2_EPISODES" --stage1-search-generations "$STAGE1_GENERATIONS" --stage2-search-generations "$STAGE2_GENERATIONS" --stage1-search-lr "$STAGE1_LR" --stage2-search-lr "$STAGE2_LR" --random-seed "$RANDOM_SEED" --perm-trials "$PERM_TRIALS" --cost-trials "$COST_TRIALS" --budget-trials "$BUDGET_TRIALS" --noise-eval-repeat "$NOISE_EVAL_REPEAT")
  [ "$RL_COMPARE_SKIP_STAGE1_SEARCH" = "true" ] && CMD+=(--rl-skip-stage1-search)
  [ "$GA_COMPARE_SKIP_STAGE1_SEARCH" = "true" ] && CMD+=(--ga-skip-stage1-search)
  CMD+=(--rl-final-eval-source "$RL_COMPARE_FINAL_EVAL_SOURCE" --ga-final-eval-source "$GA_COMPARE_FINAL_EVAL_SOURCE")
  CMD+=(--rl-final-eval-config "$RL_COMPARE_FINAL_EVAL_CONFIG" --ga-final-eval-config "$GA_COMPARE_FINAL_EVAL_CONFIG")
  [ "$RL_COMPARE_SKIP_NOISE_SEARCH" = "true" ] && CMD+=(--rl-skip-noise-search)
  [ "$GA_COMPARE_SKIP_NOISE_SEARCH" = "true" ] && CMD+=(--ga-skip-noise-search)
  CMD+=(--rl-noise-eval-source "$RL_COMPARE_NOISE_EVAL_SOURCE" --ga-noise-eval-source "$GA_COMPARE_NOISE_EVAL_SOURCE")
  CMD+=(--rl-noise-eval-config "$RL_COMPARE_NOISE_EVAL_CONFIG" --ga-noise-eval-config "$GA_COMPARE_NOISE_EVAL_CONFIG")
  [ -n "$RL_COMPARE_CUDA_VISIBLE_DEVICES" ] && CMD+=(--rl-cuda-visible-devices "$RL_COMPARE_CUDA_VISIBLE_DEVICES")
  [ -n "$GA_COMPARE_CUDA_VISIBLE_DEVICES" ] && CMD+=(--ga-cuda-visible-devices "$GA_COMPARE_CUDA_VISIBLE_DEVICES")
else
  if [ "$SEARCH_ALGORITHM" = "rl" ]; then
    CMD=(python rl_tune.py --base_model "$BASE_MODEL" --data_path "$DATA_PATH" --output_dir "$RUN_ROOT" --batch_size "$BATCH_SIZE" --micro_batch_size "$BATCH_SIZE" --num_epochs 1 --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 --eval_step 80 --adapter_name lora --target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]" --stage1_rl_episodes "$STAGE1_EPISODES" --stage2_rl_episodes "$STAGE2_EPISODES" --stage1_rl_episodes_specified "$S_STAGE1_EPISODES" --stage2_rl_episodes_specified "$S_STAGE2_EPISODES" --use_ist --final_eval_config_source "$FINAL_EVAL_SOURCE" --final_eval_config_path "$FINAL_EVAL_CONFIG" --manual_final_gelu "$MANUAL_GELU" --manual_final_softmax "$MANUAL_SOFTMAX" --final_eval_random_seed "$RANDOM_SEED" --final_eval_permutation_trials "$PERM_TRIALS" --final_eval_cost_equivalent_trials "$COST_TRIALS" --final_eval_budget_equivalent_trials "$BUDGET_TRIALS" --skip_noise_rl "$SKIP_NOISE_SEARCH" --noise_eval_config_source "$NOISE_EVAL_SOURCE" --noise_eval_config_path "$NOISE_EVAL_CONFIG" --manual_noise_config "$MANUAL_NOISE_CONFIG" --noise_eval_repeat_n "$NOISE_EVAL_REPEAT" --skip_stage1_rl "$SKIP_STAGE1_SEARCH" --skip_stage1_final_eval "$SKIP_STAGE1_FINAL_EVAL" --skip_noise_final_eval "$SKIP_NOISE_FINAL_EVAL" --resume_run_dir "$RESUME_FROM" --stage1_rl_lr "$STAGE1_LR" --stage2_rl_lr "$STAGE2_LR")
  else
    CMD=(python rl_tune_genetic.py --base_model "$BASE_MODEL" --data_path "$DATA_PATH" --output_dir "$RUN_ROOT" --batch_size "$BATCH_SIZE" --micro_batch_size "$BATCH_SIZE" --num_epochs 1 --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 --eval_step 80 --adapter_name lora --target_modules "[\"q_proj\", \"k_proj\", \"v_proj\", \"up_proj\", \"down_proj\"]" --use_ist --final_eval_config_source "$FINAL_EVAL_SOURCE" --final_eval_config_path "$FINAL_EVAL_CONFIG" --manual_final_gelu "$MANUAL_GELU" --manual_final_softmax "$MANUAL_SOFTMAX" --final_eval_random_seed "$RANDOM_SEED" --final_eval_permutation_trials "$PERM_TRIALS" --final_eval_cost_equivalent_trials "$COST_TRIALS" --final_eval_budget_equivalent_trials "$BUDGET_TRIALS" --skip_noise_rl "$SKIP_NOISE_SEARCH" --noise_eval_config_source "$NOISE_EVAL_SOURCE" --noise_eval_config_path "$NOISE_EVAL_CONFIG" --manual_noise_config "$MANUAL_NOISE_CONFIG" --noise_eval_repeat_n "$NOISE_EVAL_REPEAT" --skip_stage1_rl "$SKIP_STAGE1_SEARCH" --skip_stage1_final_eval "$SKIP_STAGE1_FINAL_EVAL" --skip_noise_final_eval "$SKIP_NOISE_FINAL_EVAL" --resume_run_dir "$RESUME_FROM")
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

if [ "$SEARCH_ALGORITHM" = "rl" ]; then
  show "Stage-1 回合数" "$STAGE1_EPISODES" "$S_STAGE1_EPISODES"
  show "Stage-2 回合数" "$STAGE2_EPISODES" "$S_STAGE2_EPISODES"
  show "Stage-1 学习率" "$STAGE1_LR" "$S_STAGE1_LR"
  show "Stage-2 学习率" "$STAGE2_LR" "$S_STAGE2_LR"
  show "Stage-1 配置来源" "$(srczh "$FINAL_EVAL_SOURCE")" "$S_FINAL_EVAL_SOURCE"
  show "Stage-2 配置来源" "$(srczh "$NOISE_EVAL_SOURCE")" "$S_NOISE_EVAL_SOURCE"
  show "跳过 Stage-1 搜索" "$(boolzh "$SKIP_STAGE1_SEARCH")" "$S_SKIP_STAGE1_SEARCH"
  show "跳过 Stage-2 搜索" "$(boolzh "$SKIP_NOISE_SEARCH")" "$S_SKIP_NOISE_SEARCH"
elif [ "$SEARCH_ALGORITHM" = "ga" ]; then
  show "Stage-1 迭代代数" "$STAGE1_GENERATIONS" "$S_STAGE1_GENERATIONS"
  show "Stage-2 迭代代数" "$STAGE2_GENERATIONS" "$S_STAGE2_GENERATIONS"
  show "Stage-1 配置来源" "$(srczh "$FINAL_EVAL_SOURCE")" "$S_FINAL_EVAL_SOURCE"
  show "Stage-2 配置来源" "$(srczh "$NOISE_EVAL_SOURCE")" "$S_NOISE_EVAL_SOURCE"
  show "跳过 Stage-1 搜索" "$(boolzh "$SKIP_STAGE1_SEARCH")" "$S_SKIP_STAGE1_SEARCH"
  show "跳过 Stage-2 搜索" "$(boolzh "$SKIP_NOISE_SEARCH")" "$S_SKIP_NOISE_SEARCH"
elif [ "$SEARCH_ALGORITHM" = "rl-and-ga-compare" ]; then
  show "RL Stage-1 回合数" "$STAGE1_EPISODES" "$S_STAGE1_EPISODES"
  show "RL Stage-2 回合数" "$STAGE2_EPISODES" "$S_STAGE2_EPISODES"
  show "GA Stage-1 迭代代数" "$STAGE1_GENERATIONS" "$S_STAGE1_GENERATIONS"
  show "GA Stage-2 迭代代数" "$STAGE2_GENERATIONS" "$S_STAGE2_GENERATIONS"
  show "RL Stage-1 学习率" "$STAGE1_LR" "$S_STAGE1_LR"
  show "RL Stage-2 学习率" "$STAGE2_LR" "$S_STAGE2_LR"
  show "噪声最终评估重复次数" "$NOISE_EVAL_REPEAT" "$S_NOISE_EVAL_REPEAT"
  echo "  RL Stage-1 搜索：$(boolzh "$RL_COMPARE_SKIP_STAGE1_SEARCH")（跳过=是时改走 $(srczh "$RL_COMPARE_FINAL_EVAL_SOURCE")）"
  [ -n "$RL_COMPARE_FINAL_EVAL_CONFIG" ] && echo "  RL Stage-1 JSON：$RL_COMPARE_FINAL_EVAL_CONFIG"
  echo "  RL Stage-2 搜索：$(boolzh "$RL_COMPARE_SKIP_NOISE_SEARCH")（跳过=是时改走 $(srczh "$RL_COMPARE_NOISE_EVAL_SOURCE")）"
  [ -n "$RL_COMPARE_NOISE_EVAL_CONFIG" ] && echo "  RL Stage-2 JSON：$RL_COMPARE_NOISE_EVAL_CONFIG"
  echo "  GA Stage-1 搜索：$(boolzh "$GA_COMPARE_SKIP_STAGE1_SEARCH")（跳过=是时改走 $(srczh "$GA_COMPARE_FINAL_EVAL_SOURCE")）"
  [ -n "$GA_COMPARE_FINAL_EVAL_CONFIG" ] && echo "  GA Stage-1 JSON：$GA_COMPARE_FINAL_EVAL_CONFIG"
  echo "  GA Stage-2 搜索：$(boolzh "$GA_COMPARE_SKIP_NOISE_SEARCH")（跳过=是时改走 $(srczh "$GA_COMPARE_NOISE_EVAL_SOURCE")）"
  [ -n "$GA_COMPARE_NOISE_EVAL_CONFIG" ] && echo "  GA Stage-2 JSON：$GA_COMPARE_NOISE_EVAL_CONFIG"
  echo "  子运行目录：${RUN_ROOT}/children/rl（普通 RL）"
  echo "  子运行目录：${RUN_ROOT}/children/ga（遗传算法）"
  echo "  对比结果目录：${RUN_ROOT}/reports"
  echo "  元信息目录：${RUN_ROOT}/meta"
  echo "  设备分配：${COMPARE_CUDA_NOTE}"
  echo "  说明：该模式支持 RL/GA 分别跳过 Stage-1/Stage-2 搜索，并从各自 JSON 配置直接做最终评估。"
else
  show "通用强化学习模式" "$GENERAL_MODE" "$S_GENERAL_MODE"
  show "跳过 Stage-2" "$(boolzh "$GENERAL_SKIP_STAGE2")" "$S_GENERAL_SKIP_STAGE2"
  if [ "$GENERAL_MODE" = "train" ]; then
    show "任务集合标识" "$GENERAL_TASKSET_ID" "true"
    show "训练任务" "${GENERAL_TASKS:-$DATASET}" "$S_GENERAL_TASKS"
    show "训练轮数" "$GENERAL_ROUNDS" "$S_GENERAL_ROUNDS"
    show "每轮每任务回合数" "$GENERAL_EPISODES_PER_ROUND" "$S_GENERAL_EPISODES_PER_ROUND"
    show "通用策略学习率" "$GENERAL_LR" "$S_GENERAL_LR"
  else
    show "Stage-1 策略文件" "$GENERAL_STAGE1_POLICY" "$S_GENERAL_STAGE1_POLICY"
    show "离线 rollout 次数" "$GENERAL_NUM_ROLLOUTS" "$S_GENERAL_NUM_ROLLOUTS"
    show "是否贪心 rollout" "$(boolzh "$GENERAL_GREEDY")" "$S_GENERAL_GREEDY"
    [ -n "$GENERAL_STAGE2_POLICY" ] && show "Stage-2 策略文件" "$GENERAL_STAGE2_POLICY" "$S_GENERAL_STAGE2_POLICY"
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
  echo "  RL 子进程错误摘要：${RUN_ROOT}/children/rl/logs/error_summary.txt"
  echo "  GA 子进程错误摘要：${RUN_ROOT}/children/ga/logs/error_summary.txt"
  echo "  Compare 元信息：${RUN_ROOT}/meta/compare_metadata.json"
  echo "  Stage-1 对比报告：${RUN_ROOT}/reports/stage1_compare_report_${DATASET}.md"
  echo "  Stage-2 对比报告：${RUN_ROOT}/reports/stage2_compare_report_${DATASET}.md"
else
  echo "  优雅停止（Graceful Stop）：kill -INT $JOB_PID"
fi
