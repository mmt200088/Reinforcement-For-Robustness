set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_boost_apply_${TS}"
mkdir -p "$OUT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$OUT/HEAD.txt"; cat "$OUT/HEAD.txt"; git log --oneline -4
fi

echo "#################### [B] å å¤§ç²¾åº¦ éªè¯ä¸è½ map ####################"
echo "==================== [B0] åæµï¼block2/4/5_n2/5_n4 + SF-direct==index + boostedâinstalled cfgï¼çå® replan/torchï¼===================="
python3 -m unittest tests.test_blb_precision_boost -v 2>&1 | tee "$OUT/boost_selftest.txt"
grep -qE "^OK" "$OUT/boost_selftest.txt" || { echo "[FATAL] å å¤§ç²¾åº¦åæµå¤±è´¥ â ä¸è½ map"; exit 1; }
# æå¡å¨æ torch+ROï¼28 é¡¹åºå
¨è·ãè¥ skipped>=4 è¯´æ torch/RO ç¼ºå¤± â SF-direct==index ç­ä»·é¨ç¦æ æï¼ç¦æ­¢è½ mapã
if grep -qE "skipped=([4-9]|[0-9]{2,})" "$OUT/boost_selftest.txt"; then
  echo "[FATAL] SF-direct==index / boostedâinstalled ç­ä»·æµè¯è¢« skipï¼torch/RO ç¼ºå¤±ï¼â é¨ç¦æ æï¼ç¦æ­¢è½ map"; exit 1; fi
echo "[B0] PASSï¼å« SF-direct==index + boostedâinstalled cfg ç­ä»·é¨ç¦ï¼"

echo "==================== [B1] å¯¹å·²æäº¤ map åºç¨ boostï¼block2/4/5_n2/5_n4ï¼ç§çº§ï¼ç­ä»· builder æ«æ­¥ï¼===================="
cp -a blb_stage2_rl/fusion_maps/mrpc "$OUT/old_maps" 2>/dev/null || true
python3 scripts/blb_apply_precision_boost.py --profile mrpc 2>&1 | tee "$OUT/boost_apply.txt"
grep -qE "options boosted" "$OUT/boost_apply.txt" || { echo "[FATAL] boost æªåºç¨"; exit 1; }

echo "==================== [B2] è½çåå¤éªï¼map ä»å¯ load + option0==baseline + boosted éé¡¹å¸¦ explicit_field_values ===================="
python3 - <<'PY' 2>&1 | tee "$OUT/boost_verify.txt" || { echo "[FATAL] boost å map å¤éªå¤±è´¥"; exit 1; }
import json, glob, os
from blb_stage2_rl.fusion_count_map import FusionCountMap
FusionCountMap.load("mrpc")
print("FusionCountMap.load('mrpc') OK â ææå¾ä»å¯å è½½ï¼option0==baselineã")
nb = 0
for f in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    b = os.path.basename(f)
    if b.startswith("_"):
        continue
    d = json.load(open(f))
    bs = [o for o in d["options"] if o.get("boosted")]
    for o in bs:
        assert o.get("explicit_field_values"), f"{b} boosted éé¡¹ç¼º explicit_field_values"
    nb += len(bs)
    print(f"  {b:16s} options={len(d['options'])} boosted={len(bs)}")
print(f"[ok] å
± {nb} ä¸ª boosted éé¡¹ï¼åå¸¦ explicit_field_valuesã")
PY

echo "#################### [DONE] ####################"
echo "boosted åç map åæ´ï¼"; git status --short blb_stage2_rl/fusion_maps/ | tee "$OUT/boost_map_diff.txt"
echo "è¯æ®ç®å½ï¼$OUTï¼boost_selftest / boost_apply / boost_verify / boost_map_diff / old_mapsï¼"
echo "è¯·æ $OUT ä¸è¯æ® + æ¹å¨åç blb_stage2_rl/fusion_maps/mrpc/*.json ä¸å¹¶ git add/commit/push åä¼ ã"
echo "KV-cache å·²å¤ NOT EFFECTIVE â ä¿æé»è®¤ OFFï¼ä¸ä¸æ­¥èµ° on-deck ç ADR-016 reward 60kï¼ä¸å¸¦ kv-cache flagï¼ã"
