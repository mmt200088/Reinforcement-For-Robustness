set -uo pipefail
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}.:Rescale_optimizer"
export HF_HOME=/hy-tmp/hf_cache HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 GLUE_LOCAL_DATASET_DIR=/hy-tmp/glue_data

TS=$(date +%Y%m%d_%H%M%S)
OUT="experiments/server_command_runs/stage2_boost_apply_${TS}"
mkdir -p "$OUT"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD > "$OUT/HEAD.txt"; cat "$OUT/HEAD.txt"; git log --oneline -4
fi

echo "#################### [B] precision boost: tests + apply to maps ####################"
echo "==================== [B0] unit tests (block2/4/5_n2/5_n4 + SF-direct==index + boosted->installed + boost_options_for_block guard; real replan/torch) ===================="
python3 -m unittest tests.test_blb_precision_boost -v 2>&1 | tee "$OUT/boost_selftest.txt"
grep -qE "^OK" "$OUT/boost_selftest.txt" || { echo "[FATAL] precision-boost unit tests failed - not applying maps"; exit 1; }
# server has torch+RO: all tests should run. skipped>=4 => torch/RO missing => equivalence gate void => do NOT apply.
if grep -qE "skipped=([4-9]|[0-9]{2,})" "$OUT/boost_selftest.txt"; then
  echo "[FATAL] equivalence/guard tests skipped (torch/RO missing) - gate void, not applying maps"; exit 1; fi
echo "[B0] PASS"

echo "==================== [B1] apply boost to committed maps (block2/4/5_n2/5_n4; seconds, == builder final step) ===================="
cp -a blb_stage2_rl/fusion_maps/mrpc "$OUT/old_maps" 2>/dev/null || true
python3 scripts/blb_apply_precision_boost.py --profile mrpc 2>&1 | tee "$OUT/boost_apply.txt"
grep -qE "options boosted" "$OUT/boost_apply.txt" || { echo "[FATAL] boost not applied"; exit 1; }

echo "==================== [B2] post-write re-verify: maps still load + option0==baseline + boosted options carry explicit_field_values ===================="
python3 - <<'PY' 2>&1 | tee "$OUT/boost_verify.txt" || { echo "[FATAL] post-boost map verify failed"; exit 1; }
import json, glob, os
from blb_stage2_rl.fusion_count_map import FusionCountMap
FusionCountMap.load("mrpc")
print("FusionCountMap.load('mrpc') OK - all maps still load, option0==baseline.")
nb = 0
for f in sorted(glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")):
    b = os.path.basename(f)
    if b.startswith("_"):
        continue
    d = json.load(open(f))
    bs = [o for o in d["options"] if o.get("boosted")]
    for o in bs:
        assert o.get("explicit_field_values"), b + " boosted option missing explicit_field_values"
    nb += len(bs)
    print("  %-16s options=%d boosted=%d" % (b, len(d["options"]), len(bs)))
print("[ok] %d boosted options total, all carry explicit_field_values." % nb)
PY

echo "#################### [DONE] ####################"
echo "boosted map changes:"; git status --short blb_stage2_rl/fusion_maps/ | tee "$OUT/boost_map_diff.txt"
echo "evidence dir: $OUT (boost_selftest / boost_apply / boost_verify / boost_map_diff / old_maps)"
echo "Please git add/commit/push the evidence in $OUT and the changed blb_stage2_rl/fusion_maps/mrpc/*.json."
echo "KV-cache already NOT EFFECTIVE -> keep default OFF; next is on-deck ADR-016 reward 60k (no kv-cache flag)."
