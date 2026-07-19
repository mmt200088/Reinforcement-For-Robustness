#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /empty/recovery/directory" >&2
  exit 2
fi

bundle_dir="$(cd "$(dirname "$0")" && pwd -P)"
dest="$1"
mkdir -p "$dest/repo" "$dest/hy-tmp"

if find "$dest/repo" "$dest/hy-tmp" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to restore into non-empty recovery trees: $dest" >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1" | awk '{print $1}'; }
  (cd "$bundle_dir" && sha256sum -c SHA256SUMS)
else
  hash_file() { shasum -a 256 "$1" | awk '{print $1}'; }
  (cd "$bundle_dir" && shasum -a 256 -c SHA256SUMS)
fi

archives="$bundle_dir/archives"
repo="$dest/repo"
hy_tmp="$dest/hy-tmp"
run_rel='Parting Chapter/persistent/rl/bert-base/mrpc/s1t0.001_s2t0.001_s2st2.0__dual_resource_natconv_fresh_20260718_190924'
raw_rel='rl_training_data_points/stage2/bert-base/mrpc/Parting_Chapter_persistent_rl_bert-base_mrpc_s1t0.001_s2t0.001_s2st2.0__dual_resource_natconv_fresh_20260718_190924__20260718T111042090914Z__pid1943348'

tar -xzf "$archives/persistent_run_without_large_jsonl.tar.gz" -C "$repo"
tar -xzf "$archives/structured_writer_metadata.tar.gz" -C "$repo"
tar -xzf "$archives/server_experiment_results.tar.gz" -C "$repo"
tar -xzf "$archives/report_and_graceful_stop_snapshots.tar.gz" -C "$hy_tmp"
tar -xzf "$archives/final_glue_submission_results.tar.gz" -C "$hy_tmp"

cat "$archives"/candidate_store.jsonl.gz.part* | gzip -dc > "$repo/$run_rel/stage2_noise/progress/candidate_store.jsonl"
cat "$archives"/diagnostics_episodes.jsonl.gz.part* | gzip -dc > "$repo/$run_rel/stage2_noise/progress/diagnostics/episodes.jsonl"
gzip -dc "$archives/ppo_updates.jsonl.gz" > "$repo/$run_rel/stage2_noise/progress/diagnostics/ppo_updates.jsonl"
cat "$archives"/structured_episodes.jsonl.gz.part* | gzip -dc > "$repo/$raw_rel/episodes.jsonl"
gzip -dc "$archives/structured_ppo_updates.jsonl.gz" > "$repo/$raw_rel/ppo_updates.jsonl"

verify_original() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(hash_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "restored hash mismatch: $path" >&2
    echo "expected $expected" >&2
    echo "actual   $actual" >&2
    exit 1
  fi
}

verify_original 519abe44720141bec974e65d3ddc4d9b584fd194f888257a55a55e6a7c1b9407 "$repo/$run_rel/stage2_noise/progress/candidate_store.jsonl"
verify_original aaf09b29e0a4ebf6ade73853d121e5b378999e7d3a83bcd4b3e5f9d85d3989f6 "$repo/$run_rel/stage2_noise/progress/diagnostics/episodes.jsonl"
verify_original 9b829ef413f87c3acaebf791cdece11d5093bae30324264e3111c02be765a36c "$repo/$run_rel/stage2_noise/progress/diagnostics/ppo_updates.jsonl"
verify_original 0d2e4b13f550a964c6c68b8243a0f4f949b729da46b2d3bf185c809c663e4335 "$repo/$raw_rel/episodes.jsonl"
verify_original 265bc21d55164c1a9052e21b98a08e25faed8d8aeee53b5f52c22034043b6e9a "$repo/$raw_rel/ppo_updates.jsonl"
verify_original c039f3de3619261880aa3eb771d80318b1b17984cae926c4f6e819a3a03b1ab4 "$repo/$run_rel/stage2_noise/progress/blb_stage2_rl_checkpoint_live.pt"

echo "restore and SHA-256 verification complete: $dest"
