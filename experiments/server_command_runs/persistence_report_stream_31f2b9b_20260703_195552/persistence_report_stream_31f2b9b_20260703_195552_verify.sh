#!/usr/bin/env bash
set +e
cd "/hy-tmp/persistence_report_stream_31f2b9b_20260703_195552"
mkdir -p logs old_src head_src

timeout 20 git ls-remote https://github.com/mmt200088/Reinforcement-For-Robustness.git refs/heads/jk_standard_rl > logs/git_ls_remote.log 2>&1
git_rc=$?
printf '%s\n' "$git_rc" > logs/git_ls_remote.rc

tar -xf persistence_report_stream_31f2b9b_20260703_195552_old.tar -C old_src
tar -xf persistence_report_stream_31f2b9b_20260703_195552_head.tar -C head_src

(cd old_src && python3 ../persistence_report_stream_31f2b9b_20260703_195552_red.py . > ../logs/red_old.log 2>&1)
red_rc=$?
printf '%s\n' "$red_rc" > logs/red_old.rc

(cd head_src && python3 -m unittest tests.test_blb_stage2_rl_regressions.BLBTraceWriterRegressionTests.test_persistence_report_writers_stream_line_outputs -v > ../logs/green_head_unittest.log 2>&1)
green_rc=$?
printf '%s\n' "$green_rc" > logs/green_head_unittest.rc

printf '%s\n' '31f2b9bf9b85ca89fa214c60e6d53e38b4e782ee' > logs/head_commit.txt
printf '%s\n' 'ce030a264eacda70d9579a6db030565dfcde24f0' > logs/red_old_commit.txt
python3 - <<'PY' > logs/summary.json
import json
from pathlib import Path
logs = Path('logs')
summary = {
    'head_commit': logs.joinpath('head_commit.txt').read_text().strip(),
    'red_old_commit': logs.joinpath('red_old_commit.txt').read_text().strip(),
    'git_ls_remote_rc': int(logs.joinpath('git_ls_remote.rc').read_text().strip()),
    'red_old_rc': int(logs.joinpath('red_old.rc').read_text().strip()),
    'green_head_unittest_rc': int(logs.joinpath('green_head_unittest.rc').read_text().strip()),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
cat logs/summary.json
if [ "$red_rc" -eq 0 ] && [ "$green_rc" -eq 0 ]; then
  exit 0
fi
exit 1
