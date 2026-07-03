#!/usr/bin/env bash
set +e
cd "/hy-tmp/stage1_gpu_eval_scripts_4324577_20260703_200810"
mkdir -p logs old_src head_src

timeout 20 git ls-remote https://github.com/mmt200088/Reinforcement-For-Robustness.git refs/heads/jk_standard_rl > logs/git_ls_remote.log 2>&1
git_rc=$?
printf '%s\n' "$git_rc" > logs/git_ls_remote.rc

tar -xf stage1_gpu_eval_scripts_4324577_20260703_200810_head.tar -C old_src
tar -xf stage1_gpu_eval_scripts_4324577_20260703_200810_old.tar -C old_src
tar -xf stage1_gpu_eval_scripts_4324577_20260703_200810_head.tar -C head_src

(cd old_src && python3 -m unittest tests.test_stage1_eval_accel.Stage1GpuEvalScriptSourceTest -v > ../logs/red_old_unittest.log 2>&1)
red_raw_rc=$?
printf '%s\n' "$red_raw_rc" > logs/red_old_unittest_raw.rc
if [ "$red_raw_rc" -ne 0 ]; then
  red_rc=0
else
  red_rc=1
fi
printf '%s\n' "$red_rc" > logs/red_old.rc

(cd head_src && python3 -m unittest tests.test_stage1_eval_accel.Stage1GpuEvalScriptSourceTest -v > ../logs/green_head_unittest.log 2>&1)
green_rc=$?
printf '%s\n' "$green_rc" > logs/green_head_unittest.rc

printf '%s\n' '43245772e6b91012e1d01fced6aebfa35d4dbc20' > logs/head_commit.txt
printf '%s\n' '9d013d2a8dd471c165eadc01a7fc3adf8e007b34' > logs/red_old_commit.txt
python3 - <<'PY' > logs/summary.json
import json
from pathlib import Path
logs = Path('logs')
summary = {
    'head_commit': logs.joinpath('head_commit.txt').read_text().strip(),
    'red_old_commit': logs.joinpath('red_old_commit.txt').read_text().strip(),
    'git_ls_remote_rc': int(logs.joinpath('git_ls_remote.rc').read_text().strip()),
    'red_old_unittest_raw_rc': int(logs.joinpath('red_old_unittest_raw.rc').read_text().strip()),
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
