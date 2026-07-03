#!/usr/bin/env bash
set +e
cd "/hy-tmp/diag_report_stream_16fdda8_20260703_193735"
mkdir -p logs old_src head_src

git ls-remote https://github.com/mmt200088/Reinforcement-For-Robustness.git refs/heads/jk_standard_rl > logs/git_ls_remote.log 2>&1
git_rc=$?
printf '%s\n' "$git_rc" > logs/git_ls_remote.rc

tar -xf diag_report_stream_16fdda8_20260703_193735_old.tar -C old_src
tar -xf diag_report_stream_16fdda8_20260703_193735_head.tar -C head_src

(cd old_src && python3 ../diag_report_stream_16fdda8_20260703_193735_red.py . > ../logs/red_old.log 2>&1)
red_rc=$?
printf '%s\n' "$red_rc" > logs/red_old.rc

(cd head_src && python3 -m unittest tests.test_rl_data_points -v > ../logs/green_head_unittest.log 2>&1)
green_rc=$?
printf '%s\n' "$green_rc" > logs/green_head_unittest.rc

printf '%s\n' '16fdda80b628e0145cdb4362cd8c88beda3089d2' > logs/head_commit.txt
printf '%s\n' 'ba2c3bfea2bbc2fa3acbaf5bb43577986f566691' > logs/red_old_commit.txt
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
