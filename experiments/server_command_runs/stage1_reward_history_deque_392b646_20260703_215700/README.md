# Stage-1 Reward History Deque Verification

Purpose: verify the Stage-1 reward-history sliding-window optimization in
`layer_importance_evaluator.py`.

- Red commit: `35a08de`
- Source optimization commit: `61c8c57`
- Green/source head commit: `392b646`
- Server run roots:
  - `/hy-tmp/rfr_stage1_reward_history_deque_35a08de_20260703_215700`
  - `/hy-tmp/rfr_stage1_reward_history_deque_61c8c57_20260703_220100`
  - `/hy-tmp/rfr_stage1_reward_history_deque_392b646_20260703_220300`

Checks:

- Red target unittest:
  `tests.test_stage1_eval_accel.Stage1RewardHistoryWindowSourceTest.test_reward_history_uses_bounded_deque_not_front_pop`
  failed because the old Stage-1 runtime state initialized
  `self.reward_history` as a list and trimmed it with `pop(0)`.
- The first green attempt at `61c8c57` compiled successfully but failed the
  target unittest because the test expected a one-line `deque(...)` expression
  and the resume path used a multi-line call. That attempt is kept as audit
  evidence and is superseded by `392b646`.
- Final green `py_compile` for `layer_importance_evaluator.py` and
  `tests/test_stage1_eval_accel.py` returned `0`.
- Final green target unittest returned `OK`.

Scope:

- This keeps the same Stage-1 reward normalization window semantics while
  replacing list overflow trimming with `deque(maxlen=RUNNING_REWARD_HISTORY_SIZE)`.
- It does not change reward formulas, PPO update logic, validation split
  selection, checkpoint output schema, or Stage-2 RL behavior.
