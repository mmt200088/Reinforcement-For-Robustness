# Stage-2 all4 configuration gate

- Verified source: `2857285523b115e1533ac3a745380ec064d7568e`
- Server: school RTX 4090 host
- Dataset/model: MRPC / BERT base
- Resolved Stage-2 prerequisite: GELU `[4] * 12`, Softmax `[6] * 12`
- Fusion map path: enabled, with every Block 5 action using `block5_n4`

## Tests

- Red snapshot `1248662`: 8 expected failures, 10 passes, 6 subtest passes.
- Core green snapshot `32d314a`: 6 passes, 10 subtest passes.
- Final focused snapshot `2857285`: 92 passes, 10 subtest passes.
- `bash -n` and Python compilation passed on the server.

## Runtime smoke

The successful smoke used 20 episodes with a smoke-only PPO/rollout window of
20. It completed one PPO update and exited naturally.

- `episodes.jsonl`: 20 rows
- PPO updates: 1
- Invalid steps: 0 in the final episode; no error summary was produced
- `block5_n4` occurrences: 240, exactly 12 layers x 20 episodes
- `block5_n1` / `block5_n2` occurrences: 0
- Structured persistent-output verifier: `VERIFY_OK`

The strict `--require-png` verifier reported the two curve PNG files missing.
That requirement is retained in `final_persistent_verify.txt`; the 20-episode
startup smoke was not promoted as a formal RL result. The NPZ curve, status,
live summary, diagnostics, details batch, episodes, and PPO updates were all
present.

## Environment retries

1. The first launcher process exited before Python because the server shell did
   not include the project conda `bin` directory in `PATH`.
2. The second process loaded the model and dataset, then correctly rejected 20
   episodes against the default PPO update interval of 120.
3. The final process explicitly used the conda `PATH` and matched both the PPO
   update interval and Stage-2 rollout size to 20. No source files were edited
   on the server.

## Switch back

To use the searched Stage-1 result in a future Stage-2 run:

```bash
--stage2-fixed-config-source stage1_result
```

Explicit `json` and `manual` sources remain supported as well.
