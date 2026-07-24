# Stage-1 record: bert large mrpc

- GELU source: BERT-large MRPC Stage-1 PPO entropy-converged best, 2026-06-26
- Selection protocol: `stage1_ppo_entropy_converged_best`
- Source server commit: `4269109bf5570eff5ad614026d4112ce29512892`
- Repository config commit: `237314146003f68c82830abfa886cf2ef7086baf`
- Repository config: `Model_analysis/configs/approx_per_dataset.json#mrpc_large.stage1`
- Independent report SHA-256: `e9fab28cdc4ae68eb5b9031030108a7bcd6faf40c387245f72ee171b01434271`
- Validation split size: `408`
- GELU degrees: `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`
- Stage-2 Softmax contract: degree `6` in all 24 layers
- Current-contract metrics: pending this server evaluation

The user-provided BERT-large validation report independently records this exact
GELU vector with Softmax degree 6 in every layer. It measured validation_full
loss `1.421497045`, accuracy `0.889706`, and weighted F1 `0.885077` with no
Stage-2/noise hooks. The current experiment reruns the configuration through
the current production resolver and uses those values only as provenance.
