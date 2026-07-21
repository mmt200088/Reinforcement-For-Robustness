# Stage-2 v10 Policy-Network Ablation

## Frozen Baseline

The pre-ablation Stage-2 v10 shared actor-critic is preserved by the annotated
Git tag:

```text
stage2-rl-v10-shared-baseline-20260721
```

The complete selectable large-network ablation implementation immediately
before the small-network change is also frozen at:

```text
stage2-rl-v10-large-network-ablation-20260721
```

`shared_gtrxl_v1` deliberately keeps the historical `rl_variant` and algorithm
contract byte-for-byte unchanged. A checkpoint created before this ablation can
therefore resume only through this arm.

## Selectable Arms

| CLI value | Actor | Critic | Shared actor/value trunk |
|---|---|---|---|
| `shared_gtrxl_small_v1` | GTrXL 128d, 2 layers, 4 heads, FFN 256 | value head | yes |
| `shared_gtrxl_v1` | original GTrXL | original value head | yes |
| `separate_critic_gtrxl_v1` | original GTrXL | independent isomorphic GTrXL | no |
| `separate_critic_mlp_v1` | original GTrXL | independent 512-512-256 MLP | no |

Select an arm without changing source:

```bash
bash llama_7B_LayerImportance.sh run rl \
  --preset mrpc-blb-stage2-rl \
  --mode stage2-only \
  --blb-v3-policy-network-variant separate_critic_gtrxl_v1 \
  --run-tag network_ablation_separate_gtrxl_seed20260721 \
  --blb-v3-seed 20260721 \
  --fresh
```

Fresh Stage-2 launches default to `shared_gtrxl_small_v1`. Always pass
`--blb-v3-policy-network-variant shared_gtrxl_v1` when intentionally starting
or resuming the historical large shared network.

Every arm and seed must have a distinct `--run-tag`. The launcher persists the
network variant in `metadata.json`; the runner also writes it into the run
manifest, checkpoint, diagnostics manifest, and final summary. Cross-arm resume
is a fatal error.

## Controlled Comparison

Keep reward, action space, data split, constraints, trial counts, PPO settings,
initial actor seed, and evaluation seed banks fixed. Change only
`--blb-v3-policy-network-variant`.

1. Run a short plumbing smoke for all four arms. It may establish only that
   construction, PPO update, diagnostics, checkpoint, and resume work.
2. Run the screening budget for all arms with the same seed set. For the small
   arm, expect useful convergence evidence by 100k-150k episodes and cap the
   first controlled assessment at 200k episodes.
3. Compare robust-feasible rate, best authoritative resource objective,
   time/episodes to first robust-feasible candidate, critic explained variance,
   value RMSE, shared actor/value gradient cosine, per-slot entropy/KL/clip, and
   throughput.
4. Promote promising arms to the same long-run seed protocol. Do not select an
   arm from a single short run.

The reward, action mapping, constraints, candidate promotion, and final
selection logic are intentionally unchanged by this ablation.

## Final Selection And Rollback

After the evidence is reviewed, select the production arm by keeping its exact
CLI value in the formal preset/launch command. The other arms remain available
for audit but are no longer launched.

To return to the original algorithm implementation exactly:

```bash
git switch --detach stage2-rl-v10-shared-baseline-20260721
```

To return to the complete pre-small-network ablation source exactly:

```bash
git switch --detach stage2-rl-v10-large-network-ablation-20260721
```

To use the original arm while retaining the ablation-capable source, select
`--blb-v3-policy-network-variant shared_gtrxl_v1`.
