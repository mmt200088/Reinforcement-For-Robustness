# Phase-1B Result Summary

## Verdict

Phase-1B local optimizer/code integrity work is complete enough to hand to GPT 5.5 Pro, but server sync and F1 GPU smoke are blocked by SSH reset. Do not start long training from this evidence alone.

## Current Code Identity

- git HEAD: `6341ceab2bb15cd6e4cb0b98805bc88d7343a984`
- branch: `jk_standard_rl`
- tracked diff hash: `5ceffb26b9a14856169b876fc9ffc3334b50de7b788da78e6bf5495b6243b9bd`
- action width: `877`
- registry hash: `6c3662ba26160952e27dca8a8e3ae164af8326ac01819677c7b1a453fe342412`
- max_sfs hash: `bee17f0ccab949b79b4ca011a97da4cebd1d749e6ad49bffa272a701895e09f6`
- Stage-1 config hash: `6454e0556f54ddb4519d9d2998582bca40a41fe2910d2ece679e455f8854eed3`
- Rescale_optimizer mode/root/hash: `in_process_real` / `Rescale_optimizer` / `ed28392d4078e4eb7734740023d281d5b87f1abde68340d7776f4e2855e4278e`
- mask hash: `332b30d017d92e7bd5b27255b005413eb2a49cd147174750ac71e213b99f6d08`

## Optimizer Consistency

- all invariants pass: `True`

- all_max_raw: mode=evaluate_baseline_blocks, request_count=59, send_block1_L0=False, send_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=e18db2a9a1b3..., effective_hash=e18db2a9a1b3...
- all_max_via_candidate_path: mode=evaluate_blocks, request_count=59, send_block1_L0=False, send_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=e18db2a9a1b3..., effective_hash=e18db2a9a1b3...
- inactive_l0b1_mutation: mode=evaluate_blocks, request_count=59, send_block1_L0=False, send_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=e6bd245c733c..., effective_hash=e18db2a9a1b3...
- inactive_first_input_mutation: mode=evaluate_blocks, request_count=59, send_block1_L0=False, send_first_input=False, valid=True, total_bits_sum=14889, fusion_count=0, raw_hash=f74ae731417f..., effective_hash=e18db2a9a1b3...
- effective_single_mutation: mode=evaluate_blocks, request_count=59, send_block1_L0=False, send_first_input=False, valid=True, total_bits_sum=14873, fusion_count=0, raw_hash=37fbb939cbb9..., effective_hash=37fbb939cbb9...

## Registry Consistency

- full slot count: `877`
- effective slot count: `791`
- ineffective/compat-extra count: `86`
- layer-0 block1 or first_input incorrectly effective count: `0`

## F0 Optimizer-Only Result

- baseline optimizer_valid: `True`
- baseline total_bits_sum: `14889`
- baseline fusion_count: `0`
- baseline avg_k: `13.0`
- baseline effective_action_hash: `e18db2a9a1b3d00d1855894bbf812de4267daa56b82cd6a0d7fd384b8e42ef81`

Masked random validity:

- mutation_count=1: valid=50/50 (1.000), total_bits=14873/14887.28/14893, fusion=0/0.02/1
- mutation_count=2: valid=50/50 (1.000), total_bits=14873/14885.60/14895, fusion=0/0.10/1
- mutation_count=4: valid=50/50 (1.000), total_bits=14851/14881.88/14897, fusion=0/0.16/1
- mutation_count=8: valid=50/50 (1.000), total_bits=14843/14874.92/14891, fusion=0/0.42/3

Top multi-random valid candidates:

- mutation_count=32, total_bits_sum=14795, fusion_count=1, effective_hash=41e3ad0fee3e93b2e6ca50ee770ed308297c3dae4eb3cce1fb95cfda6782fe9d
- mutation_count=32, total_bits_sum=14803, fusion_count=0, effective_hash=cc9dd6ef0c6e57f413b675bb10732b145be9b7c978401a6b46c7aaf240c88e1a
- mutation_count=32, total_bits_sum=14803, fusion_count=0, effective_hash=25a87e6d9e982717426610335f833d2771cca7cfa862c0c82dae9eff36071a60
- mutation_count=16, total_bits_sum=14809, fusion_count=1, effective_hash=56a7fd7549ebe600215aa2951dfa930e504d4eb3ea131b8249fb7ca4d9351975
- mutation_count=32, total_bits_sum=14809, fusion_count=1, effective_hash=4995c4994bd113df62115012231b8c8f0d854334329eaf1dbdee4e3bb3147557

## Server / GPU Status

Blocked. SSH command to `root@100.84.74.99:8722` still fails during key exchange with `Connection reset`. No server sync, server tests, or F1 GPU smoke is claimed.
