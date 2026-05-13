# Phase-1B Optimizer Mode Consistency

- profile: `mrpc`
- num_layers: `12`
- rescale_optimizer_mode: `in_process_real`
- rescale_optimizer_hash: `ed28392d4078e4eb7734740023d281d5b87f1abde68340d7776f4e2855e4278e`
- stage1_config_hash: `6454e0556f54ddb4519d9d2998582bca40a41fe2910d2ece679e455f8854eed3`

## Invariants

| invariant | status |
|---|---|
| `all_max_baseline_path_equals_candidate_path` | PASS |
| `inactive_l0b1_mutation_equals_all_max_candidate_path` | PASS |
| `inactive_first_input_mutation_equals_all_max_candidate_path` | PASS |
| `effective_single_mutation_may_differ_from_all_max_candidate_path` | PASS |

## Cases

| case | mode | valid | bits | fusion | requests | raw_hash | effective_hash |
|---|---|---:|---:|---:|---:|---|---|
| `all_max_raw` | `evaluate_baseline_blocks` | true | 14889 | 0 | 59 | `e18db2a9a1b3` | `e18db2a9a1b3` |
| `all_max_via_candidate_path` | `evaluate_blocks` | true | 14889 | 0 | 59 | `e18db2a9a1b3` | `e18db2a9a1b3` |
| `inactive_l0b1_mutation` | `evaluate_blocks` | true | 14889 | 0 | 59 | `e6bd245c733c` | `e18db2a9a1b3` |
| `inactive_first_input_mutation` | `evaluate_blocks` | true | 14889 | 0 | 59 | `f74ae731417f` | `e18db2a9a1b3` |
| `effective_single_mutation` | `evaluate_blocks` | true | 14873 | 0 | 59 | `37fbb939cbb9` | `37fbb939cbb9` |
