# BLB Action Registry Required-Slot Check

- profile: `mrpc`
- num_layers: `12`
- expected_required_slots_per_layer: `59`
- status: `mismatch`
- effective_required_total: `799`

| layer | required_slots | status |
|---:|---:|---|
| 0 | 64 | mismatch |
| 1 | 65 | mismatch |
| 2 | 67 | mismatch |
| 3 | 67 | mismatch |
| 4 | 67 | mismatch |
| 5 | 71 | mismatch |
| 6 | 67 | mismatch |
| 7 | 65 | mismatch |
| 8 | 67 | mismatch |
| 9 | 67 | mismatch |
| 10 | 67 | mismatch |
| 11 | 65 | mismatch |

Safe handling: keep every current action field in the registry; mark non-required fields as compat or ineffective extras instead of deleting them.
