# BLB Action Registry Current-Code Slot Check

- profile: `mrpc`
- num_layers: `12`
- expected_slots_per_layer: `73`
- status: `mismatch`
- effective_required_total: `875`

| layer | required_slots | status |
|---:|---:|---|
| 0 | 72 | mismatch |
| 1 | 73 | ok |
| 2 | 73 | ok |
| 3 | 73 | ok |
| 4 | 73 | ok |
| 5 | 73 | ok |
| 6 | 73 | ok |
| 7 | 73 | ok |
| 8 | 73 | ok |
| 9 | 73 | ok |
| 10 | 73 | ok |
| 11 | 73 | ok |

Safe handling: keep every current action field in the registry; mark non-required fields as compat or ineffective extras instead of deleting them.
