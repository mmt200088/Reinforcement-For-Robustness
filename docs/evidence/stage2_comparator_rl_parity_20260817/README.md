# Stage-2 comparator/RL parity server evidence

This directory contains the compact acceptance evidence for
`stage2-comparator-rl-parity-20260816`. The server used one NVIDIA RTX 4090 and
ran real `textattack/bert-base-uncased-MRPC` inference on GLUE MRPC.

## Files

- `compact_server_evidence_3d2439d0.json`: compact server, three-backend,
  same-action, strict, test, Paean, and checkpoint receipt.
- `same_action_online_be690e8c.json.gz`: complete online PPO/comparator
  same-action payload.
- `same_action_strict_feasible_230276eb.json.gz`: complete top-5 strict A/B/C
  payload. The uncompressed JSON SHA-256 is
  `5b60b322d2067f2cc08875f4ef061fa8d7ca440d0cd7629dd021e05bd94b389a`.
- `frozen_checkpoint_read_3d2439d0.json`: immutable 60k PPO checkpoint read and
  hash receipt.
- `optional_paean_failure_3d2439d0.json`: failed-optional Paean publication
  receipt.

## SHA-256

```text
2dd22a604311418abb12ff5194326751e5865f3e0cd9dd8829e8005d486bc7d7  compact_server_evidence_3d2439d0.json
fc9efc317a610d93a8e28f900e8a76c808f8aeeb24b72572038f68bb1a65521a  frozen_checkpoint_read_3d2439d0.json
b0b47cd37be55bb875ace261ec2613c7ac7eecbeb8e9f83eb0e3c6b8f7c3f464  optional_paean_failure_3d2439d0.json
f9d62c13e82f9756039cfb9645dde951746c6773983f865d1bc0dff42ea439ad  same_action_online_be690e8c.json.gz
556893db4fdf7300667a341bcc24cde14f0c7479adf112959e8b28861ed1053f  same_action_strict_feasible_230276eb.json.gz
```

The abandoned PPO diagnostic and the intentionally rejected all-zero strict
target are recorded in the compact receipt and are not acceptance evidence.
