# All-agent aggregate verification

Tested aggregate source:
`ee94c28f97b2e7b3b9c58cd0c207d9e843c88ffd`

Tested aggregate tree:
`cbb95e40e1dd8222f08abc4bb569e9e93c6f0d2e`

The aggregate combines the latest runtime-efficiency integration with the
completed multi-profile installed-SF audit, GLUE noise-seed replay fix and
artifacts, and small-GTrXL ablation documentation. Source selection and
supersession decisions are recorded in
`docs/source_integration/2026-07-24-all-agents-aggregate.md`.

The newly combined audit and GLUE tests passed 13/13. The focused runtime suite
passed 290 tests and 132 subtests, with one five-GPU integration test skipped
because the server has only four healthy GPUs.

The broad `test_blb_*` run had the exact same ten failing test/subtest entries
as the pre-aggregate runtime base. The aggregate passed 981 tests versus 979 on
the base because it adds two passing GLUE seed tests. `failure_set.diff` is
empty.

An initial focused run used physical indices `0,1,2,4`; PyTorch then failed CUDA
initialization with `device=3, num_gpus=3`. Repeating with the four healthy GPU
UUIDs passed. This is a server ordinal-mapping issue, not a source regression.
GPU 3 still reports 33058 MiB used with unavailable utilization and requires a
server-side reset before a five-GPU gate can run.
