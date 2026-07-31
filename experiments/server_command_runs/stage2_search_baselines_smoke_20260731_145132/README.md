# Stage-2 Search Baselines Server Smoke

Source: `d2919e3d8079460802521f8a39ed12ba85ab7a32` (tree `858ae877343fbc0499794781fca3fcd6fa04b8e2`).

Target: cached BERT-large RTE, 24 layers, Stage-2 all4/Softmax6, 256-example probe, K=3. These are smoke-only runs; no validation_full A/B/C scientific gate was requested or exported.

- `greedy`: 2 real candidates, phases=['initial', 'neighbor_scan'], status=smoke_only_complete, standard scientific best exported=false.
- `bo_rf`: 4 real candidates, phases=['initial_design', 'constrained_ei', 'constrained_ei'], status=smoke_only_complete, standard scientific best exported=false.
- `coinn_ga`: 6 real candidates, phases=['population', 'population'], status=smoke_only_complete, standard scientific best exported=false.

Every candidate passed runtime assertions for model forward, replan installation before forward, 119/119 applied configs, zero apply errors, exact 24-layer actions, three trials, and all six metric/probability channels.
