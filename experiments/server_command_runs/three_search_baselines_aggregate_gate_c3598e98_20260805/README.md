# Three Search Baselines Aggregate Server Gate

Task: `three-search-baselines-20260804`

Source: `c3598e98064558cef1f39cfa15c9468e65dbf9f9`
Tree: `24b3596d0632efe5b73cddb7182c47719ca51520`
Canonical baseline: `6c36532a47349ffc43d38616a030b37dd1b29153`

Result: **mandatory project regression gate failed**.

The task-focused suites, compile/static checks, five-GPU CUDA gate, real
`textattack/bert-base-uncased-MRPC` + GLUE MRPC forward, and the
`two_stage_search_final_v2`/optional-Paean isolation gate passed.

After exact sparse materialization, the complete CPU/no-GPU project suite
reported `2097 passed, 13 failed, 14 skipped, 388 subtests passed`.
A matched targeted canonical run reported 12 failures; the candidate reported
13. The newly introduced failure is:

`tests/test_blb_layerwise_runner.py::LayerwiseDispatchRulesTests::test_launcher_locks_stage2_directory_before_fresh_cleanup`

The `flock` call remains before fresh cleanup, but the task changed the static
guard's persistent-directory anchor. Per protocol, this evidence is diagnostic
only. The task handoff remains in progress and canonical must not advance until
an ordinary task agent fixes the contract locally and all mandatory gates pass.

No formal BO/Greedy/COINN-GA search and no RL rerun was started.
