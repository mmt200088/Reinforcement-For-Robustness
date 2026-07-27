# Stage-2 27-Group Odd/Even K Extension

**Goal:** Add two one-based odd/even layer K schedules under every fusion
profile and rerun the full BERT-base MRPC validation experiment.

**Schedules:**

- Odd human layers high precision; even human layers low precision.
- Odd human layers medium precision; even human layers low precision.

Layer index 0 is human layer 1 and therefore uses the odd-layer profile.

- [x] Add failing contracts for 27 groups and one-based odd/even schedules.
- [x] Extend the experiment-only action builder to accept per-layer K.
- [x] Preserve the uniform K helper and production action-to-model chain.
- [x] Audit all 60 post-materialization K slots against the per-layer schedule.
- [x] Run local compilation and focused tests.
- [x] Sync the verified commit to Git and the server.
- [x] Run five experiment seeds and aggregate 25 trials per group.
- [x] Deliver HTML/raw JSON and archive compact evidence in Git.
