# Stage-2 21-Group K7 Extension

**Goal:** Add an all-K=7 profile under every fusion profile and rerun the full
BERT-base MRPC validation experiment.

**Scope:** Keep the existing production action-to-model chain, paired seeds,
five trials per group, and installed-configuration audits unchanged. Extend the
grid from 3x6 to 3x7 only.

- [x] Add a failing 21-group grid contract.
- [x] Add `K=(7,7,7,7,7)` to the shared experiment profile list.
- [ ] Run local compilation and focused tests.
- [ ] Sync the verified commit to Git and the server.
- [ ] Run five experiment seeds and aggregate 25 trials per group.
- [ ] Audit 21 groups, 60 installed K slots per group, and exact fusion totals.
- [ ] Deliver the HTML and raw JSON locally and archive compact evidence in Git.
