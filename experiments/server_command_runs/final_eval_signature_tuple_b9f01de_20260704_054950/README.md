# final_eval_signature_tuple_b9f01de_20260704_054950

Source commit: `b9f01de`

Optimization: Paean/final-eval cache signatures now reuse
`_full_signature()` and build integer tuple keys directly with `_int_tuple()`.
This removes repeated `np.asarray(...).tolist()` materialization from both the
shared signature helper and the `run()`-local `_noise_eval()` cache key path.

Server workflow:

- Red package: `/hy-tmp/rfr_final_signature_red_0afacfb_20260704_`
- Green package: `/hy-tmp/rfr_final_signature_green_20260704_`
- Server canonical worktree was not modified.

Verification:

- `red_unittest.log`: expected failure on
  `test_full_signature_avoids_tolist_materialization`, proving the old
  signature path still materialized arrays through `.tolist()`.
- `green_validation.log`: `python3 -m py_compile` passed, target unittest passed
  5 tests, and the source guard confirmed `_full_signature()` / `_noise_eval()`
  no longer use `.tolist()` for signature construction.
