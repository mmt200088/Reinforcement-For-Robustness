# handler_layer_resolve_cache_ee8a68e_20260704_061121

Source commit: `ee8a68e`

Optimization: `ReversibleLayerHandler` now caches resolved transformer layer
sequences by `layer_name` via `_resolve_layers()`. Stage-1 GELU/Softmax
install/restore and shared legacy input/projection/linear/softmax-value noise
install/restore paths reuse that tuple instead of repeatedly evaluating and
copying `eval("self." + layer_name)`.

Server workflow:

- Red package: `/hy-tmp/rfr_layer_resolve_cache_red_c0f3d78_20260704_`
- Green package: `/hy-tmp/rfr_layer_resolve_cache_green_20260704_`
- Server canonical worktree was not modified.

Verification:

- `red_unittest.log`: expected failure on
  `test_reversible_handler_reuses_resolved_layer_sequences`, proving the cache
  helper was absent before the optimization.
- `green_validation.log`: `python3 -m py_compile` passed and
  `FunctionHandlerForwardAllocationSourceTest` passed 9 tests, including the
  source check for cached layer resolution in the targeted install/restore
  paths.
