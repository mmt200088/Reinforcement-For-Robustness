"""Stage-1 eval acceleration (2026-06-13) — correctness locks.

Three accelerations are covered:

1. ``PolynomialGELU._poly`` Horner rewrite — must match the untouched
   module-level stacked-powers reference ``function_handler.polynomial`` to
   fp32 rounding (the polynomial is mathematically identical; only the
   evaluation order changes).
2. ``approximation_exponential`` repeated-squaring rewrite (BERT + GPT-2
   helpers) — must match the old ``torch.pow(1 + x/2^d, 2^d)`` form, and
   ``approximation_softmax`` must stay invariant to additive -10000 padding
   columns (this is what makes dynamic padding / eval batch size safe).
3. ``Stage1EvalCache`` — exact-value store; and ``_run_evaluation``'s
   deferred-sync loop must return bit-identical values to the old
   per-batch-sync loop (locked here against an inline reference
   implementation on a fake model).

The poly/exp/eval tests need torch (+ transformers for the eval test) and run
in the server contract gate; they skip on torch-less local boxes. The cache
test is torch-free via direct file import (``stage1_rl/__init__`` pulls
torch).
"""
import importlib.util
import pathlib
import sys
import threading
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _load_eval_cache_module():
    """Import stage1_rl/eval_cache.py WITHOUT triggering the package __init__
    (which imports parallel_runner -> torch)."""
    path = _REPO_ROOT / "stage1_rl" / "eval_cache.py"
    spec = importlib.util.spec_from_file_location("_stage1_eval_cache_solo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _source_region(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start + len(start_marker))
    return source[start:end]


def _method_region(source: str, method_name: str) -> str:
    start = source.index(f"    def {method_name}")
    next_method = source.find("\n    def ", start + 1)
    if next_method == -1:
        next_method = len(source)
    return source[start:next_method]


class FunctionHandlerForwardAllocationSourceTest(unittest.TestCase):
    def test_ones_mask_encode_samples_noise_without_full_shape_ones_prefill(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        regions = {
            "block2_qkt": _source_region(
                source,
                "def _make_block2_qkt_merge_hook(",
                "    return hook\n\n\n# ============================================================================\n# BLB Block 3",
            ),
            "block2_bsgs": _source_region(
                source,
                "def _make_block2_bsgs_mask_hook(",
                "# ============================================================================\n# BLB Block 4",
            ),
            "block4_input": _source_region(
                source,
                "def _make_block4_input_mask_hook(",
                "def _make_block4_softmax_v_hook",
            ),
            "block4_softmax_v": _source_region(
                source,
                "def _make_block4_softmax_v_hook(",
                "def _make_block4_wo_forward",
            ),
        }

        self.assertIn(
            "noisy_mask = _sample_gaussian_for_point(qkt_result, merge_mask_encode)",
            regions["block2_qkt"],
        )
        self.assertIn("noisy_mask.add_(1.0)", regions["block2_qkt"])
        self.assertIn(
            "noisy_mask1 = _sample_gaussian_for_point(tensor, mask1_encode)",
            regions["block2_bsgs"],
        )
        self.assertIn("noisy_mask1.add_(1.0)", regions["block2_bsgs"])
        self.assertIn(
            "noisy_mask2 = _sample_gaussian_for_point(out, mask2_encode)",
            regions["block2_bsgs"],
        )
        self.assertIn("noisy_mask2.add_(1.0)", regions["block2_bsgs"])
        self.assertIn(
            "noisy_mask = _sample_gaussian_for_point(out, mask_encode_point)",
            regions["block4_input"],
        )
        self.assertIn("noisy_mask.add_(1.0)", regions["block4_input"])
        self.assertIn(
            "noisy_mask = _sample_gaussian_for_point(tensor, mask_encode)",
            regions["block4_softmax_v"],
        )
        self.assertIn("noisy_mask.add_(1.0)", regions["block4_softmax_v"])
        for region in regions.values():
            self.assertNotIn("torch.ones_like", region)

    def test_gelu_piecewise_select_uses_scalar_zero_branch(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        helper_region = _source_region(
            source,
            "def _select_piecewise_gelu_output(x: Tensor, y_neg: Tensor, y_pos: Tensor) -> Tensor:",
            "def _make_block5_gelu_forward",
        )

        self.assertIn("out = torch.where(x < 0, y_neg, y_pos)", helper_region)
        self.assertIn("out = torch.where(x >= -2.7, out, 0.0)", helper_region)
        self.assertIn("return torch.where(x > 2.7, x, out)", helper_region)
        self.assertEqual(helper_region.count("torch.where("), 3)
        self.assertNotIn(" & ", helper_region)
        self.assertNotIn("torch.zeros_like", helper_region)

    def test_large_cuda_gelu_pairs_piece_evaluation_behind_size_gate(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        gelu_region = _source_region(
            source,
            "class PolynomialGELU",
            "# change BertsdpaAttention",
        )

        self.assertIn("_GELU_PAIRED_POLY_MIN_NUMEL = 12_000_000", source)
        self.assertIn("def _paired_coeff_tensor(", gelu_region)
        self.assertIn("def _poly_pair(", gelu_region)
        self.assertIn("x.is_cuda", gelu_region)
        self.assertIn("x.dtype == torch.float32", gelu_region)
        self.assertIn("self.degree in (2, 4)", gelu_region)
        self.assertIn("x.numel() >= _GELU_PAIRED_POLY_MIN_NUMEL", gelu_region)
        self.assertIn("y1, y2 = self._poly_pair(x)", gelu_region)

    def test_softmax_lower_bound_zero_branch_uses_scalar_zero(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        bert_region = _source_region(
            source,
            "    def approximation_softmax(self, x: torch.Tensor) -> torch.Tensor:",
            "    # error construction",
        )
        gpt2_region = _source_region(
            source,
            "def _approx_softmax(x: torch.Tensor, degree: int, lower_bound: float) -> torch.Tensor:",
            "def _make_gpt2_approx_attn_forward",
        )

        self.assertIn("torch.where(x < self.lower_bound, 0.0, exp_approx)", bert_region)
        self.assertIn("torch.where(x < lower_bound, 0.0, exp_approx)", gpt2_region)
        self.assertNotIn("torch.zeros_like", bert_region)
        self.assertNotIn("torch.zeros_like", gpt2_region)

    def test_scalar_encode_constants_sample_noise_without_full_shape_prefill(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        block1_region = _source_region(
            source,
            "class NoisyBlock1LayerNorm",
            "# ============================================================================\n# BLB Block 3",
        )
        block3_region = _source_region(
            source,
            "def _make_block3_approximation_exponential(cfg: Block3NoiseConfig):",
            "    return block3_approx_exp",
        )
        block4_region = _source_region(
            source,
            "class NoisyBlock4LayerNorm",
            "# ============================================================================\n# BLB Block 5",
        )

        self.assertIn("noisy_inv_d = _sample_gaussian_for_point(x, cfg.mean_inv_d_encode)", block1_region)
        self.assertIn("noisy_inv_d.add_(1.0 / D)", block1_region)
        self.assertIn("noisy_inv_d_var = _sample_gaussian_for_point(sq, cfg.var_inv_d_encode)", block1_region)
        self.assertIn("noisy_inv_d_var.add_(1.0 / D)", block1_region)
        self.assertNotIn("torch.full_like(x, 1.0 / D)", block1_region)
        self.assertNotIn("torch.full_like(sq, 1.0 / D)", block1_region)

        self.assertIn("noisy_inv_2n = _sample_gaussian_for_point(x, cfg.inv_2n_encode)", block3_region)
        self.assertIn("noisy_inv_2n.add_(inv_2n_value)", block3_region)
        self.assertNotIn("torch.full_like(x, inv_2n_value)", block3_region)

        self.assertIn("noisy_inv_d = _sample_gaussian_for_point(x, cfg4.ln_mean_inv_d_encode)", block4_region)
        self.assertIn("noisy_inv_d.add_(1.0 / D)", block4_region)
        self.assertIn("noisy_inv_d_var = _sample_gaussian_for_point(sq, cfg4.ln_var_inv_d_encode)", block4_region)
        self.assertIn("noisy_inv_d_var.add_(1.0 / D)", block4_region)
        self.assertNotIn("torch.full_like(x, 1.0 / D)", block4_region)
        self.assertNotIn("torch.full_like(sq, 1.0 / D)", block4_region)

    def test_block5_gelu_coeff_encode_reuses_input_as_noise_reference(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        block5_region = _source_region(
            source,
            "def _make_block5_gelu_forward(original_gelu, cfg5: Block5NoiseConfig):",
            "# tensor polynomial approximation",
        )

        self.assertIn("def _compute_polynomial(", block5_region)
        self.assertIn("_sample_gaussian_for_point(x_ref, cfg5.gelu_coeff_encode)", block5_region)
        self.assertIn("noisy_coeff.add_(coeff_value)", block5_region)
        self.assertNotIn("coeff_broadcast", block5_region)
        self.assertNotIn("torch.full_like(x_ref, coeff_value)", block5_region)

    def test_block5_gelu_power_builder_skips_unused_x0_tensor(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        block5_region = _source_region(
            source,
            "def _make_block5_gelu_forward(original_gelu, cfg5: Block5NoiseConfig):",
            "# tensor polynomial approximation",
        )

        self.assertIn("def _compute_powers(x: Tensor):", block5_region)
        self.assertNotIn("powers[0] = torch.ones_like", block5_region)
        self.assertNotIn("powers[0] =", block5_region)

    def test_gelu_piecewise_select_avoids_low_mask_zero_fill(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        regions = [
            _source_region(
                source,
                "    def block5_gelu_forward(x: Tensor) -> Tensor:",
                "    return block5_gelu_forward",
            ),
            _source_region(
                source,
                "class PolynomialGELU",
                "# change BertsdpaAttention",
            ),
        ]

        self.assertTrue(
            "def _select_piecewise_gelu_output(" in source,
            "function_handler must share the GELU piecewise select helper",
        )
        for region in regions:
            self.assertIn("_select_piecewise_gelu_output(", region)
            self.assertNotIn("mask_low", region)
            self.assertNotIn("y0 = torch.zeros_like", region)
            self.assertNotIn("torch.where(mask_low", region)
            self.assertNotIn("torch.zeros_like(x))", region)

    def test_attention_forward_consumes_positional_tail_without_front_pop(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        region = _source_region(
            source,
            "    def forward(\n"
            "        self,\n"
            "        hidden_states,",
            "        if past_key_value is None and past_key_values is not None:",
        )

        self.assertIn("tail_pos = 0", region)
        self.assertNotIn("pop(0)", region)

    def test_reversible_handler_reuses_resolved_layer_sequences(self):
        source = (_REPO_ROOT / "function_handler.py").read_text(encoding="utf-8")
        handler_region = source.split("class ReversibleLayerHandler:", 1)[1]
        init_region = _method_region(handler_region, "__init__")
        if "    def _resolve_layers(self, layer_name):" not in handler_region:
            self.fail("ReversibleLayerHandler is missing _resolve_layers cache helper")
        helper_region = _method_region(handler_region, "_resolve_layers")

        self.assertIn("self._resolved_layers_cache = {}", init_region)
        self.assertIn("def _resolve_layers(self, layer_name):", helper_region)
        self.assertIn("self._resolved_layers_cache.get(layer_name)", helper_region)
        self.assertIn("tuple(eval(\"self.\" + layer_name))", helper_region)

        for method_name in (
            "replace_layer_gelu",
            "replace_layer_softmax",
            "replace_layer_input_noise",
            "replace_layer_softmax_value_noise",
            "_replace_attention_projection_noise",
            "_replace_layer_linear_module_noise",
            "restore_layer_gelu",
            "restore_layer_softmax",
            "restore_layer_input_noise",
            "restore_layer_softmax_value_noise",
        ):
            region = _method_region(handler_region, method_name)
            self.assertIn("self._resolve_layers(layer_name)", region, method_name)
            self.assertNotIn('eval("self." + layer_name)', region, method_name)
            self.assertNotIn('list(eval("self." + layer_name))', region, method_name)


class Stage1EvalCacheTest(unittest.TestCase):
    def test_make_key_normalizes_sequences(self):
        mod = _load_eval_cache_module()
        c = mod.Stage1EvalCache()
        k1 = c.make_key([1, 2, 4], (6, 6, 6), "validation_full")
        k2 = c.make_key((1, 2, 4), [6, 6, 6], "validation_full")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, c.make_key([1, 2, 4], (6, 6, 6), "train"))

    def test_hit_returns_exact_stored_value_and_counts(self):
        mod = _load_eval_cache_module()
        c = mod.Stage1EvalCache()
        key = c.make_key([1], [6], "validation_full")
        self.assertIsNone(c.get(key))
        value = (0.123456789, 0.8672, 0.8651, 412.5)
        c.put(key, value)
        got = c.get(key)
        self.assertIs(got, value)          # the exact object, not a re-derivation
        self.assertEqual(c.hits, 1)
        self.assertEqual(c.misses, 1)
        self.assertEqual(len(c), 1)
        self.assertIn("hit_rate=50.0%", c.stats_line())

    def test_concurrent_get_put_smoke(self):
        mod = _load_eval_cache_module()
        c = mod.Stage1EvalCache()

        def hammer(seed):
            for i in range(200):
                key = c.make_key([seed % 3, i % 5], [6], "validation_full")
                if c.get(key) is None:
                    c.put(key, (float(seed), float(i)))

        threads = [threading.Thread(target=hammer, args=(s,)) for s in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 15 distinct keys (seed%3 x i%5) end up present exactly once;
        # misses counts computes (>= distinct under benign double-compute
        # races), and the vast majority of the 1600 gets are hits.
        self.assertEqual(len(c), 3 * 5)
        self.assertGreaterEqual(c.misses, 3 * 5)
        self.assertGreater(c.hits, 1000)


class Stage1EvaluateModelCacheSourceTest(unittest.TestCase):
    def test_single_gpu_evaluate_model_uses_shared_cache_helper(self):
        source = (_REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        init_region = _source_region(
            source,
            "        self._eval_cache =",
            "        # Track one-time device placement;",
        )
        eval_region = _source_region(
            source,
            "    def evaluate_model(",
            "    @staticmethod\n    def _logits_to_classes",
        )

        self.assertIn("self._eval_cache = Stage1EvalCache()", init_region)
        self.assertIn("split_name = self._resolve_eval_split", eval_region)
        self.assertIn("cache_key = self._eval_cache.make_key(", eval_region)
        self.assertIn("self._eval_cache.put(cache_key, result)", eval_region)
        self.assertNotIn("self._eval_cache[cache_key] = result", eval_region)
        self.assertLess(
            eval_region.index("split_name = self._resolve_eval_split"),
            eval_region.index("cache_key = self._eval_cache.make_key("),
        )


class Stage1RolloutPackingSourceTest(unittest.TestCase):
    def test_recurrent_rollout_tensor_pack_batches_scalar_transfers_to_target_device(self):
        source = (_REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        if "def _stage1_scalar_episode_values_to_tensor(" not in source:
            self.fail("Stage-1 recurrent rollout scalar tensors must batch directly to target device")
        helper_region = _source_region(
            source,
            "def _stage1_scalar_episode_values_to_tensor(",
            "def _pack_recurrent_rollout_tensor_arrays(",
        )
        pack_region = _source_region(
            source,
            "def _pack_recurrent_rollout_tensor_arrays(",
            "\n\nclass RecurrentRolloutBuffer:",
        )

        self.assertIn("torch.stack", helper_region)
        self.assertIn("stacked.to(device=device, dtype=torch.float32)", helper_region)
        self.assertIn("_stage1_scalar_episode_values_to_tensor(episodes, 'logprobs', device)", pack_region)
        self.assertIn("_stage1_scalar_episode_values_to_tensor(episodes, 'values', device)", pack_region)
        self.assertNotIn("_stage1_scalar_episode_values_to_numpy", pack_region)
        self.assertNotIn("_rollout_scalar_to_float", pack_region)


class Stage1RewardHistoryWindowSourceTest(unittest.TestCase):
    def test_reward_history_uses_bounded_deque_not_front_pop(self):
        source = (_REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        init_region = _source_region(
            source,
            "        # ==================== PPO 7.1: 运行时回报归一化状态 ====================",
            "        # ==================== PDF 6.3: Return Normalization (PopArt风格) ====================",
        )
        reset_region = _source_region(
            source,
            "    def _reset_runtime_ppo_state(",
            "    def _get_stage1_resume_checkpoint_path(",
        )
        resume_region = _source_region(
            source,
            "                _ev_rt = ckpt.get(\"ev_runtime_state\", {})",
            "                # 恢复 return_normalizer（RunningMeanStd）状态",
        )
        update_region = _source_region(
            source,
            "    def update_reward_statistics(self, episode_reward):",
            "    def _detect_layer_attribute(self):",
        )

        for region in (init_region, reset_region, resume_region):
            self.assertIn("self.reward_history = deque", region)
            self.assertIn("maxlen=RUNNING_REWARD_HISTORY_SIZE", region)
        self.assertNotIn("reward_history.pop(0)", update_region)
        self.assertNotIn("len(self.reward_history) > RUNNING_REWARD_HISTORY_SIZE", update_region)

    def test_reward_statistics_maintain_running_sums_not_numpy_window_scans(self):
        source = (_REPO_ROOT / "layer_importance_evaluator.py").read_text(encoding="utf-8")
        init_region = _source_region(
            source,
            "        # ==================== PPO 7.1: 运行时回报归一化状态 ====================",
            "        # ==================== PDF 6.3: Return Normalization (PopArt风格) ====================",
        )
        reset_region = _source_region(
            source,
            "    def _reset_runtime_ppo_state(",
            "    def _get_stage1_resume_checkpoint_path(",
        )
        resume_region = _source_region(
            source,
            "                _ev_rt = ckpt.get(\"ev_runtime_state\", {})",
            "                # 恢复 return_normalizer（RunningMeanStd）状态",
        )
        helper_marker = "    def _rebuild_reward_statistics_accumulators(self):"
        if helper_marker not in source:
            self.fail("LayerImportanceEvaluator is missing reward statistics accumulator rebuild helper")
        helper_region = _source_region(
            source,
            helper_marker,
            "    def _update_curriculum_phase(self, episode):",
        )
        update_region = _source_region(
            source,
            "    def update_reward_statistics(self, episode_reward):",
            "    def _detect_layer_attribute(self):",
        )

        for region in (init_region, reset_region):
            self.assertIn("self.reward_history_sum = 0.0", region)
            self.assertIn("self.reward_history_sumsq = 0.0", region)
        self.assertIn("self._rebuild_reward_statistics_accumulators()", resume_region)
        self.assertIn("self.reward_history_sum = float(sum(", helper_region)
        self.assertIn("self.reward_history_sumsq = float(sum(", helper_region)
        self.assertIn("old_reward = float(self.reward_history[0])", update_region)
        self.assertIn("self.reward_history_sum -= old_reward", update_region)
        self.assertIn("self.reward_history_sumsq -= old_reward * old_reward", update_region)
        self.assertIn("self.reward_history_sum += episode_reward", update_region)
        self.assertIn("self.reward_history_sumsq += episode_reward * episode_reward", update_region)
        self.assertIn("variance = max(", update_region)
        self.assertIn("math.sqrt(variance)", update_region)
        self.assertNotIn("np.mean(self.reward_history)", update_region)
        self.assertNotIn("np.std(self.reward_history)", update_region)


@unittest.skipUnless(_HAS_TORCH, "layer_importance_evaluator imports torch")
class Stage1ApplyConfigurationReuseTest(unittest.TestCase):
    @staticmethod
    def _empty_split_registry(evaluator):
        evaluator.dataset_splits = {}
        evaluator.dataloaders = {}
        evaluator.dataset_splits_mm = {}
        evaluator.dataloaders_mm = {}

    def test_validation_full_batches_are_collated_once_for_repeated_evaluation(self):
        from layer_importance_evaluator import LayerImportanceEvaluator

        class CountingLoader:
            def __init__(self, batch):
                self.batch = batch
                self.iter_calls = 0

            def __iter__(self):
                self.iter_calls += 1
                yield self.batch

        evaluator = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
        self._empty_split_registry(evaluator)
        validation_dataset = object()
        validation_batch = {"input_ids": object(), "labels": object()}
        validation_loader = CountingLoader(validation_batch)
        evaluator._make_dataloader = lambda dataset: validation_loader

        LayerImportanceEvaluator._register_dataset_split(
            evaluator,
            "validation_full",
            validation_dataset,
        )

        self.assertEqual(evaluator.dataloaders["validation_full"], (validation_batch,))
        self.assertEqual(validation_loader.iter_calls, 1)
        list(evaluator.dataloaders["validation_full"])
        list(evaluator.dataloaders["validation_full"])
        self.assertEqual(validation_loader.iter_calls, 1)

    def test_training_split_keeps_lazy_dataloader(self):
        from layer_importance_evaluator import LayerImportanceEvaluator

        evaluator = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
        self._empty_split_registry(evaluator)
        train_dataset = object()
        train_loader = object()
        evaluator._make_dataloader = lambda dataset: train_loader

        LayerImportanceEvaluator._register_dataset_split(
            evaluator,
            "train",
            train_dataset,
        )

        self.assertIs(evaluator.dataloaders["train"], train_loader)

    def test_repeated_same_config_skips_handler_reinstall_but_keeps_eval_mode(self):
        from layer_importance_evaluator import (
            LayerImportanceEvaluator,
            STAGE1_ORIGINAL_FUNCTION_DEGREE,
        )

        class FakeModel:
            def __init__(self):
                self.eval_calls = 0

            def eval(self):
                self.eval_calls += 1

        class FakeHandler:
            def __init__(self):
                self.calls = []

            def restore_layer_gelu(self, layers, layer_name):
                self.calls.append(("restore_gelu", tuple(layers), layer_name))

            def restore_layer_softmax(self, layers, layer_name):
                self.calls.append(("restore_softmax", tuple(layers), layer_name))

            def replace_layer_gelu(self, layers, layer_name, *, degree):
                self.calls.append(("replace_gelu", tuple(layers), layer_name, int(degree)))

            def replace_layer_softmax(self, layers, layer_name, *, degree):
                self.calls.append(("replace_softmax", tuple(layers), layer_name, int(degree)))

        ev = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
        ev.model = FakeModel()
        ev.reversible_handler = FakeHandler()
        ev.layers_attribute = "bert.encoder.layer"
        ev._last_applied_config = None

        gelu = [1, 2, STAGE1_ORIGINAL_FUNCTION_DEGREE]
        softmax = [6, 2, STAGE1_ORIGINAL_FUNCTION_DEGREE]

        LayerImportanceEvaluator.apply_configuration(ev, gelu, softmax)
        first_install_calls = list(ev.reversible_handler.calls)
        self.assertGreater(len(first_install_calls), 0)
        self.assertEqual(ev.model.eval_calls, 1)

        LayerImportanceEvaluator.apply_configuration(ev, tuple(gelu), tuple(softmax))
        self.assertEqual(ev.reversible_handler.calls, first_install_calls)
        self.assertEqual(ev.model.eval_calls, 2)

        changed_gelu = [2, 2, STAGE1_ORIGINAL_FUNCTION_DEGREE]
        LayerImportanceEvaluator.apply_configuration(ev, changed_gelu, softmax)
        self.assertGreater(len(ev.reversible_handler.calls), len(first_install_calls))
        self.assertEqual(ev.model.eval_calls, 3)

    def test_worker_eval_path_reuses_handler_install_without_eval_cache(self):
        from layer_importance_evaluator import (
            LayerImportanceEvaluator,
            STAGE1_ORIGINAL_FUNCTION_DEGREE,
        )

        class FakeModel:
            def __init__(self):
                self.eval_calls = 0

            def eval(self):
                self.eval_calls += 1

        class FakeHandler:
            def __init__(self):
                self.calls = []

            def restore_layer_gelu(self, layers, layer_name):
                self.calls.append(("restore_gelu", tuple(layers), layer_name))

            def restore_layer_softmax(self, layers, layer_name):
                self.calls.append(("restore_softmax", tuple(layers), layer_name))

            def replace_layer_gelu(self, layers, layer_name, *, degree):
                self.calls.append(("replace_gelu", tuple(layers), layer_name, int(degree)))

            def replace_layer_softmax(self, layers, layer_name, *, degree):
                self.calls.append(("replace_softmax", tuple(layers), layer_name, int(degree)))

        ev = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
        ev.layers_attribute = "bert.encoder.layer"
        ev.dataloaders = {"validation_full": object()}
        ev._stage1_worker_eval_cache = None
        eval_calls = []
        ev._run_evaluation = lambda *_args, **_kwargs: eval_calls.append("forward") or (0.1, 0.2, 0.3, 4.0)

        model = FakeModel()
        handler = FakeHandler()
        gelu = [1, 2, STAGE1_ORIGINAL_FUNCTION_DEGREE]
        softmax = [6, 2, STAGE1_ORIGINAL_FUNCTION_DEGREE]

        LayerImportanceEvaluator._stage1_evaluate_on_model(
            ev,
            model=model,
            handler=handler,
            device="cpu",
            gelu_degrees=gelu,
            softmax_degrees=softmax,
            split_name="validation_full",
        )
        first_install_calls = list(handler.calls)
        self.assertGreater(len(first_install_calls), 0)

        LayerImportanceEvaluator._stage1_evaluate_on_model(
            ev,
            model=model,
            handler=handler,
            device="cpu",
            gelu_degrees=tuple(gelu),
            softmax_degrees=tuple(softmax),
            split_name="validation_full",
        )
        self.assertEqual(handler.calls, first_install_calls)
        self.assertEqual(model.eval_calls, 2)
        self.assertEqual(eval_calls, ["forward", "forward"])

        changed_gelu = [2, 2, STAGE1_ORIGINAL_FUNCTION_DEGREE]
        LayerImportanceEvaluator._stage1_evaluate_on_model(
            ev,
            model=model,
            handler=handler,
            device="cpu",
            gelu_degrees=changed_gelu,
            softmax_degrees=softmax,
            split_name="validation_full",
        )
        self.assertGreater(len(handler.calls), len(first_install_calls))
        self.assertEqual(model.eval_calls, 3)


class Stage1GpuEvalScriptSourceTest(unittest.TestCase):
    def test_stage1_plaintext_repeat_eval_uses_async_transfer_and_defers_sync(self):
        source = (_REPO_ROOT / "scripts" / "stage1_plaintext_repeat_eval.py").read_text(
            encoding="utf-8",
        )
        dataloader_region = _source_region(source, "def _build_dataloader(", "def _evaluate(")
        eval_region = _source_region(source, "def _evaluate(", "def main(")
        loop_region = eval_region.split("for batch in dataloader:", 1)[1].split(
            "loss_sum = float",
            1,
        )[0]

        self.assertIn("pin_memory=torch.cuda.is_available()", dataloader_region)
        self.assertIn('batch.pop("labels").to(device, non_blocking=True)', eval_region)
        self.assertIn("v.to(device, non_blocking=True)", eval_region)
        self.assertIn("loss_sum_t = torch.zeros((), dtype=torch.float64, device=device)", eval_region)
        self.assertIn("correct_t = torch.zeros((), dtype=torch.long, device=device)", eval_region)
        self.assertNotIn(".item()", loop_region)

    def test_stage1_plaintext_repeat_eval_streams_stdout_json_summary(self):
        source = (_REPO_ROOT / "scripts" / "stage1_plaintext_repeat_eval.py").read_text(
            encoding="utf-8",
        )
        main_region = _source_region(source, "def main() -> int:", 'if __name__ == "__main__":')

        self.assertIn("json.dump(summary, sys.stdout", main_region)
        self.assertIn('sys.stdout.write("\\n")', main_region)
        self.assertNotIn("print(json.dumps(summary", main_region)

    def test_mrpc_layer_noise_eval_uses_async_transfer_and_defers_cpu_lists(self):
        source = (_REPO_ROOT / "scripts" / "bert_mrpc_layer_noise_experiment.py").read_text(
            encoding="utf-8",
        )
        dataloader_region = _source_region(source, "def build_dataloader(", "def evaluate_condition(")
        eval_region = _source_region(source, "def evaluate_condition(", "def run_repeated_condition(")
        loop_region = eval_region.split("for batch in dataloader:", 1)[1].split(
            "labels_all = (",
            1,
        )[0]

        self.assertIn("pin_memory=torch.cuda.is_available()", dataloader_region)
        self.assertIn("torch_module.inference_mode()", eval_region)
        self.assertIn("labels.to(device, non_blocking=True)", eval_region)
        self.assertIn("value.to(device, non_blocking=True)", eval_region)
        self.assertIn("label_tensors.append(labels.detach().reshape(-1))", eval_region)
        self.assertIn("pred_tensors.append(preds.detach().reshape(-1))", eval_region)
        self.assertNotIn(".tolist()", loop_region)


@unittest.skipUnless(_HAS_TORCH, "torch unavailable")
class HornerPolyEquivalenceTest(unittest.TestCase):
    """Horner ``_poly`` vs the stacked-powers reference ``polynomial``."""

    def _x(self):
        torch.manual_seed(7)
        # include the piecewise boundaries and 0 exactly
        x = torch.empty(4, 3, 257).uniform_(-4.0, 4.0)
        x.view(-1)[:5] = torch.tensor([-2.7, 0.0, 2.7, -0.0, 1.0])
        return x

    def test_poly_matches_stacked_reference_all_degrees_and_signs(self):
        from function_handler import GELU_COEEF, PolynomialGELU, polynomial
        x = self._x()
        for degree in sorted(GELU_COEEF.keys()):
            mod = PolynomialGELU(degree=degree)
            for sign in (0, 1):
                got = mod._poly(x, sign)
                ref = polynomial(x, GELU_COEEF[degree], sign)
                self.assertEqual(got.shape, ref.shape)
                torch.testing.assert_close(
                    got, ref, rtol=1e-5, atol=1e-6,
                    msg=f"degree={degree} sign={sign}",
                )

    def test_paired_polys_match_independent_piece_evaluation_exactly(self):
        from function_handler import PolynomialGELU

        x = self._x()
        for degree in (2, 4):
            mod = PolynomialGELU(degree=degree)
            paired_neg, paired_pos = mod._poly_pair(x)

            self.assertTrue(torch.equal(paired_neg, mod._poly(x, 1)))
            self.assertTrue(torch.equal(paired_pos, mod._poly(x, 0)))

            first = mod._paired_coeff_tensor(x.device, x.dtype)
            second = mod._paired_coeff_tensor(x.device, x.dtype)
            self.assertIs(first, second)

    @unittest.skipUnless(
        _HAS_TORCH and torch.cuda.is_available(),
        "paired GELU size gate requires CUDA",
    )
    def test_cuda_forward_size_gate_matches_legacy_and_skips_small_tensors(self):
        from function_handler import (
            PolynomialGELU,
            _GELU_PAIRED_POLY_MIN_NUMEL,
            _select_piecewise_gelu_output,
        )

        torch.manual_seed(19)
        small = torch.empty(1024, device="cuda").uniform_(-4.0, 4.0)
        large = torch.empty(
            _GELU_PAIRED_POLY_MIN_NUMEL,
            device="cuda",
        ).uniform_(-4.0, 4.0)
        for degree in (2, 4):
            mod = PolynomialGELU(degree=degree).cuda().eval()
            mod(small)
            self.assertEqual(len(mod._paired_coeff_cache), 0)

            legacy = _select_piecewise_gelu_output(
                large,
                mod._poly(large, 1),
                mod._poly(large, 0),
            )
            got = mod(large)

            self.assertTrue(torch.equal(got, legacy))
            self.assertEqual(len(mod._paired_coeff_cache), 1)

    def test_forward_matches_reference_piecewise(self):
        from function_handler import GELU_COEEF, PolynomialGELU, polynomial
        x = self._x()
        for degree in sorted(GELU_COEEF.keys()):
            mod = PolynomialGELU(degree=degree)
            got = mod(x)
            if degree == 0:
                ref = polynomial(x, GELU_COEEF[degree], 1)
            else:
                y0 = torch.zeros_like(x)
                y1 = polynomial(x, GELU_COEEF[degree], 1)
                y2 = polynomial(x, GELU_COEEF[degree], 0)
                ref = torch.where(x < -2.7, y0, torch.zeros_like(x))
                ref = torch.where((x >= -2.7) & (x < 0), y1, ref)
                ref = torch.where((x >= 0) & (x <= 2.7), y2, ref)
                ref = torch.where(x > 2.7, x, ref)
            torch.testing.assert_close(
                got, ref, rtol=1e-5, atol=1e-6, msg=f"degree={degree}",
            )

    def test_piecewise_selector_matches_legacy_boundaries_and_special_values(self):
        from function_handler import _select_piecewise_gelu_output

        x = torch.tensor([
            float("-inf"),
            -3.0,
            -2.7,
            -1.0,
            -0.0,
            0.0,
            1.0,
            2.7,
            3.0,
            float("inf"),
            float("nan"),
        ])
        y_neg = torch.arange(x.numel(), dtype=x.dtype) + 10.0
        y_pos = torch.arange(x.numel(), dtype=x.dtype) + 20.0

        legacy = torch.where((x >= -2.7) & (x < 0), y_neg, 0.0)
        legacy = torch.where((x >= 0) & (x <= 2.7), y_pos, legacy)
        legacy = torch.where(x > 2.7, x, legacy)
        got = _select_piecewise_gelu_output(x, y_neg, y_pos)

        self.assertTrue(torch.equal(got, legacy))


@unittest.skipUnless(_HAS_TORCH, "torch unavailable")
class ExpSquaringEquivalenceTest(unittest.TestCase):
    # (degree, lower_bound) pairs from the Stage-1 softmax install path
    _LB = {1: -2.0, 2: -4.0, 3: -10.0, 4: -13.0, 5: -13.0, 6: -13.0}

    @staticmethod
    def _bert_exp(degree):
        from function_handler import BertSelfAttentionWithAproximation
        obj = BertSelfAttentionWithAproximation.__new__(
            BertSelfAttentionWithAproximation
        )
        obj.degree = degree
        return obj.approximation_exponential

    def test_matches_torch_pow_in_band(self):
        from function_handler import _approx_exponential
        torch.manual_seed(11)
        for degree in range(1, 7):
            x = torch.empty(2048).uniform_(self._LB[degree], 0.0)
            ref = torch.pow(1 + x / (2 ** degree), 2 ** degree)
            for fn in (self._bert_exp(degree), lambda v, d=degree: _approx_exponential(v, d)):
                got = fn(x)
                torch.testing.assert_close(
                    got, ref, rtol=1e-5, atol=1e-7, msg=f"degree={degree}",
                )

    def test_below_band_values_match_including_saturation(self):
        # Far-below-lower-bound inputs (additive -10000 attention mask) are
        # where-discarded by the caller, but the raw values must still agree:
        # both forms produce the same finite value or both saturate to +inf
        # (2^d is even, so negative bases square to positive).
        for degree in (1, 4, 6):
            x = torch.tensor([-50.0, -1000.0, -10000.0])
            ref = torch.pow(1 + x / (2 ** degree), 2 ** degree)
            got = self._bert_exp(degree)(x)
            self.assertTrue(
                torch.equal(torch.isinf(got), torch.isinf(ref)),
                f"degree={degree}: inf pattern diverged",
            )
            finite = torch.isfinite(ref)
            if finite.any():
                torch.testing.assert_close(
                    got[finite], ref[finite], rtol=1e-4, atol=0.0,
                )

    def test_softmax_invariant_to_additive_mask_padding_columns(self):
        """Real query rows' probs must not depend on -10000-masked pad width.

        This is the property that makes dynamic batch padding (and therefore
        ``--batch-size`` changes) safe for the approximated model: padded key
        columns sit far below ``lower_bound`` after the row-max shift, get
        where-zeroed, and contribute exactly 0 to the normalizer.
        """
        from function_handler import BertSelfAttentionWithAproximation
        torch.manual_seed(13)
        for degree in (1, 4, 6):
            obj = BertSelfAttentionWithAproximation.__new__(
                BertSelfAttentionWithAproximation
            )
            obj.degree = degree
            obj.lower_bound = self._LB[degree]
            s = 9
            scores = torch.empty(2, 4, s, s).uniform_(-3.0, 3.0)
            probs = obj.approximation_softmax(scores)
            for pad in (1, 7):
                padded = torch.cat(
                    [scores, torch.full((2, 4, s, pad), -10000.0)], dim=-1
                )
                probs_padded = obj.approximation_softmax(padded)
                torch.testing.assert_close(
                    probs_padded[..., :s], probs, rtol=1e-6, atol=1e-8,
                    msg=f"degree={degree} pad={pad}",
                )
                self.assertEqual(
                    float(probs_padded[..., s:].abs().max().item()), 0.0,
                    f"degree={degree} pad={pad}: padded columns leaked probability",
                )


@unittest.skipUnless(_HAS_TORCH, "torch unavailable")
class RunEvaluationDeferredSyncTest(unittest.TestCase):
    """The deferred-sync ``_run_evaluation`` loop must be bit-identical to the
    old per-batch-sync loop (same per-batch arrays, same float64 loss
    accumulation order)."""

    class _FakeOutput:
        def __init__(self, loss, logits):
            self.loss = loss
            self.logits = logits

    @classmethod
    def _make_fake_model(cls):
        outer = cls

        class _FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._g = torch.Generator().manual_seed(99)

            def forward(self, input_ids=None, labels=None, **kwargs):
                bs = int(input_ids.shape[0])
                logits = (
                    input_ids.float().sum(dim=-1, keepdim=True)
                    * torch.tensor([[0.013, -0.007]])
                    + torch.randn(bs, 2, generator=self._g)
                )
                loss = torch.nn.functional.cross_entropy(
                    logits, labels.reshape(-1)
                )
                return outer._FakeOutput(loss, logits)

        return _FakeModel()

    @staticmethod
    def _batches():
        g = torch.Generator().manual_seed(5)
        out = []
        for bs in (4, 4, 3):
            out.append({
                "input_ids": torch.randint(0, 1000, (bs, 7), generator=g),
                "labels": torch.randint(0, 2, (bs,), generator=g),
            })
        return out

    def _evaluator(self):
        from layer_importance_evaluator import LayerImportanceEvaluator
        ev = LayerImportanceEvaluator.__new__(LayerImportanceEvaluator)
        ev.dataset_key = "mrpc"
        ev.model = object()        # != the model override -> no .to() branch
        ev.device = "cpu"
        ev._eval_infra_ready = True
        return ev

    def _reference_old_loop(self, ev, model, dataloader):
        import numpy as np
        total_loss = 0.0
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for batch in dataloader:
                labels = ev._normalize_labels_for_metrics(
                    batch["labels"].detach().numpy()
                )
                outputs = model(**batch)
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                logits = ev._normalize_logits_for_metrics(
                    outputs.logits.detach().cpu().numpy(),
                    expected_batch_size=len(labels),
                )
                all_preds.extend(logits.tolist())
                all_labels.extend(labels.tolist())
        avg_loss = total_loss / len(dataloader)
        from sklearn.metrics import accuracy_score, f1_score
        pred_classes = np.argmax(np.array(all_preds), axis=1)
        m1 = accuracy_score(all_labels, pred_classes)
        m2 = f1_score(all_labels, pred_classes, average="weighted")
        return avg_loss, m1, m2

    def test_bit_identical_to_per_batch_sync_loop(self):
        ev = self._evaluator()
        # Two fake models with identical RNG streams: one consumed by the
        # reference loop, one by _run_evaluation (each forward draws randn).
        model_a = self._make_fake_model()
        model_b = self._make_fake_model()
        batches = self._batches()
        ref_loss, ref_m1, ref_m2 = self._reference_old_loop(ev, model_a, batches)
        loss, m1, m2, _t = ev._run_evaluation(
            batches, use_train=False, split_name="validation_full",
            model=model_b, device="cpu",
        )
        self.assertEqual(loss, ref_loss)   # exact: same fp32 values, same fp64 order
        self.assertEqual(m1, ref_m1)
        self.assertEqual(m2, ref_m2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
