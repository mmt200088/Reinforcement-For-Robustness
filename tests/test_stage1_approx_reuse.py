"""Bit-identity test for the Stage-1 approx-module reuse optimization.

The optimization caches the per-layer ``BertSelfAttentionWithAproximation`` /
``PolynomialGELU`` modules across episodes instead of reconstructing them every
time ``replace_layer_softmax`` / ``replace_layer_gelu`` is called. The claim it
must satisfy is strict: *"accelerated and non-accelerated produce exactly the
same result"*.

This test proves that by running two identical models — one with
``reuse_approx_modules=True`` (fast path) and one with ``False`` (original
reconstruct-every-call path) — through the exact Stage-1 install sequence over a
schedule of changing GELU/Softmax degrees, and asserting the output logits are
``torch.equal`` (CPU → deterministic, so equality is bitwise) at every step.

Requires torch + transformers (mirrors test_blb_action_mask.py); skipped
otherwise. Run on the server with: ``python tests/test_stage1_approx_reuse.py``.
"""
from __future__ import annotations

import copy
import unittest

try:
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    from rfr.search.runtime.model_handler import ReversibleLayerHandler
    _HAVE_TORCH = True
except Exception as _exc:  # pragma: no cover - env-dependent
    _HAVE_TORCH = False
    _IMPORT_ERROR = _exc


_ORIGINAL_DEGREE = -1
_GELU_GROUPS = [0, 1, 2, 4]
_SOFTMAX_GROUPS = list(range(2, 7))
_LAYERS_ATTR = "model.bert.encoder.layer"


def _install_stage1_config(handler, gelu_degrees, softmax_degrees):
    """Mirror ``LayerImportanceEvaluator._stage1_evaluate_on_model`` install."""
    original_gelu = [i for i, d in enumerate(gelu_degrees) if int(d) == _ORIGINAL_DEGREE]
    if original_gelu:
        handler.restore_layer_gelu(original_gelu, _LAYERS_ATTR)
    original_softmax = [i for i, d in enumerate(softmax_degrees) if int(d) == _ORIGINAL_DEGREE]
    if original_softmax:
        handler.restore_layer_softmax(original_softmax, _LAYERS_ATTR)

    gelu_map = {d: [] for d in _GELU_GROUPS}
    for i, d in enumerate(gelu_degrees):
        d = int(d)
        if d in gelu_map:
            gelu_map[d].append(i)
    for d in _GELU_GROUPS:
        if gelu_map[d]:
            handler.replace_layer_gelu(gelu_map[d], _LAYERS_ATTR, degree=d)

    softmax_map = {d: [] for d in _SOFTMAX_GROUPS}
    for i, d in enumerate(softmax_degrees):
        d = int(d)
        if d in softmax_map:
            softmax_map[d].append(i)
    for d in _SOFTMAX_GROUPS:
        if softmax_map[d]:
            handler.replace_layer_softmax(softmax_map[d], _LAYERS_ATTR, degree=d)


def _build_tiny_model(seed=1234):
    torch.manual_seed(seed)
    cfg = BertConfig(
        vocab_size=120,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=2,
    )


    cfg._attn_implementation = "eager"
    model = BertForSequenceClassification(cfg)
    model.eval()
    return model


def _fixed_batch(seed=7):
    g = torch.Generator().manual_seed(seed)
    bsz, seqlen = 4, 12
    input_ids = torch.randint(0, 120, (bsz, seqlen), generator=g)
    attention_mask = torch.ones((bsz, seqlen), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@unittest.skipUnless(_HAVE_TORCH, "torch/transformers not available")
class Stage1ApproxReuseBitIdentityTest(unittest.TestCase):


    SCHEDULE = [
        ([4, 4, 4], [6, 6, 6]),
        ([2, 1, 4], [4, 2, 5]),
        ([4, 4, 4], [6, 6, 6]),
        ([1, 2, 1], [2, 3, 6]),
        ([2, 4, 2], [5, 4, 3]),
    ]

    def test_logits_bit_identical_reuse_vs_reconstruct(self):
        base = _build_tiny_model()
        model_fast = copy.deepcopy(base)
        model_slow = copy.deepcopy(base)

        handler_fast = ReversibleLayerHandler(model_fast)
        handler_fast.reuse_approx_modules = True
        handler_slow = ReversibleLayerHandler(model_slow)
        handler_slow.reuse_approx_modules = False

        batch = _fixed_batch()

        with torch.inference_mode():
            for step, (gelu, softmax) in enumerate(self.SCHEDULE):
                _install_stage1_config(handler_fast, gelu, softmax)
                _install_stage1_config(handler_slow, gelu, softmax)
                model_fast.eval()
                model_slow.eval()
                out_fast = model_fast(**batch).logits
                out_slow = model_slow(**batch).logits
                self.assertTrue(
                    torch.equal(out_fast, out_slow),
                    msg=(f"step {step} config gelu={gelu} softmax={softmax}: "
                         f"logits differ; max|diff|="
                         f"{(out_fast - out_slow).abs().max().item():.3e}"),
                )

    def test_reuse_actually_avoids_reconstruction(self):
        """The fast handler must rebuild each module once; the slow handler
        rebuilds on every config. This proves the mechanism engaged (so the
        bit-identity above is meaningful, not a no-op)."""
        base = _build_tiny_model()
        handler_fast = ReversibleLayerHandler(copy.deepcopy(base))
        handler_fast.reuse_approx_modules = True
        handler_slow = ReversibleLayerHandler(copy.deepcopy(base))
        handler_slow.reuse_approx_modules = False

        for gelu, softmax in self.SCHEDULE:
            _install_stage1_config(handler_fast, gelu, softmax)
            _install_stage1_config(handler_slow, gelu, softmax)

        n_layers = 3
        n_configs = len(self.SCHEDULE)

        self.assertEqual(handler_slow._approx_softmax_rebuilds, n_layers * n_configs)


        self.assertLessEqual(handler_fast._approx_softmax_rebuilds, n_layers)
        self.assertLess(handler_fast._approx_softmax_rebuilds, handler_slow._approx_softmax_rebuilds)

    def test_real_apply_configuration_engages_cache_and_is_identical(self):
        """Drive the REAL ``LayerImportanceEvaluator.apply_configuration`` (the
        single-GPU Stage-1 RL install path) across a schedule of self-selected
        configs, proving the cache engages on the production code (not just the
        test mirror) and that results stay bit-identical to reconstruct-every-call.

        The multi-GPU worker path (``_stage1_evaluate_on_model``) uses a
        character-for-character identical install block on a per-worker handler,
        so this also covers its cache engagement."""
        try:
            from rfr.search.common import evaluator as lie
        except Exception as exc:  # pragma: no cover - heavy import, env-dependent
            self.skipTest(f"layer_importance_evaluator import failed: {exc}")
        from types import SimpleNamespace

        base = _build_tiny_model()
        model_fast = copy.deepcopy(base)
        model_slow = copy.deepcopy(base)
        h_fast = ReversibleLayerHandler(model_fast)
        h_fast.reuse_approx_modules = True
        h_slow = ReversibleLayerHandler(model_slow)
        h_slow.reuse_approx_modules = False


        self_fast = SimpleNamespace(model=model_fast, reversible_handler=h_fast,
                                    layers_attribute="bert.encoder.layer")
        self_slow = SimpleNamespace(model=model_slow, reversible_handler=h_slow,
                                    layers_attribute="bert.encoder.layer")

        batch = _fixed_batch()
        with torch.inference_mode():
            for step, (gelu, softmax) in enumerate(self.SCHEDULE):
                lie.LayerImportanceEvaluator.apply_configuration(self_fast, gelu, softmax)


                self_slow._last_applied_config = None
                h_slow._last_stage1_applied_config = None
                lie.LayerImportanceEvaluator.apply_configuration(self_slow, gelu, softmax)
                model_fast.eval()
                model_slow.eval()
                out_fast = model_fast(**batch).logits
                out_slow = model_slow(**batch).logits
                self.assertTrue(
                    torch.equal(out_fast, out_slow),
                    msg=(f"[real apply_configuration] step {step} gelu={gelu} "
                         f"softmax={softmax}: logits differ"),
                )

        n_layers = 3
        n_configs = len(self.SCHEDULE)

        self.assertEqual(h_slow._approx_softmax_rebuilds, n_layers * n_configs)

        self.assertLessEqual(h_fast._approx_softmax_rebuilds, n_layers)

    def test_fresh_equivalence_guard_blocks_reuse_when_hook_present(self):
        """If a BLB-style per-instance hook is present, reuse must fall back to
        reconstruct (so the BLB Stage-2 path is never silently changed)."""
        base = _build_tiny_model()
        handler = ReversibleLayerHandler(copy.deepcopy(base))
        handler.reuse_approx_modules = True

        _install_stage1_config(handler, [4, 4, 4], [6, 6, 6])
        before = handler._approx_softmax_rebuilds


        cached = handler._approx_softmax_cache[1]
        cached._block4_v_hook = lambda v: v
        self.assertFalse(handler._approx_attn_is_fresh_equivalent(cached))


        _install_stage1_config(handler, [4, 4, 4], [6, 6, 6])
        self.assertEqual(handler._approx_softmax_rebuilds, before + 1)


if __name__ == "__main__":
    if not _HAVE_TORCH:
        print(f"[skip] torch/transformers unavailable: {_IMPORT_ERROR}")
    unittest.main()
