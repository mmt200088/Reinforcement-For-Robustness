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
