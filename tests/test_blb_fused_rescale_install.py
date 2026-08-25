"""Regression: a FUSED-away rescale's noise must NOT be installed into the model.

When the Rescale_optimizer fuses a rescale (merges consecutive cut-points into one
prime), that rescale does not happen in the real protocol, so no rescale noise may
be injected there. The optimizer signals a fused rescale by KEEPING the node in
``new_compact_config.cut_point_sf`` as a PASSTHROUGH — it carries ``sf`` (the
accumulated scale flowing through) but has NO ``sf_post`` key — NOT by dropping the
node. ``apply_optimizer_output_to_cfg`` must null the cfg rescale field for BOTH
forms (passthrough AND fully-absent node), detected straight from the optimizer
output so it adapts to ANY modulus chain (never hard-coded).

Bug (fixed 2026-06-23): the fused detection handled only ``cpt is None`` (a node
absent from cut_point_sf). A fused rescale kept as a passthrough (e.g. block4 fc=1
``ctct_rot_softmax_mul_v``: the 27-prime fuses into the next stage; the node stays
with ``sf=66`` and no ``sf_post``) fell through both branches → the cfg kept its
rescale SF → noise was installed at a rescale that never happens.

Two lanes:
  * SyntheticTest — torch-free AND rescale_optimizer-free: a hand-built optimizer
    output with one real rescale, one fused passthrough, one fully-absent fused
    node, and a tuple-indexed rescale; asserts exactly the fused ones are nulled.
  * RealReplanTest — drives a REAL replan for every committed boosted fusion option
    (block2/4/5_n2/5_n4) and asserts the cfg fields nulled by
    apply_optimizer_output_to_cfg are EXACTLY the t_new rescale positions whose
    optimizer cut_point carries no real sf_post. Skipped where rescale_optimizer
    can't be imported.
"""

import importlib.util
import json
import pathlib
import sys
import types
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[1]
for p in (
    str(_REPO),
    str(_REPO / "blb_stage2_rl"),
    str(_REPO / "configs/preparation/rescale"),
):
    if p not in sys.path:
        sys.path.insert(0, p)


try:
    from rfr.preparation.rescale.bridge import (
        DEFAULT_CFG_TO_T_NEW_MAP,
        _SkelEntry,
        apply_optimizer_output_to_cfg,
        default_rotation_name_map,
    )
    _BRIDGE_OK = True
except Exception:
    _BRIDGE_OK = False


class _NP:
    """Minimal noise-point stand-in. Carries ``distribution`` / ``N`` too because
    setting a tuple-typed rescale slot copies them onto a fresh NoisePoint."""

    def __init__(self, sf, distribution="uniform", N=16384):
        self.scaling_factor = sf
        self.distribution = distribution
        self.N = N

    def __repr__(self):
        return f"_NP({self.scaling_factor})"


@unittest.skipUnless(_BRIDGE_OK, "rfr.preparation.rescale.bridge (torch) required")
class SyntheticTest(unittest.TestCase):
    """Lock the detection on a hand-built optimizer output (no replan needed)."""

    def test_passthrough_and_absent_are_both_nulled(self):


        table = {
            "g": (
                _SkelEntry("src_fresh"),
                _SkelEntry("r_real"),
                _SkelEntry("r_passthrough"),
                _SkelEntry("r_absent"),
                _SkelEntry("sq_rescales", tuple_index=0),
            ),
        }
        baseline_skeleton = [0, 1, 2, 3, 4]
        raw = {
            "result": {"valid": True},
            "new_compact_config": {
                "cut_point_sf": [
                    {"i": 0, "name": "src", "sf": 20, "type": "source"},
                    {"i": 1, "name": "a", "sf_pre": 40, "sf_post": 18, "drop": 22, "type": "rescale"},
                    {"i": 2, "name": "b", "sf": 50, "type": "mul"},

                    {"i": 4, "name": "c", "sf_pre": 60, "sf_post": 30, "drop": 30, "type": "rescale"},
                ],
            },
        }
        cfg = types.SimpleNamespace(
            src_fresh=_NP(99), r_real=_NP(99), r_passthrough=_NP(99),
            r_absent=_NP(99), sq_rescales=[_NP(99)],
        )
        apply_optimizer_output_to_cfg(
            cfg, output_raw=raw, block_idx=1, graph_key="g",
            baseline_skeleton=baseline_skeleton, cfg_to_t_new_table=table,
        )
        self.assertEqual(cfg.src_fresh.scaling_factor, 20, "fresh source set from sf")
        self.assertEqual(cfg.r_real.scaling_factor, 18, "real rescale set from sf_post")
        self.assertIsNone(cfg.r_passthrough, "FUSED passthrough (no sf_post) must be nulled")
        self.assertIsNone(cfg.r_absent, "FUSED absent node must be nulled")
        self.assertEqual(cfg.sq_rescales[0].scaling_factor, 30, "real tuple rescale set from sf_post")

    def test_no_fusion_keeps_every_rescale(self):


        table = {"g": (_SkelEntry("src_fresh"), _SkelEntry("r1"), _SkelEntry("r2"))}
        raw = {
            "result": {"valid": True},
            "new_compact_config": {"cut_point_sf": [
                {"i": 0, "name": "src", "sf": 20, "type": "source"},
                {"i": 1, "name": "a", "sf_post": 17, "type": "rescale"},
                {"i": 2, "name": "b", "sf_post": 19, "type": "rescale"},
            ]},
        }
        cfg = types.SimpleNamespace(src_fresh=_NP(99), r1=_NP(99), r2=_NP(99))
        apply_optimizer_output_to_cfg(
            cfg, output_raw=raw, block_idx=1, graph_key="g",
            baseline_skeleton=[0, 1, 2], cfg_to_t_new_table=table,
        )
        self.assertEqual(cfg.r1.scaling_factor, 17)
        self.assertEqual(cfg.r2.scaling_factor, 19)


_BUILD = {
    "block2_mrpc": (
        "block2_mrpc.json", 2,
        ["inv_std_fresh_sf", "gamma_rescale_sf", "kt_mask1_rescale_sf", "qkt_matmul_rescale_sf"],
        {"ctct_x_mean_over_std": "x2", "ctpt_gama1": "gamma_sf", "ctpt_wq_wk": "wk_sf",
         "ctpt_rotKT_mask1": "kt_mask1_sf", "ctpt_rotKT_mask2": "kt_mask2_sf",
         "ctct_preprocess_qkt": "x2", "ctpt_mask": "qkt_merge_mask_sf"},
    ),
    "block4": (
        "block4.json", 4,
        ["softmax_out_fresh_sf", "softmax_v_matmul_rescale_sf", "ln_mean_rescale_sf", "ln_square_rescale_sf"],
        {"ctpt_mask2": "softmax_out_mask_sf", "ctct_rot_softmax_mul_v": ("v_fresh_sf", "softmax_out_mask_sf"),
         "ctpt_mask": "softmax_v_mask_sf", "ctpt_wo_attnout": "wo_sf", "ctpt_inv_d_1": "ln_mean_inv_d_sf",
         "ctct_square": "x2", "ctpt_inv_d_2": "ln_var_inv_d_sf"},
    ),
    "block5_n2": (
        "block5_n2.json", 5,
        ["x_centered_fresh_sf", "normalize_rescale_sf", "wffn1_rescale_sf", "gelu_coeff_mul_rescale_sf_0"],
        {"ctct_xmean_over_std": "x2", "ctpt_gamal": "gamma_sf", "ctpt_wffn1": "wffn1_sf",
         "ctct_gelu_x2": "x2", "ctpt_gelu_coeff": "gelu_coeff_sf"},
    ),
    "block5_n4": (
        "block5_n4.json", 5,
        ["x_centered_fresh_sf", "normalize_rescale_sf", "wffn1_rescale_sf",
         "gelu_power_rescale_sf_0", "gelu_coeff_mul_rescale_sf_0"],
        {"ctct_xmean_over_std": "x2", "ctpt_gamal": "gamma_sf", "ctpt_wffn1": "wffn1_sf",
         "ctct_gelu_x2": "x2", "ctct_gelu_x4": "x2", "ctpt_gelu_coeff": "gelu_coeff_sf"},
    ),
}


@unittest.skipUnless(
    _BRIDGE_OK and importlib.util.find_spec("rescale_optimizer") is not None,
    "rfr.preparation.rescale.bridge (torch) + rescale_optimizer required",
)
class RealReplanTest(unittest.TestCase):
    """Drive REAL replans for every boosted fusion option and assert the fields
    nulled by apply_optimizer_output_to_cfg are EXACTLY the t_new rescale positions
    whose optimizer cut_point lacks a real sf_post — proving auto-detection."""

    @classmethod
    def setUpClass(cls):
        from rfr.preparation.rescale.optimizer import ReplanSession
        from rfr.preparation.rescale.bridge import InProcessInvoker
        root = str(_REPO / "configs/preparation/rescale")
        cls.S = ReplanSession.from_profile(profile="mrpc", root=root)
        cls.inv = InProcessInvoker.from_profile(rescale_optimizer_root=root, profile="mrpc")

    def _boosted_slots(self, map_file):
        m = json.loads((_REPO / "blb_stage2_rl" / "fusion_maps" / "mrpc" / map_file).read_text())
        opt = next(o for o in m["options"] if int(o.get("fusion_count", 0)) == 1)
        self.assertTrue(opt.get("boosted") and opt.get("explicit_field_values"),
                        f"{map_file} fc=1 must be a boosted option")
        return {k: int(v) for k, v in opt["explicit_field_values"].items()}

    @staticmethod
    def _mock_cfg(graph_key, block_idx):


        entries = DEFAULT_CFG_TO_T_NEW_MAP[graph_key]
        cfg = types.SimpleNamespace()
        for se in entries:
            if se.tuple_index is None:
                setattr(cfg, se.cfg_field, _NP(99))
            else:
                setattr(cfg, se.cfg_field, [_NP(99) for _ in range(16)])
        for mapped in default_rotation_name_map(block_idx).values():
            flags = (mapped,) if isinstance(mapped, str) else tuple(mapped)
            for flag in flags:
                setattr(cfg, str(flag), False)
        return cfg

    def test_fused_rescales_auto_detected_and_nulled(self):
        for graph_key, (map_file, block_idx, t_keys, delta_spec) in _BUILD.items():
            with self.subTest(graph_key=graph_key):
                s = self._boosted_slots(map_file)
                t_new = [s[k] for k in t_keys]
                deltas = {}
                for node, spec in delta_spec.items():
                    if spec == "x2":
                        deltas[node] = "x2"
                    elif isinstance(spec, tuple):
                        deltas[node] = s[spec[0]] + s[spec[1]]
                    else:
                        deltas[node] = s[spec]
                raw = self.S.replan(graph_key, t_new=t_new, delta_overrides=deltas, return_dict=True)
                self.assertTrue(raw["result"]["valid"])
                self.assertGreaterEqual(len(raw["result"]["fusions"]), 1,
                                        f"{graph_key} fc=1 must actually fuse")
                cps = {int(e["i"]): e for e in raw["new_compact_config"]["cut_point_sf"]}
                bsk = list(self.inv.baselines[graph_key][0])
                entries = DEFAULT_CFG_TO_T_NEW_MAP[graph_key]


                expect_fused, expect_real = [], []
                for r, se in enumerate(entries):
                    if r == 0 or r >= len(bsk):
                        continue
                    cpt = cps.get(int(bsk[r]))
                    (expect_fused if (cpt is None or cpt.get("sf_post") is None)
                     else expect_real).append((se.cfg_field, se.tuple_index, cpt))
                self.assertTrue(expect_fused, f"{graph_key} fc=1 should fuse at least one rescale")

                cfg = self._mock_cfg(graph_key, block_idx)
                apply_optimizer_output_to_cfg(
                    cfg, output_raw=raw, block_idx=block_idx, graph_key=graph_key,
                    baseline_skeleton=bsk,
                )

                def _slot(field, idx):
                    return getattr(cfg, field) if idx is None else getattr(cfg, field)[int(idx)]

                for field, idx, _cpt in expect_fused:
                    self.assertIsNone(_slot(field, idx),
                                      f"{graph_key}: fused rescale {field}[{idx}] not nulled")
                for field, idx, cpt in expect_real:
                    slot = _slot(field, idx)
                    self.assertIsNotNone(slot, f"{graph_key}: real rescale {field} wrongly nulled")
                    self.assertEqual(slot.scaling_factor, int(cpt["sf_post"]),
                                     f"{graph_key}: real rescale {field} not set to optimizer sf_post")

    def test_block4_users_reported_case(self):


        s = self._boosted_slots("block4.json")
        _, block_idx, t_keys, _delta_spec = _BUILD["block4"]
        t_new = [s[k] for k in t_keys]
        deltas = {
            "ctpt_mask2": s["softmax_out_mask_sf"],
            "ctct_rot_softmax_mul_v": s["v_fresh_sf"] + s["softmax_out_mask_sf"],
            "ctpt_mask": s["softmax_v_mask_sf"], "ctpt_wo_attnout": s["wo_sf"],
            "ctpt_inv_d_1": s["ln_mean_inv_d_sf"], "ctct_square": "x2",
            "ctpt_inv_d_2": s["ln_var_inv_d_sf"],
        }
        raw = self.S.replan("block4", t_new=t_new, delta_overrides=deltas, return_dict=True)
        bsk = list(self.inv.baselines["block4"][0])
        cfg = self._mock_cfg("block4", 4)
        apply_optimizer_output_to_cfg(
            cfg, output_raw=raw, block_idx=block_idx, graph_key="block4", baseline_skeleton=bsk,
        )
        self.assertIsNone(cfg.softmax_v_matmul_rescale, "fused softmax_v_matmul must NOT install noise")
        self.assertIsNotNone(cfg.ln_mean_result_rescale, "ln_mean rescale must install noise")
        self.assertIsNotNone(cfg.ln_square_result_rescale, "ln_square rescale must install noise")


if __name__ == "__main__":
    unittest.main(verbosity=2)
