import ast
import pathlib
import unittest


REPO = pathlib.Path(__file__).resolve().parents[1]


def _function_def(module_path: pathlib.Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {module_path}")


def _called_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            out.add(func.id)
        elif isinstance(func, ast.Attribute):
            out.add(func.attr)
    return out


class FusionSinglePathGuardTest(unittest.TestCase):
    def test_final_eval_fusion_decoder_uses_sf_direct_builder(self):
        fn = _function_def(REPO / "Paean" / "blb_action_eval.py", "_decode_fusion_count_fixed_action")
        calls = _called_names(fn)

        self.assertIn("build_block_cfg_from_field_values", calls)
        for forbidden in (
            "_build_block1_action",
            "_build_block2_action",
            "_build_block4_action",
            "_build_block5_action",
            "build_block1_cfg_from_action",
            "build_block2_cfg_from_action",
            "build_block4_cfg_from_action",
            "build_block5_cfg_from_action",
        ):
            self.assertNotIn(forbidden, calls)

    def test_terminal_prepare_can_replay_boosted_overrides(self):
        fn = _function_def(REPO / "blb_stage2_rl" / "env.py", "prepare_action_for_terminal_probe")
        arg_names = [arg.arg for arg in fn.args.args + fn.args.kwonlyargs]
        self.assertIn("boosted_overrides", arg_names)

        forwarded = False
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute) or func.attr != "_materialize_action":
                continue
            forwarded = any(kw.arg == "boosted_overrides" for kw in node.keywords)
        self.assertTrue(
            forwarded,
            "prepare_action_for_terminal_probe must pass boosted_overrides "
            "to the canonical materialization path",
        )

    def test_final_strict_revalidation_replays_boosted_fusion_config(self):
        source = (REPO / "blb_stage2_rl" / "layerwise_runner.py").read_text(encoding="utf-8")
        self.assertTrue(
            'boosted_overrides=strict_best_snapshot["boosted_overrides"]'
            in source,
            "final strict revalidation must pass boosted overrides into terminal prepare",
        )


if __name__ == "__main__":
    unittest.main()
