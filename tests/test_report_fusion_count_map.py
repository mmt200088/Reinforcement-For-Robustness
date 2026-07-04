import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import report_fusion_count_map as report


class FusionCountMapReportTest(unittest.TestCase):
    def test_build_report_payload_reuses_decoded_base_option(self):
        class CountingList(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        base_action = CountingList([0])
        graphs = {
            "block1_mrpc": {
                "graph_key": "block1_mrpc",
                "block_idx": 1,
                "k_slot_index": 0,
                "block_num_slots": 1,
                "options": [
                    {"option_id": 0, "fusion_count": 0, "action_indices": base_action, "slots": {}},
                    {"option_id": 1, "fusion_count": 1, "action_indices": [1], "slots": {}},
                    {"option_id": 2, "fusion_count": 1, "action_indices": [2], "slots": {}},
                ],
            }
        }
        fields_by_block = {1: [("output_truncation_k", "K", 0)]}

        payload = report._build_report_payload(
            graphs=graphs,
            fields_by_block=fields_by_block,
            schedule=[],
            group_specs=[],
            action_config_paths={},
            profile="mrpc",
            gelu=[1],
            softmax=[6],
        )

        self.assertEqual(len(payload["graphs"][0]["options"]), 3)
        self.assertLessEqual(base_action.iterations, 2)

    def test_build_report_payload_avoids_repeated_ordered_option_scans(self):
        class CountingOptions(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        options = CountingOptions(
            [
                {
                    "option_id": idx,
                    "fusion_count": idx % 3,
                    "action_indices": [idx % 2],
                    "slots": {},
                }
                for idx in range(12)
            ]
        )
        graphs = {
            "block1_mrpc": {
                "graph_key": "block1_mrpc",
                "block_idx": 1,
                "k_slot_index": 0,
                "block_num_slots": 1,
                "options": options,
            }
        }
        fields_by_block = {1: [("output_truncation_k", "K", 0)]}

        payload = report._build_report_payload(
            graphs=graphs,
            fields_by_block=fields_by_block,
            schedule=[],
            group_specs=[],
            action_config_paths={},
            profile="mrpc",
            gelu=[1],
            softmax=[6],
        )

        self.assertEqual(payload["graphs"][0]["available_fusion_counts"], [0, 1, 2])
        self.assertEqual(len(payload["graphs"][0]["options"]), 12)
        self.assertLessEqual(options.iterations, 2)

    def test_options_in_id_order_skips_sort_when_already_ordered(self):
        options = [
            {"option_id": 0, "fusion_count": 0},
            {"option_id": 1, "fusion_count": 1},
            {"option_id": 2, "fusion_count": 1},
        ]

        def fail_sorted(*_args, **_kwargs):
            raise AssertionError("ordered options should not be sorted again")

        original_sorted = getattr(report, "sorted", None)
        report.sorted = fail_sorted
        try:
            ordered = report._options_in_id_order(options)
        finally:
            if original_sorted is None:
                delattr(report, "sorted")
            else:
                report.sorted = original_sorted

        self.assertEqual([int(item["option_id"]) for item in ordered], [0, 1, 2])

    def test_options_in_id_order_sorts_unordered_options(self):
        options = [
            {"option_id": 2, "fusion_count": 1},
            {"option_id": 0, "fusion_count": 0},
            {"option_id": 1, "fusion_count": 1},
        ]

        ordered = report._options_in_id_order(options)

        self.assertEqual([int(item["option_id"]) for item in ordered], [0, 1, 2])

    def test_choose_option_scans_candidates_without_sorting(self):
        graph = {
            "graph_key": "blockX",
            "options": [
                {"option_id": 8, "fusion_count": 0, "total_bits": 1.0},
                {"option_id": 3, "fusion_count": 1, "total_bits": 99.0},
                {"option_id": 2, "fusion_count": 1, "total_bits": 100.0},
                {"option_id": 4, "fusion_count": 1, "total_bits": 0.0},
            ],
        }

        def fail_sorted(*_args, **_kwargs):
            raise AssertionError("_choose_option should not sort candidate lists")

        original_sorted = getattr(report, "sorted", None)
        report.sorted = fail_sorted
        try:
            option_id, count, clamped = report._choose_option(graph, 1)
        finally:
            if original_sorted is None:
                delattr(report, "sorted")
            else:
                report.sorted = original_sorted

        self.assertEqual(option_id, 2)
        self.assertEqual(count, 1)
        self.assertFalse(clamped)

    def test_load_maps_does_not_read_non_map_sidecars(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "block1_mrpc.json").write_text(
                json.dumps({
                    "graph_key": "block1_mrpc",
                    "options": [{"option_id": 0, "fusion_count": 0}],
                }),
                encoding="utf-8",
            )
            sidecar = root / "map_summary.json"
            sidecar.write_text("{not-json", encoding="utf-8")

            original_read_text = Path.read_text

            def guarded_read_text(path, *args, **kwargs):
                if Path(path) == sidecar:
                    raise AssertionError("sidecar should not be opened as a map")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", guarded_read_text):
                graphs = report._load_maps(root)

            self.assertEqual(list(graphs), ["block1_mrpc"])

    def test_load_maps_scans_directory_without_path_glob(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_summary.json").write_text("{}", encoding="utf-8")
            (root / "._block1_mrpc.json").write_text("{}", encoding="utf-8")
            (root / "map_summary.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "block1_mrpc.json").write_text(
                json.dumps({
                    "graph_key": "block1_mrpc",
                    "options": [{"option_id": 0, "fusion_count": 0}],
                }),
                encoding="utf-8",
            )

            with mock.patch.object(
                Path,
                "glob",
                side_effect=AssertionError("fusion map loading should not use Path.glob"),
            ):
                graphs = report._load_maps(root)

        self.assertEqual(list(graphs), ["block1_mrpc"])

    def test_public_iter_fusion_map_paths_filters_sidecars(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_summary.json").write_text("{}", encoding="utf-8")
            (root / "map_summary.json").write_text("{}", encoding="utf-8")
            (root / "block1_mrpc.json").write_text("{}", encoding="utf-8")
            (root / "block4.json").write_text("{}", encoding="utf-8")
            (root / "notes.json").write_text("{}", encoding="utf-8")

            paths = list(report.iter_fusion_map_paths(root))

        self.assertEqual([path.name for path in paths], ["block1_mrpc.json", "block4.json"])

    def test_server_command_phase2_uses_shared_map_path_filter(self):
        text = Path("SERVER_COMMAND.md").read_text(encoding="utf-8")

        self.assertIn("from scripts.report_fusion_count_map import iter_fusion_map_paths", text)
        self.assertIn('iter_fusion_map_paths(Path("blb_stage2_rl/fusion_maps/mrpc"))', text)
        self.assertNotIn('glob.glob("blb_stage2_rl/fusion_maps/mrpc/*.json")', text)

    def test_group_specs_reuses_graph_target_choices(self):
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "options": [
                    {"option_id": 0, "fusion_count": 0},
                    {"option_id": 1, "fusion_count": 1},
                ],
            },
            "block4": {
                "graph_key": "block4",
                "options": [
                    {"option_id": 0, "fusion_count": 0},
                    {"option_id": 1, "fusion_count": 1},
                ],
            },
            "block5_n1": {
                "graph_key": "block5_n1",
                "options": [
                    {"option_id": 0, "fusion_count": 0},
                    {"option_id": 1, "fusion_count": 1},
                ],
            },
        }
        schedule = [
            {"step_idx": idx, "layer_idx": idx, "graph_key": graph_key}
            for idx, graph_key in enumerate(graphs)
        ]
        calls = {}
        original_choose_option = report._choose_option

        def counted_choose_option(graph, target):
            key = (str(graph["graph_key"]), target)
            calls[key] = calls.get(key, 0) + 1
            return original_choose_option(graph, target)

        with mock.patch.object(report, "_choose_option", counted_choose_option):
            specs = report._group_specs(graphs, schedule)

        self.assertTrue(specs)
        self.assertTrue(calls)
        self.assertLessEqual(max(calls.values()), 1)

    def test_write_action_configs_reuses_static_baseline_layout(self):
        fields_by_block = {
            block_idx: [("output_truncation_k", "K", 0)]
            for block_idx in (1, 2, 3, 4, 5)
        }
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "block_idx": 2,
                "k_slot_index": 0,
                "options": [
                    {"option_id": 0, "fusion_count": 0, "action_indices": [0], "slots": {}},
                    {"option_id": 1, "fusion_count": 1, "action_indices": [1], "slots": {}},
                ],
            }
        }
        schedule = [
            {"step_idx": 0, "layer_idx": 0, "block_idx": 2, "graph_key": "block2_mrpc"},
        ]
        group_specs = [
            {
                "name": "first",
                "option_by_graph": {"block2_mrpc": 0},
                "fusion_count_by_graph": {"block2_mrpc": 0},
            },
            {
                "name": "second",
                "option_by_graph": {"block2_mrpc": 1},
                "fusion_count_by_graph": {"block2_mrpc": 1},
            },
        ]
        with tempfile.TemporaryDirectory() as td:
            original = report._make_all_max_action
            calls = 0

            def counted_make_all_max_action(*args, **kwargs):
                nonlocal calls
                calls += 1
                return original(*args, **kwargs)

            with mock.patch.object(report, "_make_all_max_action", counted_make_all_max_action):
                paths = report._write_action_configs(
                    output_dir=Path(td),
                    fields_by_block=fields_by_block,
                    graphs=graphs,
                    num_layers=1,
                    schedule=schedule,
                    group_specs=group_specs,
                    profile="mrpc",
                    gelu=[1],
                    softmax=[6],
                )

        self.assertEqual(set(paths), {"first", "second"})
        self.assertEqual(calls, 1)

    def test_write_action_configs_uses_prebuilt_option_index(self):
        fields_by_block = {
            block_idx: [("output_truncation_k", "K", 0)]
            for block_idx in (1, 2, 3, 4, 5)
        }
        fields_by_block[2] = [
            ("rescale_sf", "F", 14),
            ("output_truncation_k", "K", 0),
        ]
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "block_idx": 2,
                "k_slot_index": 1,
                "options": [
                    {"option_id": 0, "fusion_count": 0, "action_indices": [0, 0], "slots": {}},
                    {
                        "option_id": 1,
                        "fusion_count": 1,
                        "action_indices": [3, 5],
                        "slots": {"rescale_sf": 11},
                    },
                ],
            }
        }
        schedule = [
            {"step_idx": idx, "layer_idx": idx, "block_idx": 2, "graph_key": "block2_mrpc"}
            for idx in range(2)
        ]
        group_specs = [
            {
                "name": "first",
                "option_by_graph": {"block2_mrpc": 1},
                "fusion_count_by_graph": {"block2_mrpc": 1},
            },
            {
                "name": "second",
                "option_by_graph": {"block2_mrpc": 0},
                "fusion_count_by_graph": {"block2_mrpc": 0},
            },
        ]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(
                report,
                "_option_by_id",
                side_effect=AssertionError("action-config writer should reuse a prebuilt option index"),
            ):
                paths = report._write_action_configs(
                    output_dir=Path(td),
                    fields_by_block=fields_by_block,
                    graphs=graphs,
                    num_layers=2,
                    schedule=schedule,
                    group_specs=group_specs,
                    profile="mrpc",
                    gelu=[1, 1],
                    softmax=[6, 6],
                )

            first = json.loads(Path(paths["first"]).read_text(encoding="utf-8"))

        self.assertEqual(set(paths), {"first", "second"})
        self.assertEqual(first["base"][1:3], [3, report.BASELINE_K_INDEX])
        self.assertEqual(first["base"][7:9], [3, report.BASELINE_K_INDEX])
        self.assertEqual(first["slots"][0]["scaling_factor"], 11)

    def test_splice_group_slots_reuses_field_kind_lookup(self):
        class CountingFields(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        block2_fields = CountingFields([
            ("rescale_sf", "F", 14),
            ("output_truncation_k", "K", 0),
        ])
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "block_idx": 2,
                "options": [
                    {
                        "option_id": 1,
                        "fusion_count": 1,
                        "action_indices": [3, 5],
                        "slots": {"rescale_sf": 11},
                    },
                ],
            }
        }
        schedule = [
            {"step_idx": idx, "layer_idx": idx, "block_idx": 2, "graph_key": "block2_mrpc"}
            for idx in range(2)
        ]

        slots = report._splice_group_slots(
            fields_by_block={2: block2_fields},
            graphs=graphs,
            schedule=schedule,
            option_by_graph={"block2_mrpc": 1},
        )

        self.assertEqual(len(slots), 2)
        self.assertLessEqual(block2_fields.iterations, 1)

    def test_splice_group_action_reuses_adjusted_block_actions(self):
        class CountingAction(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        action_indices = CountingAction([3, 5])
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "block_idx": 2,
                "k_slot_index": 1,
                "options": [
                    {
                        "option_id": 1,
                        "fusion_count": 1,
                        "action_indices": action_indices,
                        "slots": {},
                    },
                ],
            }
        }
        fields_by_block = {2: [("rescale_sf", "F", 14), ("output_truncation_k", "K", 0)]}
        schedule = [
            {"step_idx": idx, "layer_idx": idx, "block_idx": 2, "graph_key": "block2_mrpc"}
            for idx in range(2)
        ]

        action = report._splice_group_action(
            fields_by_block=fields_by_block,
            graphs=graphs,
            num_layers=2,
            schedule=schedule,
            option_by_graph={"block2_mrpc": 1},
            base_action=[0, 0, 0, 0],
            layer_width=2,
            block_offsets={2: 0},
        )

        self.assertEqual(action, [3, report.BASELINE_K_INDEX, 3, report.BASELINE_K_INDEX])
        self.assertLessEqual(action_indices.iterations, 1)

    def test_splice_group_slots_reuses_bound_slot_entries(self):
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "block_idx": 2,
                "options": [
                    {
                        "option_id": 1,
                        "fusion_count": 1,
                        "action_indices": [3, 5],
                        "slots": {"rescale_sf": 11},
                    },
                ],
            }
        }
        fields_by_block = {2: [("rescale_sf", "F", 14), ("output_truncation_k", "K", 0)]}
        schedule = [
            {"step_idx": idx, "layer_idx": idx, "block_idx": 2, "graph_key": "block2_mrpc"}
            for idx in range(2)
        ]
        calls = 0
        original_bound = report._bound_slot_values

        def counted_bound_slot_values(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_bound(*args, **kwargs)

        with mock.patch.object(report, "_bound_slot_values", counted_bound_slot_values):
            slots = report._splice_group_slots(
                fields_by_block=fields_by_block,
                graphs=graphs,
                schedule=schedule,
                option_by_graph={"block2_mrpc": 1},
            )

        self.assertEqual(len(slots), 2)
        self.assertLessEqual(calls, 1)

    def test_bound_slot_values_uses_mapping_items_without_dict_copy(self):
        slots = {
            "inv_std_fresh_sf": 11,
            "wk_sf": 12,
            "kt_mask1_sf": 13,
        }

        with mock.patch(
            "builtins.dict",
            side_effect=AssertionError("bound slot values should not copy mapping through dict()"),
        ):
            out = report._bound_slot_values(2, slots)

        self.assertEqual(out["inv_std_fresh_sf"], 11)
        self.assertEqual(out["x_centered_fresh_sf"], 11)
        self.assertEqual(out["wq_sf"], 12)
        self.assertEqual(out["q_mask1_sf"], 13)

    def test_option_slot_summary_uses_mapping_items_without_dict_copy(self):
        graph = {"block_idx": 2}
        fields = [("rescale_sf", "F", 14), ("output_truncation_k", "K", 0)]
        option = {
            "option_id": 1,
            "fusion_count": 1,
            "action_indices": [3, 5],
            "slots": {"rescale_sf": 11},
        }
        base = {
            "option_id": 0,
            "fusion_count": 0,
            "action_indices": [0, 0],
            "slots": {"rescale_sf": 10},
        }

        with mock.patch(
            "builtins.dict",
            side_effect=AssertionError("option slot summary should not copy mapping through dict()"),
        ):
            summary = report._option_slot_summary(graph, fields, option, base)

        self.assertEqual(summary["real_slots"], {"rescale_sf": 11})
        self.assertEqual(summary["changed_real_slots_vs_fusion0"][0]["base_value"], 10)
        self.assertEqual(summary["changed_real_slots_vs_fusion0"][0]["value"], 11)

    def test_build_report_payload_normalizes_base_slots_without_dict_copy(self):
        source = inspect.getsource(report._build_report_payload)

        self.assertNotIn('dict(base.get("slots"', source)

    def test_graph_occurrences_accumulates_unique_layers_without_list_dedupe(self):
        source = inspect.getsource(report._graph_occurrences)

        self.assertNotIn('append(int(step["layer_idx"]))', source)
        self.assertNotIn("sorted(set(v))", source)

        schedule = [
            {"graph_key": "block2_mrpc", "layer_idx": 2},
            {"graph_key": "block2_mrpc", "layer_idx": 0},
            {"graph_key": "block2_mrpc", "layer_idx": 2},
            {"graph_key": "block4", "layer_idx": 1},
        ]

        self.assertEqual(
            report._graph_occurrences(schedule),
            {"block2_mrpc": [0, 2], "block4": [1]},
        )

    def test_option_slot_summary_indexes_action_sequences_without_list_copy(self):
        class CountingAction(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        source = inspect.getsource(report._option_slot_summary)
        self.assertNotIn('action = [int(v) for v in option.get("action_indices", [])]', source)
        self.assertNotIn('base_action = [int(v) for v in base_option.get("action_indices", [])]', source)

        action = CountingAction([3, 5])
        base_action = CountingAction([0, 0])
        summary = report._option_slot_summary(
            {"block_idx": 2},
            [("rescale_sf", "F", 14), ("output_truncation_k", "K", 0)],
            {
                "option_id": 1,
                "fusion_count": 1,
                "action_indices": action,
                "slots": {},
            },
            {
                "option_id": 0,
                "fusion_count": 0,
                "action_indices": base_action,
                "slots": {},
            },
        )

        self.assertEqual(action.iterations, 0)
        self.assertEqual(base_action.iterations, 0)
        self.assertEqual(summary["changed_raw_slots_vs_fusion0"][0]["action_index"], 3)

    def test_build_report_payload_passes_base_action_sequence_without_copy(self):
        class CountingAction(list):
            def __init__(self, values):
                super().__init__(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return super().__iter__()

        source = inspect.getsource(report._build_report_payload)
        self.assertNotIn('base_action = [int(v) for v in base.get("action_indices", [])]', source)

        base_action = CountingAction([0, 0])
        graphs = {
            "block2_mrpc": {
                "graph_key": "block2_mrpc",
                "block_idx": 2,
                "k_slot_index": 1,
                "block_num_slots": 2,
                "options": [
                    {
                        "option_id": 0,
                        "fusion_count": 0,
                        "action_indices": base_action,
                        "slots": {},
                    },
                    {
                        "option_id": 1,
                        "fusion_count": 1,
                        "action_indices": [3, 5],
                        "slots": {},
                    },
                ],
            }
        }
        payload = report._build_report_payload(
            graphs=graphs,
            fields_by_block={2: [("rescale_sf", "F", 14), ("output_truncation_k", "K", 0)]},
            schedule=[],
            group_specs=[],
            action_config_paths={},
            profile="mrpc",
            gelu=[1],
            softmax=[6],
        )

        self.assertEqual(base_action.iterations, 0)
        self.assertEqual(payload["graphs"][0]["options"][1]["changed_raw_slots_vs_fusion0"][0]["action_index"], 3)


if __name__ == "__main__":
    unittest.main()
