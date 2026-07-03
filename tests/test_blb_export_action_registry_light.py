import inspect
import unittest
from unittest import mock


class BLBExportActionRegistryLightTests(unittest.TestCase):
    def test_module_import_is_dependency_light(self):
        import scripts.blb_export_action_registry as registry

        self.assertTrue(callable(registry.build_registry_payload))

    def test_build_registry_payload_reuses_field_offsets(self):
        import scripts.blb_export_action_registry as registry

        calls = {"offsets": 0}

        def fake_per_layer_field_offsets():
            calls["offsets"] += 1
            return [
                (1, "gelu_out_sf", "F"),
                (2, "output_truncation_k", "K"),
            ]

        def fake_describe_action_vector(*_args, **_kwargs):
            return {
                "records": [
                    {
                        "global_index": 0,
                        "layer": 0,
                        "block_index": 1,
                        "block": "block1",
                        "field": "gelu_out_sf",
                        "kind": "F",
                        "operation": "fresh",
                        "location": "L0.B1.gelu_out_sf",
                        "config_name": "block1_mrpc@layer=0",
                        "effective": True,
                        "value_type": "scaling_factor",
                        "N": 16384,
                        "max_sf": 30,
                        "num_levels": 3,
                        "action_index": 2,
                        "level_values": [28, 29, 30],
                        "value": 30,
                        "distribution": "fresh",
                    },
                    {
                        "global_index": 1,
                        "layer": 0,
                        "block_index": 2,
                        "block": "block2",
                        "field": "output_truncation_k",
                        "kind": "K",
                        "operation": "truncation",
                        "location": "L0.B2.output_truncation_k",
                        "config_name": "block2_mrpc@layer=0",
                        "effective": True,
                        "value_type": "truncation_k",
                        "N": None,
                        "max_sf": None,
                        "num_levels": 3,
                        "action_index": 2,
                        "level_values": [9, 10, 11],
                        "value": 11,
                        "distribution": "",
                    },
                ]
            }

        fake_deps = {
            "K_LEVELS": [9, 10, 11],
            "describe_action_vector": fake_describe_action_vector,
            "load_max_sfs": lambda profile: {"profile": profile},
            "make_all_max_action_vector": lambda num_layers: [2, 2],
            "per_layer_field_offsets": fake_per_layer_field_offsets,
        }
        with mock.patch.object(registry, "_load_action_space_deps", return_value=fake_deps):
            payload = registry.build_registry_payload(profile="mrpc", num_layers=1)

        self.assertEqual(calls["offsets"], 1)
        self.assertEqual(payload["summary"]["per_layer_slot_count"], 2)
        self.assertEqual(payload["summary"]["block_slot_counts_per_layer"], {"block1": 1, "block2": 1})
        k_records = [r for r in payload["slot_registry_full"] if r["value_type"] == "truncation_k"]
        self.assertEqual(k_records[0]["all_max_action_index"], 2)

    def test_all_max_action_index_scans_k_levels_once_without_copy(self):
        import scripts.blb_export_action_registry as registry

        class CountingLevels:
            def __init__(self, values):
                self.values = list(values)
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                return iter(self.values)

        source = inspect.getsource(registry._all_max_action_index)
        self.assertNotIn("list(k_levels).index(max(k_levels))", source)

        k_levels = CountingLevels([9, 13, 11, 13])
        record = {"value_type": "truncation_k"}

        self.assertEqual(registry._all_max_action_index(record, k_levels=k_levels), 1)
        self.assertEqual(k_levels.iterations, 1)


if __name__ == "__main__":
    unittest.main()
