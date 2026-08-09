import builtins
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class BLBCandidateStoreIdentityTests(unittest.TestCase):
    @staticmethod
    def _representative_compact_fixture():
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [
            (action_idx * 5 + 1) % 6
            for action_idx in range(73 * 12 + 1)
        ]
        action_matrix = [
            [
                (layer_idx * 3 + slot_idx) % 6
                for slot_idx in range(6)
            ]
            for layer_idx in range(12)
        ]
        base_context = {
            "action_space_version": "stage2_layerwise_12x6_v1",
            "registry_hash": "registry-" + "a" * 64,
            "max_sfs_hash": "max-sfs-" + "b" * 64,
            "stage1_config_content_hash": "stage1-" + "c" * 64,
            "stage1_gelu_degrees": [4, 2, 1, 4, 2, 1, 4, 2, 1, 4, 2, 1],
            "stage1_softmax_degrees": [6] * 12,
            "profile": "mrpc",
            "dataset": "mrpc",
            "model": "bert-base",
            "rescale_optimizer_mode": "in_process_real",
            "rescale_optimizer_root": "Rescale_optimizer",
            "rescale_optimizer_canonical_hash": "rescale-" + "d" * 64,
            "decode_version": "layerwise-decode-v1",
            "metric_policy_version": "mrpc-acc-f1-std-v1",
            "threshold_policy_hash": "threshold-" + "e" * 64,
        }
        contexts = {
            fidelity: {**base_context, "fidelity": fidelity}
            for fidelity in ("F1", "F4")
        }
        boosted_overrides = [
            {
                "block_idx": block_idx,
                "layer_idx": layer_idx,
                "field_values": {
                    "q_mask_rescale_sf": 47 + layer_idx,
                    "v_mask_rescale_sf": 53 + block_idx + layer_idx,
                },
            }
            for layer_idx in range(12)
            for block_idx in (2, 4)
        ]
        f1_trials = TrialSeries(
            loss=[0.301, 0.302, 0.303, 0.304, 0.305],
            metric1=[0.901, 0.902, 0.903, 0.904, 0.905],
            metric2=[0.801, 0.802, 0.803, 0.804, 0.805],
            seeds=[101, 102, 103, 104, 105],
        )
        f4_trials = TrialSeries(
            loss=[0.290 + idx * 0.001 for idx in range(8)],
            metric1=[0.910 - idx * 0.001 for idx in range(8)],
            metric2=[0.810 - idx * 0.001 for idx in range(8)],
            seeds=list(range(201, 209)),
        )
        metadata = {
            "F1": {
                "identity_context": contexts["F1"],
                "fidelity": "F1",
                "episode_index": 120,
                "action_matrix": action_matrix,
                "variable_cost": 0.625,
                "boosted_overrides_hash": "overrides-" + "f" * 64,
                "boosted_overrides": boosted_overrides,
                "boosted_overrides_provenance": "layerwise_env",
                "assessment_bootstrap_seed": 77,
                "episode_reward": 1.25,
                "promotion_marker": "online_group",
            },
            "F4": {
                "identity_context": contexts["F4"],
                "fidelity": "F4",
                "action_matrix": action_matrix,
                "variable_cost": 0.625,
                "boosted_overrides_hash": "overrides-" + "f" * 64,
                "boosted_overrides": boosted_overrides,
                "boosted_overrides_provenance": "layerwise_env",
                "assessment_bootstrap_seed": 77,
                "episode_reward": 1.25,
                "promotion_marker": "fresh_top_up",
                "promotion_status": "pending_reassessment",
            },
        }
        status_metadata = {
            "action_matrix": action_matrix,
            "variable_cost": 0.625,
            "boosted_overrides": boosted_overrides,
            "episode_reward": 1.25,
            "assessment_bootstrap_seed": 77,
        }
        return {
            "action": action,
            "contexts": contexts,
            "trials": {"F1": f1_trials, "F4": f4_trials},
            "metadata": metadata,
            "status_metadata": status_metadata,
            "boosted_overrides": boosted_overrides,
        }

    def _populate_equivalent_store(self, store, fixture, *, compact):
        for fidelity in ("F1", "F4"):
            kwargs = {"compact": True} if compact else {}
            store.append_trial_group(
                fixture["action"],
                fixture["trials"][fidelity],
                fixture["metadata"][fidelity],
                **kwargs,
            )
        if compact:
            store.append_promotion_status(
                fixture["action"],
                fixture["contexts"]["F4"],
                status="promoted",
                metadata=fixture["status_metadata"],
            )
        else:
            store.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": fixture["action"],
                "effective_action_indices": fixture["action"],
                "identity_context": fixture["contexts"]["F4"],
                "fidelity": "F4",
                "valid": True,
                "promotion_status": "promoted",
                "promotion_metadata": fixture["status_metadata"],
            })

    @staticmethod
    def _logical_evidence_snapshot(evidence, fidelity):
        metadata_fields = (
            "identity_context",
            "fidelity",
            "episode_index",
            "action_matrix",
            "variable_cost",
            "boosted_overrides_hash",
            "boosted_overrides_provenance",
            "assessment_bootstrap_seed",
            "episode_reward",
            "promotion_marker",
            "promotion_status",
        )
        if fidelity == "F4":
            metadata_fields += ("boosted_overrides",)
        return {
            "candidate_key": evidence.candidate_key,
            "action_indices": evidence.action_indices,
            "loss": evidence.trials.loss,
            "metric1": evidence.trials.metric1,
            "metric2": evidence.trials.metric2,
            "seeds": evidence.trials.seeds,
            "groups": tuple(
                {
                    name: group[name]
                    for name in metadata_fields
                    if name in group
                }
                for group in evidence.groups
            ),
            "promotion_attempted": evidence.promotion_attempted,
            "promotion_status": evidence.promotion_status,
            "trial_count": evidence.trial_count,
        }

    def test_flat_action_normalization_avoids_per_item_nested_checks(self):
        from blb_stage2_rl import candidate_store as store_mod

        class IntLike:
            def __init__(self, value):
                self.value = int(value)

            def __int__(self):
                return self.value

        items = [IntLike(1), IntLike(2), IntLike(3)]
        item_ids = {id(item) for item in items}
        original_isinstance = builtins.isinstance

        def guarded_isinstance(obj, classinfo):
            if id(obj) in item_ids and classinfo == (list, tuple):
                raise AssertionError("flat action normalization should not run nested checks per item")
            return original_isinstance(obj, classinfo)

        with mock.patch.object(
            store_mod,
            "isinstance",
            create=True,
            side_effect=guarded_isinstance,
        ):
            self.assertEqual(store_mod.normalize_action_indices(items), [1, 2, 3])

    def test_normalize_action_indices_accepts_ndarray_without_tolist_materialization(self):
        import numpy as np

        from blb_stage2_rl import candidate_store as store_mod

        class NoToListArray(np.ndarray):
            def tolist(self):
                raise AssertionError("candidate action ndarray should not be copied through tolist")

        action = np.asarray([[1, 2], [3, 4]], dtype=int).view(NoToListArray)
        self.assertEqual(store_mod.normalize_action_indices(action), [1, 2, 3, 4])

    def test_nested_one_shot_action_iterator_is_not_partially_consumed(self):
        from blb_stage2_rl import candidate_store as store_mod

        action = (item for item in (1, [2, 3], 4))

        self.assertEqual(
            store_mod.normalize_action_indices(action),
            [1, 2, 3, 4],
        )
        self.assertNotEqual(
            store_mod.action_hash(item for item in (1, [2, 3], 4)),
            store_mod.action_hash([4]),
        )

    def test_action_hash_caches_by_normalized_action_tuple(self):
        from blb_stage2_rl import candidate_store as store_mod

        action = [1, 2, 3]
        original_dumps = store_mod.json.dumps
        dumps_calls = 0

        def counting_dumps(*args, **kwargs):
            nonlocal dumps_calls
            dumps_calls += 1
            return original_dumps(*args, **kwargs)

        store_mod._action_hash_from_tuple.cache_clear()
        with mock.patch.object(store_mod.json, "dumps", side_effect=counting_dumps):
            first = store_mod.action_hash(action)
            second = store_mod.action_hash(list(action))
            action.append(4)
            third = store_mod.action_hash(action)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(dumps_calls, 0)

    def test_read_all_skips_blank_lines_without_strip_copy(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        class NoStripLine(str):
            def strip(self, *_args, **_kwargs):
                raise AssertionError("CandidateStore.read_all should not allocate strip() copies")

        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    NoStripLine('{"action_indices": [1], "valid": true}\n'),
                    NoStripLine("   \n"),
                    NoStripLine('{"action_indices": [2], "valid": true}\n'),
                ])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            path.touch()
            store = CandidateStore(path)
            original_open = Path.open

            def guarded_open(open_path, *args, **kwargs):
                if Path(open_path) == path:
                    return FakeHandle()
                return original_open(open_path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                records = store.read_all()

        self.assertEqual([record["action_indices"] for record in records], [[1], [2]])

    def test_append_writes_each_jsonl_record_as_one_complete_row(self):
        from blb_stage2_rl import candidate_store as store_mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = store_mod.CandidateStore(path)
            fake_handle = mock.MagicMock()
            fake_handle.__enter__.return_value = fake_handle
            fake_handle.__exit__.return_value = None
            original_open = Path.open

            def guarded_open(open_path, *args, **kwargs):
                if Path(open_path) == path:
                    return fake_handle
                return original_open(open_path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                with mock.patch.object(store_mod.os, "fsync"):
                    saved = store.append({
                        "action_indices": [1, 2, 3],
                        "fidelity": "F1",
                        "valid": True,
                    })

        self.assertEqual(saved["action_indices"], [1, 2, 3])
        fake_handle.writelines.assert_not_called()
        fake_handle.write.assert_called_once()
        row = fake_handle.write.call_args.args[0]
        self.assertTrue(row.endswith("\n"))
        self.assertEqual(json.loads(row)["action_indices"], [1, 2, 3])

    def test_first_append_fsyncs_row_and_parent_directory(self):
        from blb_stage2_rl import candidate_store as store_mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = store_mod.CandidateStore(path)
            with mock.patch.object(
                    store_mod.os, "fsync",
            ) as fsync, mock.patch.object(
                    store_mod.os, "open", wraps=store_mod.os.open,
            ) as open_directory:
                store.append({
                    "action_indices": [1, 2, 3],
                    "fidelity": "F1",
                    "valid": True,
                })

            self.assertEqual(fsync.call_count, 2)
            open_directory.assert_called_once_with(
                os.fspath(path.parent),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )

    def test_recovery_marker_is_fsynced_before_return(self):
        from blb_stage2_rl import candidate_store as store_mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = store_mod.CandidateStore(path)
            store.append({"action_indices": [1], "fidelity": "F1", "valid": True})
            committed_size = path.stat().st_size
            store.append({"action_indices": [2], "fidelity": "F1", "valid": True})

            with mock.patch.object(store_mod.os, "fsync") as fsync:
                store.recover_to_checkpoint_size(committed_size)

            fsync.assert_called_once()

    def test_malformed_tail_repair_is_fsynced_before_read_returns(self):
        from blb_stage2_rl import candidate_store as store_mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            path.write_bytes(b'{"action_indices":[1]}\n{"record_type":"broken"')
            store = store_mod.CandidateStore(path)

            with mock.patch.object(store_mod.os, "fsync") as fsync:
                records = store.read_all()

            fsync.assert_called_once()
            self.assertEqual([row["action_indices"] for row in records], [[1]])

    def test_complete_tail_newline_repair_is_fsynced_before_read_returns(self):
        from blb_stage2_rl import candidate_store as store_mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            path.write_bytes(b'{"action_indices":[1]}')
            store = store_mod.CandidateStore(path)

            with mock.patch.object(store_mod.os, "fsync") as fsync:
                records = store.read_all()

            fsync.assert_called_once()
            self.assertEqual([row["action_indices"] for row in records], [[1]])

    def test_read_all_discards_only_a_malformed_unterminated_tail(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append({"action_indices": [1], "fidelity": "F1", "valid": True})
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"record_type":"candidate_trial_group_v1"')

            records = CandidateStore(path).read_all()

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["action_indices"], [1])
            repaired = path.read_text(encoding="utf-8")
            self.assertTrue(repaired.endswith("\n"))
            self.assertNotIn("candidate_trial_group_v1", repaired)

    def test_read_all_preserves_a_complete_unterminated_tail(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            first = {"action_indices": [1], "fidelity": "F1", "valid": True}
            second = {"action_indices": [2], "fidelity": "F1", "valid": True}
            path.write_text(
                json.dumps(first, sort_keys=True) + "\n" + json.dumps(second, sort_keys=True),
                encoding="utf-8",
            )

            records = CandidateStore(path).read_all()

            self.assertEqual([row["action_indices"] for row in records], [[1], [2]])
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_read_all_still_rejects_newline_terminated_corruption(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            path.write_text(
                '{"action_indices":[1]}\n{not-json}\n{"action_indices":[2]}\n',
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                CandidateStore(path).read_all()

    def test_candidate_key_binds_action_and_context_hashes(self):
        from blb_stage2_rl.candidate_store import (
            build_candidate_identity_context,
            candidate_key,
        )

        action = [4, 4, 3, 2]
        base = build_candidate_identity_context(
            action_space_version="current-code-v1",
            registry_hash="registry-a",
            max_sfs_hash="max-sfs-a",
            stage1_hash="stage1-a",
            stage1_degrees={"gelu": [4], "softmax": [2]},
            profile="mrpc",
            rescale_optimizer_mode="in_process_real",
            rescale_optimizer_root="Rescale_optimizer",
            rescale_optimizer_hash="rescale-a",
            decode_version="decode-v1",
            dataset="mrpc",
            model="bert-base",
            metric_policy_version="mrpc-acc-f1-std-v1",
            threshold_policy_hash="threshold-a",
        )
        same_action_different_registry = dict(base)
        same_action_different_registry["registry_hash"] = "registry-b"

        self.assertEqual(candidate_key(action, base), candidate_key(action, dict(base)))
        self.assertNotEqual(candidate_key(action, base), candidate_key(action, same_action_different_registry))

    def test_candidate_identity_uses_phase1_canonical_context_fields(self):
        from blb_stage2_rl.candidate_store import build_candidate_identity_context, candidate_key

        action = [4, 4, 3, 2]
        base = build_candidate_identity_context(
            action_space_version="current-code-v1",
            registry_hash="registry-a",
            max_sfs_hash="max-sfs-a",
            stage1_config_content_hash="stage1-content-a",
            stage1_gelu_degrees=[4],
            stage1_softmax_degrees=[2],
            profile="mrpc",
            dataset="mrpc",
            model="bert-base",
            rescale_optimizer_mode="in_process_real",
            rescale_optimizer_root="Rescale_optimizer",
            rescale_optimizer_canonical_hash="rescale-canon-a",
            decode_version="action_space_v1",
            metric_policy_version="mrpc-acc-f1-std-v1",
            threshold_policy_hash="threshold-a",
            fidelity="F0_optimizer_only",
        )
        changed_fidelity = dict(base)
        changed_fidelity["fidelity"] = "F1_small_probe"

        self.assertEqual(base["stage1_config_content_hash"], "stage1-content-a")
        self.assertEqual(base["stage1_gelu_degrees"], [4])
        self.assertEqual(base["stage1_softmax_degrees"], [2])
        self.assertEqual(base["rescale_optimizer_canonical_hash"], "rescale-canon-a")
        self.assertEqual(base["fidelity"], "F0_optimizer_only")
        self.assertNotEqual(candidate_key(action, base), candidate_key(action, changed_fidelity))

    def test_context_lookup_excludes_action_hash_only_legacy_records(self):
        from blb_stage2_rl.candidate_store import (
            CandidateStore,
            build_candidate_identity_context,
        )

        ctx = build_candidate_identity_context(
            action_space_version="current-code-v1",
            registry_hash="registry-a",
            max_sfs_hash="max-sfs-a",
            stage1_hash="stage1-a",
            stage1_degrees={"gelu": [4], "softmax": [2]},
            profile="mrpc",
            rescale_optimizer_mode="in_process_real",
            rescale_optimizer_root="Rescale_optimizer",
            rescale_optimizer_hash="rescale-a",
            decode_version="decode-v1",
            dataset="mrpc",
            model="bert-base",
            metric_policy_version="mrpc-acc-f1-std-v1",
            threshold_policy_hash="threshold-a",
        )
        action = [4, 4, 3, 2]

        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            store.append({"action_indices": action, "fidelity": "F1", "valid": True})
            self.assertTrue(store.read_all()[0]["legacy_record"])
            self.assertEqual(store.read_all()[0]["action_vector_hash"], store.read_all()[0]["action_hash"])
            self.assertIsNone(store.best_for_action(action, identity_context=ctx))

            store.append({
                "action_indices": action,
                "fidelity": "F1",
                "valid": True,
                "identity_context": ctx,
            })
            self.assertIsNotNone(store.best_for_action(action, identity_context=ctx))
            self.assertFalse(store.should_evaluate(action, "F1", identity_context=ctx))

    def test_context_lookup_with_legacy_fallback_reads_store_once(self):
        from blb_stage2_rl.candidate_store import (
            CandidateStore,
            action_hash,
            build_candidate_identity_context,
            candidate_key,
        )

        ctx = build_candidate_identity_context(
            action_space_version="current-code-v1",
            registry_hash="registry-a",
            max_sfs_hash="max-sfs-a",
            stage1_hash="stage1-a",
            stage1_degrees={"gelu": [4], "softmax": [2]},
            profile="mrpc",
            rescale_optimizer_mode="in_process_real",
            rescale_optimizer_root="Rescale_optimizer",
            rescale_optimizer_hash="rescale-a",
            decode_version="decode-v1",
            dataset="mrpc",
            model="bert-base",
            metric_policy_version="mrpc-acc-f1-std-v1",
            threshold_policy_hash="threshold-a",
        )
        action = [4, 4, 3, 2]
        contextual = {
            "candidate_key": candidate_key(action, ctx),
            "fidelity": "F1",
            "valid": True,
        }
        legacy = {
            "action_hash": action_hash(action),
            "legacy_record": True,
            "fidelity": "F0",
            "valid": True,
        }
        store = CandidateStore("unused.jsonl")
        store.read_all = mock.Mock(return_value=[contextual, legacy])

        best = store.best_for_action(action, identity_context=ctx, allow_legacy=True)

        self.assertIs(best, contextual)
        store.read_all.assert_called_once_with()

    def test_store_records_raw_and_effective_action_identity(self):
        from blb_stage2_rl.candidate_store import (
            CandidateStore,
            action_hash,
            build_candidate_identity_context,
        )

        ctx = build_candidate_identity_context(
            action_space_version="current-code-v1",
            registry_hash="registry-a",
            max_sfs_hash="max-sfs-a",
            stage1_hash="stage1-a",
            stage1_degrees={"gelu": [4], "softmax": [4]},
            profile="mrpc",
            rescale_optimizer_mode="in_process_real",
            rescale_optimizer_root="Rescale_optimizer",
            rescale_optimizer_hash="rescale-a",
            decode_version="decode-v1",
            dataset="mrpc",
            model="bert-base",
            metric_policy_version="mrpc-acc-f1-std-v1",
            threshold_policy_hash="threshold-a",
        )
        raw_action = [0, 4, 4]
        effective_action = [4, 4, 4]

        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            saved = store.append({
                "action_indices": raw_action,
                "effective_action_indices": effective_action,
                "fidelity": "F0",
                "valid": True,
                "identity_context": ctx,
            })

            self.assertEqual(saved["raw_action_hash"], action_hash(raw_action))
            self.assertEqual(saved["action_hash"], saved["raw_action_hash"])
            self.assertEqual(saved["effective_action_hash"], action_hash(effective_action))
            self.assertEqual(saved["candidate_key_basis"], "effective_action_hash + identity_context")

    def test_trial_groups_pool_raw_evidence_by_canonical_action_and_context(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [3, 2, 1, 0]
        context = {"action_space_version": "layerwise-v1", "profile": "mrpc"}
        first = TrialSeries(
            loss=[0.10, 0.11, 0.12, 0.13, 0.14],
            metric1=[0.90, 0.89, 0.91, 0.88, 0.92],
            metric2=[0.80, 0.79, 0.81, 0.78, 0.82],
            seeds=[10, 11, 12, 13, 14],
        )
        second = TrialSeries(
            loss=[0.15, 0.16, 0.17, 0.18, 0.19],
            metric1=[0.87, 0.86, 0.85, 0.84, 0.83],
            metric2=[0.77, 0.76, 0.75, 0.74, 0.73],
            seeds=[20, 21, 22, 23, 24],
        )

        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            store.append_trial_group(action, first, {
                "identity_context": context,
                "action_matrix": [[0, 0, 1, 2, 3, 4]],
                "variable_cost": 0.25,
                "group_index": 0,
            })
            store.append_trial_group(action, second, {
                "identity_context": context,
                "action_matrix": [[0, 5, 1, 2, 3, 4]],
                "variable_cost": 0.25,
                "group_index": 1,
                "promotion_marker": "fresh_top_up",
            })
            pooled = store.trial_evidence_for_action(action, context)
            isolated = store.trial_evidence_for_action(
                action, {**context, "profile": "rte"},
            )

        self.assertIsNotNone(pooled)
        self.assertEqual(pooled.trial_count, 10)
        self.assertEqual(pooled.trials.seeds, (10, 11, 12, 13, 14, 20, 21, 22, 23, 24))
        self.assertEqual(pooled.trials.loss, first.loss + second.loss)
        self.assertEqual(len(pooled.groups), 2)
        self.assertEqual(pooled.groups[1]["promotion_marker"], "fresh_top_up")
        self.assertIsNone(isolated)

    def test_trial_group_rejects_missing_and_duplicate_seeds_for_same_identity(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [3, 2, 1, 0]
        context = {"action_space_version": "layerwise-v1", "profile": "mrpc"}
        metadata = {"identity_context": context}

        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            with self.assertRaisesRegex(ValueError, "nonempty aligned seeds"):
                store.append_trial_group(
                    action,
                    TrialSeries(loss=[0.1], metric1=[0.9], metric2=[0.8]),
                    metadata,
                )
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1, 0.2], metric1=[0.9, 0.8], metric2=[0.8, 0.7],
                    seeds=[1, 2],
                ),
                metadata,
            )
            with self.assertRaisesRegex(ValueError, "duplicate trial seeds.*2"):
                store.append_trial_group(
                    action,
                    TrialSeries(
                        loss=[0.3, 0.4], metric1=[0.7, 0.6], metric2=[0.6, 0.5],
                        seeds=[2, 3],
                    ),
                    metadata,
                )

    def test_exact_trial_group_replay_is_idempotent_after_checkpoint_resume(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [3, 2, 1, 0]
        context = {"action_space_version": "layerwise-v1", "profile": "mrpc"}
        trials = TrialSeries(
            loss=[0.1, 0.2],
            metric1=[0.9, 0.8],
            metric2=[0.8, 0.7],
            seeds=[11, 12],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            first = store.append_trial_group(
                action,
                trials,
                {"identity_context": context, "episode_index": 121},
            )
            committed_size = path.stat().st_size
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3, 0.4],
                    metric1=[0.7, 0.6],
                    metric2=[0.6, 0.5],
                    seeds=[21, 22],
                ),
                {"identity_context": context, "episode_index": 122},
            )
            store.recover_to_checkpoint_size(committed_size)
            store = CandidateStore(path)
            size_before_replay = path.stat().st_size
            replay = store.append_trial_group(
                action,
                trials,
                {"identity_context": context, "episode_index": 121},
            )
            evidence = store.trial_evidence_for_action(action, context)
            size_after_replay = path.stat().st_size

        self.assertEqual(first["candidate_key"], replay["candidate_key"])
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(size_after_replay, size_before_replay)
        self.assertEqual(evidence.trial_count, 2)
        self.assertEqual(evidence.trials.seeds, (11, 12))

    def test_legacy_out_of_range_loss_replay_is_normalized_and_idempotent(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [3, 2, 1, 0]
        context = {"action_space_version": "layerwise-v1", "profile": "mrpc"}
        trials = TrialSeries(
            loss=[150.0],
            metric1=[0.9],
            metric2=[0.8],
            seeds=[11],
        )
        metadata = {"identity_context": context, "episode_index": 121}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            CandidateStore(path).append_trial_group(action, trials, metadata)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            trial_row = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v1"
            )
            trial_row["trial_group"]["loss"] = [150.0]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            size_before_replay = path.stat().st_size
            replay = CandidateStore(path).append_trial_group(
                action, trials, metadata,
            )
            evidence = CandidateStore(path).trial_evidence_for_action(
                action, context,
            )
            size_after_replay = path.stat().st_size

        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(size_after_replay, size_before_replay)
        self.assertEqual(evidence.trials.loss, (100.0,))

    def test_trial_group_replay_rejects_changed_metadata(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [3, 2, 1, 0]
        context = {"action_space_version": "layerwise-v1", "profile": "mrpc"}
        trials = TrialSeries(
            loss=[0.1, 0.2],
            metric1=[0.9, 0.8],
            metric2=[0.8, 0.7],
            seeds=[11, 12],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action,
                trials,
                {
                    "identity_context": context,
                    "episode_index": 121,
                    "variable_cost": 0.4,
                },
            )
            store = CandidateStore(path)

            with self.assertRaisesRegex(ValueError, "metadata"):
                store.append_trial_group(
                    action,
                    trials,
                    {
                        "identity_context": context,
                        "episode_index": 122,
                        "variable_cost": 0.4,
                    },
                )

    def test_trial_group_replay_rejects_changed_values_after_reopen(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [3, 2, 1, 0]
        context = {"action_space_version": "layerwise-v1", "profile": "mrpc"}
        metadata = {"identity_context": context, "episode_index": 121}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            CandidateStore(path).append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1, 0.2], metric1=[0.9, 0.8], metric2=[0.8, 0.7],
                    seeds=[11, 12],
                ),
                metadata,
            )

            with self.assertRaisesRegex(ValueError, "duplicate trial seeds"):
                CandidateStore(path).append_trial_group(
                    action,
                    TrialSeries(
                        loss=[0.1, 9.9],
                        metric1=[0.9, 0.8],
                        metric2=[0.8, 0.7],
                        seeds=[11, 12],
                    ),
                    metadata,
                )

    def test_legacy_f1_replay_uses_compact_metadata_normalization_strictly(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        fixture = self._representative_compact_fixture()
        action = fixture["action"]
        trials = fixture["trials"]["F1"]
        legacy_metadata = fixture["metadata"]["F1"]
        compact_metadata = dict(legacy_metadata)
        compact_metadata.pop("boosted_overrides")

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(action, trials, legacy_metadata)
            size_before_replay = path.stat().st_size

            replay = store.append_trial_group(
                action, trials, compact_metadata, compact=True,
            )
            self.assertTrue(replay["idempotent_replay"])
            self.assertEqual(path.stat().st_size, size_before_replay)

            changed_metadata = (
                {
                    **compact_metadata,
                    "boosted_overrides_hash": "changed-overrides-hash",
                },
                {
                    **compact_metadata,
                    "boosted_overrides_provenance": "changed-provenance",
                },
                {**compact_metadata, "episode_index": 999},
            )
            for metadata in changed_metadata:
                with self.subTest(metadata=metadata):
                    with self.assertRaisesRegex(ValueError, "metadata"):
                        store.append_trial_group(
                            action, trials, metadata, compact=True,
                        )

            changed_trials = TrialSeries(
                loss=[9.9, *trials.loss[1:]],
                metric1=trials.metric1,
                metric2=trials.metric2,
                seeds=trials.seeds,
            )
            with self.assertRaisesRegex(ValueError, "duplicate trial seeds"):
                store.append_trial_group(
                    action, changed_trials, compact_metadata, compact=True,
                )

            compact_first_path = Path(td) / "compact_first.jsonl"
            compact_first = CandidateStore(compact_first_path)
            compact_first.append_trial_group(
                action, trials, compact_metadata, compact=True,
            )
            compact_size = compact_first_path.stat().st_size
            legacy_replay = compact_first.append_trial_group(
                action, trials, legacy_metadata,
            )
            self.assertTrue(legacy_replay["idempotent_replay"])
            self.assertEqual(compact_first_path.stat().st_size, compact_size)

    def test_trial_index_streams_and_keeps_only_offsets_and_seed_sets(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            writer = CandidateStore(path)
            for group_idx in range(40):
                start = group_idx * 5
                values = [float(start + offset) for offset in range(5)]
                writer.append_trial_group(
                    action,
                    TrialSeries(
                        loss=values,
                        metric1=[100.0 - value for value in values],
                        metric2=[200.0 - value for value in values],
                        seeds=list(range(start + 1, start + 6)),
                    ),
                    {
                        "identity_context": context,
                        "group_index": group_idx,
                    },
                )

            store = CandidateStore(path)
            with mock.patch.object(
                store,
                "read_all",
                side_effect=AssertionError("trial index initialization must stream"),
            ) as read_all:
                total_trial_count = store.trial_count_for_action(action, context)
                with mock.patch.object(
                    store,
                    "_decode_jsonl_row",
                    wraps=store._decode_jsonl_row,
                ) as decoded_rows:
                    bounded = store.trial_evidence_for_action(
                        action, context, max_trials=25,
                    )
                full = store.trial_evidence_for_action(action, context)

            key = candidate_key(action, context)
            offsets = store._trial_offsets_by_candidate_key[key]
            seed_index = store._trial_seeds_by_candidate_key[key]

        self.assertEqual(bounded.trial_count, 25)
        self.assertEqual(total_trial_count, 200)
        self.assertEqual(decoded_rows.call_count, 5)
        self.assertEqual(bounded.trials.loss, tuple(float(value) for value in range(25)))
        self.assertEqual(len(bounded.groups), 5)
        self.assertEqual(full.trial_count, 200)
        read_all.assert_not_called()
        self.assertEqual(len(offsets), 40)
        self.assertTrue(all(type(offset) is int for offset in offsets))
        self.assertIsInstance(seed_index, set)
        self.assertEqual(len(seed_index), 200)
        self.assertNotIn("_trial_records_by_candidate_key", store.__dict__)
        self.assertNotIn("_bounded_evidence_cache", store.__dict__)

    def test_checkpoint_recovery_preserves_complete_rows_and_hides_orphan_tail(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        metadata = {"identity_context": context}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1, 0.2], metric1=[0.9, 0.8], metric2=[0.8, 0.7],
                    seeds=[1, 2],
                ),
                metadata,
            )
            committed_size = path.stat().st_size
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3, 0.4], metric1=[0.7, 0.6], metric2=[0.6, 0.5],
                    seeds=[3, 4],
                ),
                metadata,
            )
            before_recovery = path.read_bytes()
            with path.open("ab") as handle:
                handle.write(b'{"record_type":"candidate_trial_group_v1"')

            store.recover_to_checkpoint_size(committed_size)
            after_recovery = path.read_bytes()
            evidence = store.trial_evidence_for_action(action, context)
            logical_records = store.read_all()

            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3, 0.4], metric1=[0.7, 0.6], metric2=[0.6, 0.5],
                    seeds=[3, 4],
                ),
                metadata,
            )
            evidence_after_replacement = store.trial_evidence_for_action(
                action, context,
            )

        self.assertTrue(after_recovery.startswith(before_recovery))
        self.assertNotIn(b'{"record_type":"candidate_trial_group_v1"', after_recovery)
        physical_rows = [json.loads(line) for line in after_recovery.splitlines()]
        self.assertEqual(len(physical_rows), 3)
        self.assertEqual(
            physical_rows[-1]["record_type"],
            "candidate_store_recovery_v1",
        )
        self.assertEqual(physical_rows[-1]["checkpoint_size"], committed_size)
        self.assertEqual(physical_rows[-1]["logical_generation"], 1)
        self.assertEqual(evidence.trial_count, 2)
        self.assertEqual(evidence.trials.seeds, (1, 2))
        self.assertEqual(len(logical_records), 1)
        self.assertEqual(evidence_after_replacement.trials.seeds, (1, 2, 3, 4))

    def test_repeated_checkpoint_recovery_never_revives_old_orphan_rows(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        metadata = {"identity_context": context}

        def one_trial(seed, value):
            return TrialSeries(
                loss=[value],
                metric1=[1.0 - value],
                metric2=[2.0 - value],
                seeds=[seed],
            )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(action, one_trial(1, 0.1), metadata)
            first_checkpoint_size = path.stat().st_size
            store.append_trial_group(action, one_trial(2, 0.2), metadata)

            store.recover_to_checkpoint_size(first_checkpoint_size)
            store.append_trial_group(action, one_trial(3, 0.3), metadata)
            second_checkpoint_size = path.stat().st_size
            store.append_trial_group(action, one_trial(4, 0.4), metadata)

            CandidateStore(path).recover_to_checkpoint_size(second_checkpoint_size)
            CandidateStore(path).recover_to_checkpoint_size(second_checkpoint_size)
            resumed = CandidateStore(path)
            evidence = resumed.trial_evidence_for_action(action, context)
            physical_rows = [
                json.loads(line) for line in path.read_bytes().splitlines()
            ]
            physical_trial_seeds = [
                tuple(row["trial_group"]["seeds"])
                for row in physical_rows
                if row.get("record_type") == "candidate_trial_group_v1"
            ]
            recovery_generations = [
                row["logical_generation"]
                for row in physical_rows
                if row.get("record_type") == "candidate_store_recovery_v1"
            ]

            resumed.append_trial_group(action, one_trial(2, 0.2), metadata)
            evidence_after_reusing_orphan_seed = resumed.trial_evidence_for_action(
                action, context,
            )

        self.assertEqual(evidence.trials.seeds, (1, 3))
        self.assertIn((2,), physical_trial_seeds)
        self.assertIn((4,), physical_trial_seeds)
        self.assertEqual(recovery_generations, [1, 2, 3])
        self.assertEqual(evidence_after_reusing_orphan_seed.trials.seeds, (1, 3, 2))

    def test_warm_trial_index_refreshes_after_external_checkpoint_recovery(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        metadata = {"identity_context": context}

        def one_trial(seed, value):
            return TrialSeries(
                loss=[value], metric1=[1.0 - value], metric2=[2.0 - value],
                seeds=[seed],
            )

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            warm = CandidateStore(path)
            warm.append_trial_group(action, one_trial(1, 0.1), metadata)
            committed_size = path.stat().st_size
            warm.append_trial_group(action, one_trial(2, 0.2), metadata)
            self.assertEqual(warm.trial_count_for_action(action, context), 2)

            CandidateStore(path).recover_to_checkpoint_size(committed_size)

            self.assertEqual(warm.trial_count_for_action(action, context), 1)
            self.assertEqual(
                warm.trial_evidence_for_action(action, context).trials.seeds,
                (1,),
            )

    def test_trial_evidence_index_avoids_read_all_and_tracks_new_appends(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        metadata = {"identity_context": context}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1, 0.2], metric1=[0.9, 0.8], metric2=[0.8, 0.7],
                    seeds=[1, 2],
                ),
                metadata,
            )
            store = CandidateStore(path)
            with mock.patch.object(
                store,
                "read_all",
                side_effect=AssertionError("trial index initialization must stream"),
            ) as read_all:
                first = store.trial_evidence_for_action(action, context)
                store.append_trial_group(
                    action,
                    TrialSeries(
                        loss=[0.3, 0.4], metric1=[0.7, 0.6], metric2=[0.6, 0.5],
                        seeds=[3, 4],
                    ),
                    metadata,
                )
                second = store.trial_evidence_for_action(action, context)

        self.assertEqual(first.trial_count, 2)
        self.assertEqual(second.trial_count, 4)
        read_all.assert_not_called()

    def test_repeated_appends_do_not_rescan_the_recovery_layout(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        metadata = {"identity_context": context}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1], metric1=[0.9], metric2=[0.8], seeds=[1],
                ),
                metadata,
            )
            original_open = Path.open
            recovery_layout_scans = 0

            def counting_open(open_path, *args, **kwargs):
                nonlocal recovery_layout_scans
                mode = args[0] if args else kwargs.get("mode", "r")
                if Path(open_path) == path and mode == "rb":
                    recovery_layout_scans += 1
                return original_open(open_path, *args, **kwargs)

            with mock.patch.object(Path, "open", counting_open):
                for seed in range(2, 22):
                    store.append_trial_group(
                        action,
                        TrialSeries(
                            loss=[seed / 100.0],
                            metric1=[0.9],
                            metric2=[0.8],
                            seeds=[seed],
                        ),
                        metadata,
                    )

        self.assertEqual(recovery_layout_scans, 0)

    def test_fresh_append_checks_only_new_seeds_without_scanning_full_seed_index(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.statistical_constraints import TrialSeries

        class NoIterationSet(set):
            def __iter__(self):
                raise AssertionError("fresh append must not scan all historical seeds")

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1"}
        metadata = {"identity_context": context}
        with tempfile.TemporaryDirectory() as td:
            store = CandidateStore(Path(td) / "candidate_store.jsonl")
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1, 0.2], metric1=[0.9, 0.8], metric2=[0.8, 0.7],
                    seeds=[1, 2],
                ),
                metadata,
            )
            key = candidate_key(action, context)
            store._trial_seeds_by_candidate_key[key] = NoIterationSet(
                store._trial_seeds_by_candidate_key[key]
            )

            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3, 0.4], metric1=[0.7, 0.6], metric2=[0.6, 0.5],
                    seeds=[3, 4],
                ),
                metadata,
            )

        self.assertEqual(store.trial_count_for_action(action, context), 4)

    def test_unlinked_compact_store_keeps_read_cache_but_rebuilds_before_write(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1", "fidelity": "F1"}
        metadata = {"identity_context": context, "fidelity": "F1"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1], metric1=[0.9], metric2=[0.8], seeds=[1],
                ),
                metadata,
                compact=True,
            )
            path.unlink()

            self.assertEqual(store.trial_count_for_action(action, context), 1)

            store.append_trial_group(
                action,
                TrialSeries(
                    loss=[0.2], metric1=[0.8], metric2=[0.7], seeds=[2],
                ),
                metadata,
                compact=True,
            )
            records = store.read_all()
            evidence = store.trial_evidence_for_action(action, context)
            physical_rows = [
                json.loads(line) for line in path.read_bytes().splitlines()
            ]

        self.assertEqual(len(records), 1)
        self.assertEqual(evidence.trials.seeds, (2,))
        self.assertEqual(
            [row["record_type"] for row in physical_rows],
            ["candidate_identity_context_v1", "candidate_trial_group_v2"],
        )

    def test_compact_index_append_and_evidence_avoid_full_record_hydration(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [idx % 6 for idx in range(73 * 12 + 1)]
        context = {"action_space_version": "layerwise-v1", "fidelity": "F1"}
        metadata = {"identity_context": context, "fidelity": "F1"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            CandidateStore(path).append_trial_group(
                action,
                TrialSeries(
                    loss=[0.1], metric1=[0.9], metric2=[0.8], seeds=[1],
                ),
                metadata,
                compact=True,
            )
            store = CandidateStore(path)
            with mock.patch.object(
                store,
                "_hydrate_compact_record",
                side_effect=AssertionError("hot path must not fully hydrate rows"),
            ) as hydrate:
                first = store.trial_evidence_for_action(action, context)
                store.append_trial_group(
                    action,
                    TrialSeries(
                        loss=[0.2], metric1=[0.8], metric2=[0.7], seeds=[2],
                    ),
                    metadata,
                    compact=True,
                )
                second = store.trial_evidence_for_action(action, context)

        self.assertEqual(first.trials.seeds, (1,))
        self.assertEqual(second.trials.seeds, (1, 2))
        hydrate.assert_not_called()

    def test_compact_and_legacy_candidate_store_have_identical_logical_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key

        fixture = self._representative_compact_fixture()
        with tempfile.TemporaryDirectory() as td:
            legacy = CandidateStore(Path(td) / "legacy.jsonl")
            compact = CandidateStore(Path(td) / "compact.jsonl")
            self._populate_equivalent_store(legacy, fixture, compact=False)
            self._populate_equivalent_store(compact, fixture, compact=True)

            for fidelity in ("F1", "F4"):
                context = fixture["contexts"][fidelity]
                legacy_evidence = legacy.trial_evidence_for_action(
                    fixture["action"], context,
                )
                compact_evidence = compact.trial_evidence_for_action(
                    fixture["action"], context,
                )
                self.assertEqual(
                    self._logical_evidence_snapshot(compact_evidence, fidelity),
                    self._logical_evidence_snapshot(legacy_evidence, fidelity),
                )
                self.assertEqual(
                    compact_evidence.candidate_key,
                    candidate_key(fixture["action"], context),
                )
                compact_best = compact.best_for_action(
                    fixture["action"], identity_context=context,
                )
                legacy_best = legacy.best_for_action(
                    fixture["action"], identity_context=context,
                )
                for field in (
                    "candidate_key", "action_indices", "raw_action_indices",
                    "effective_action_indices", "identity_context",
                    "identity_context_hash", "fidelity", "valid",
                ):
                    self.assertEqual(compact_best[field], legacy_best[field])
                self.assertEqual(
                    compact.should_evaluate(
                        fixture["action"], fidelity, identity_context=context,
                    ),
                    legacy.should_evaluate(
                        fixture["action"], fidelity, identity_context=context,
                    ),
                )
                if fidelity == "F1":
                    self.assertTrue(compact.should_evaluate(
                        fixture["action"], "F4", identity_context=context,
                    ))

            f4_evidence = compact.trial_evidence_for_action(
                fixture["action"], fixture["contexts"]["F4"],
            )
            strict_inputs = f4_evidence.groups[-1]
            self.assertEqual(strict_inputs["action_matrix"], fixture["metadata"]["F4"]["action_matrix"])
            self.assertEqual(strict_inputs["boosted_overrides"], fixture["boosted_overrides"])
            self.assertEqual(strict_inputs["assessment_bootstrap_seed"], 77)
            self.assertTrue(f4_evidence.promotion_attempted)
            self.assertTrue(f4_evidence.promoted)

    def test_compact_candidate_store_physical_rows_intern_context_and_omit_derivable_f1_data(self):
        from blb_stage2_rl.candidate_store import CandidateStore, sha256_json

        fixture = self._representative_compact_fixture()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "compact.jsonl"
            store = CandidateStore(path)
            self._populate_equivalent_store(store, fixture, compact=True)
            store.append_promotion_status(
                fixture["action"],
                fixture["contexts"]["F4"],
                status="final_revalidation_passed",
                metadata={
                    **fixture["status_metadata"],
                    "final_revalidation_trial_count": 31,
                },
            )
            physical_rows = [json.loads(line) for line in path.read_bytes().splitlines()]

        context_rows = [
            row for row in physical_rows
            if row.get("record_type") == "candidate_identity_context_v1"
        ]
        self.assertEqual(len(context_rows), 2)
        self.assertEqual(
            {row["identity_context_hash"] for row in context_rows},
            {sha256_json(context) for context in fixture["contexts"].values()},
        )
        for row in context_rows:
            self.assertEqual(
                row["identity_context_hash"], sha256_json(row["identity_context"]),
            )

        trial_rows = [
            row for row in physical_rows
            if row.get("record_type") == "candidate_trial_group_v2"
        ]
        self.assertEqual(len(trial_rows), 2)
        for row in trial_rows:
            self.assertEqual(
                {
                    name for name in (
                        "action_indices", "raw_action_indices",
                        "effective_action_indices",
                    )
                    if name in row
                },
                {"action_indices"},
            )
            for field in (
                "candidate_key", "action_hash", "raw_action_hash",
                "effective_action_hash", "identity_context_hash", "trial_group",
            ):
                self.assertIn(field, row)
            self.assertNotIn("identity_context", row)
            self.assertNotIn(
                "identity_context", row["trial_group_metadata"],
            )

        f1_row = next(row for row in trial_rows if row["fidelity"] == "F1")
        f4_row = next(row for row in trial_rows if row["fidelity"] == "F4")
        self.assertNotIn("boosted_overrides", f1_row["trial_group_metadata"])
        self.assertEqual(
            f1_row["trial_group_metadata"]["boosted_overrides_hash"],
            fixture["metadata"]["F1"]["boosted_overrides_hash"],
        )
        self.assertEqual(
            f1_row["trial_group_metadata"]["boosted_overrides_provenance"],
            "layerwise_env",
        )
        self.assertEqual(
            f4_row["trial_group_metadata"]["boosted_overrides"],
            fixture["boosted_overrides"],
        )

        status_rows = [
            row for row in physical_rows
            if row.get("record_type") == "candidate_promotion_status_v2"
        ]
        self.assertEqual(len(status_rows), 2)
        for row in status_rows:
            self.assertIn("action_indices", row)
            self.assertIn("identity_context_hash", row)
            self.assertNotIn("raw_action_indices", row)
            self.assertNotIn("effective_action_indices", row)
            self.assertNotIn("identity_context", row)
            self.assertEqual(
                row["promotion_metadata"]["boosted_overrides"],
                fixture["boosted_overrides"],
            )

    def test_compact_f1_candidate_store_row_is_less_than_half_legacy_bytes(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        fixture = self._representative_compact_fixture()
        with tempfile.TemporaryDirectory() as td:
            legacy_path = Path(td) / "legacy.jsonl"
            compact_path = Path(td) / "compact.jsonl"
            CandidateStore(legacy_path).append_trial_group(
                fixture["action"], fixture["trials"]["F1"], fixture["metadata"]["F1"],
            )
            CandidateStore(compact_path).append_trial_group(
                fixture["action"], fixture["trials"]["F1"], fixture["metadata"]["F1"],
                compact=True,
            )
            legacy_line = next(
                line for line in legacy_path.read_bytes().splitlines(keepends=True)
                if json.loads(line)["record_type"] == "candidate_trial_group_v1"
            )
            compact_line = next(
                line for line in compact_path.read_bytes().splitlines(keepends=True)
                if json.loads(line)["record_type"] == "candidate_trial_group_v2"
            )

        self.assertLess(len(compact_line), len(legacy_line) / 2)

    def test_compact_candidate_store_hydrates_copies_and_fails_closed_on_bad_context(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        fixture = self._representative_compact_fixture()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "compact.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                fixture["action"], fixture["trials"]["F1"], fixture["metadata"]["F1"],
                compact=True,
            )
            physical_rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            first = store.read_all()[0]
            self.assertEqual(
                [row["record_type"] for row in store.iter_active_records()],
                ["candidate_trial_group_v2"],
            )
            self.assertEqual(first["identity_context"], fixture["contexts"]["F1"])
            self.assertEqual(first["raw_action_indices"], fixture["action"])
            self.assertEqual(first["effective_action_indices"], fixture["action"])
            self.assertEqual(
                first["trial_group_metadata"]["identity_context"],
                fixture["contexts"]["F1"],
            )

            first["action_indices"][0] = 999
            first["identity_context"]["profile"] = "mutated"
            first["trial_group_metadata"]["action_matrix"][0][0] = 999
            second = store.read_all()[0]
            self.assertEqual(second["action_indices"], fixture["action"])
            self.assertEqual(second["identity_context"]["profile"], "mrpc")
            self.assertEqual(
                second["trial_group_metadata"]["action_matrix"][0][0],
                fixture["metadata"]["F1"]["action_matrix"][0][0],
            )

            cases = {}
            cases["missing"] = [
                row for row in physical_rows
                if row.get("record_type") != "candidate_identity_context_v1"
            ]
            mismatched = [dict(row) for row in physical_rows]
            context_row = next(
                row for row in mismatched
                if row.get("record_type") == "candidate_identity_context_v1"
            )
            context_row["identity_context"] = {
                **context_row["identity_context"], "profile": "tampered",
            }
            cases["hash_content_mismatch"] = mismatched
            for label, rows in cases.items():
                bad_path = Path(td) / f"bad-{label}.jsonl"
                bad_path.write_text(
                    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8",
                )
                with self.subTest(label=label):
                    with self.assertRaises(ValueError):
                        CandidateStore(bad_path).read_all()

    def test_legacy_candidate_store_rejects_cross_candidate_identity_splice(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        fixture = self._representative_compact_fixture()
        action_a = list(fixture["action"])
        action_b = list(action_a)
        action_b[0] = (action_b[0] + 1) % 6
        context = fixture["contexts"]["F1"]
        trials_b = TrialSeries(
            loss=[0.41, 0.42],
            metric1=[0.71, 0.72],
            metric2=[0.61, 0.62],
            seeds=[901, 902],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action_a,
                fixture["trials"]["F1"],
                fixture["metadata"]["F1"],
            )
            store.append_trial_group(
                action_b,
                trials_b,
                fixture["metadata"]["F1"],
            )
            rows = [json.loads(line) for line in path.read_text(
                encoding="utf-8",
            ).splitlines()]
            trial_a = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v1"
                and row["action_indices"] == action_a
            )
            trial_b = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v1"
                and row["action_indices"] == action_b
            )
            trial_a["candidate_key"] = trial_b["candidate_key"]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "candidate identity"):
                CandidateStore(path).trial_evidence_for_action(action_b, context)

    def test_legacy_candidate_store_rejects_self_consistent_effective_splice(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        fixture = self._representative_compact_fixture()
        action_a = list(fixture["action"])
        action_b = list(action_a)
        action_b[0] = (action_b[0] + 1) % 6
        context = fixture["contexts"]["F1"]
        trials_b = TrialSeries(
            loss=[0.41, 0.42],
            metric1=[0.71, 0.72],
            metric2=[0.61, 0.62],
            seeds=[901, 902],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action_a,
                fixture["trials"]["F1"],
                fixture["metadata"]["F1"],
            )
            store.append_trial_group(
                action_b,
                trials_b,
                fixture["metadata"]["F1"],
            )
            rows = [json.loads(line) for line in path.read_text(
                encoding="utf-8",
            ).splitlines()]
            trial_a = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v1"
                and row["action_indices"] == action_a
            )
            trial_b = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v1"
                and row["action_indices"] == action_b
            )
            for field in (
                    "effective_action_indices",
                    "effective_action_hash",
                    "candidate_key",
            ):
                trial_a[field] = copy.deepcopy(trial_b[field])
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "candidate identity"):
                CandidateStore(path).trial_evidence_for_action(action_b, context)

    def test_legacy_best_lookup_validates_persisted_candidate_key(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key

        fixture = self._representative_compact_fixture()
        action_a = list(fixture["action"])
        action_b = list(action_a)
        action_b[0] = (action_b[0] + 1) % 6
        context = fixture["contexts"]["F1"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.jsonl"
            CandidateStore(path).append_trial_group(
                action_a,
                fixture["trials"]["F1"],
                fixture["metadata"]["F1"],
            )
            rows = [json.loads(line) for line in path.read_text(
                encoding="utf-8",
            ).splitlines()]
            trial = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v1"
            )
            trial["candidate_key"] = candidate_key(action_b, context)
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            operations = {
                "read_all": lambda store: store.read_all(),
                "best_for_action": lambda store: store.best_for_action(
                    action_b, identity_context=context,
                ),
                "should_evaluate": lambda store: store.should_evaluate(
                    action_b, "F1", identity_context=context,
                ),
            }
            for name, operation in operations.items():
                with self.subTest(operation=name):
                    with self.assertRaisesRegex(ValueError, "candidate identity"):
                        operation(CandidateStore(path))

    def test_compact_candidate_store_rejects_cross_candidate_identity_splice(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        fixture = self._representative_compact_fixture()
        action_a = list(fixture["action"])
        action_b = list(action_a)
        action_b[0] = (action_b[0] + 1) % 6
        context = fixture["contexts"]["F1"]
        trials_b = TrialSeries(
            loss=[0.41, 0.42],
            metric1=[0.71, 0.72],
            metric2=[0.61, 0.62],
            seeds=[901, 902],
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "compact.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action_a,
                fixture["trials"]["F1"],
                fixture["metadata"]["F1"],
                compact=True,
            )
            store.append_trial_group(
                action_b,
                trials_b,
                fixture["metadata"]["F1"],
                compact=True,
            )
            rows = [json.loads(line) for line in path.read_text(
                encoding="utf-8",
            ).splitlines()]
            trial_a = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v2"
                and row["action_indices"] == action_a
            )
            trial_b = next(
                row for row in rows
                if row.get("record_type") == "candidate_trial_group_v2"
                and row["action_indices"] == action_b
            )
            trial_a["candidate_key"] = trial_b["candidate_key"]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "candidate identity"):
                CandidateStore(path).trial_evidence_for_action(action_b, context)

    def test_append_rejects_conflicting_derived_candidate_identity_fields(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        fixture = self._representative_compact_fixture()
        action = fixture["action"]
        context = fixture["contexts"]["F1"]
        conflicting_action = list(action)
        conflicting_action[0] = (conflicting_action[0] + 1) % 6
        conflicts = {
            "effective_action_indices": conflicting_action,
            "raw_action_hash": "0" * 64,
            "action_hash": "1" * 64,
            "action_vector_hash": "2" * 64,
            "effective_action_hash": "3" * 64,
            "candidate_key_basis": "caller_supplied_key",
            "candidate_key": "4" * 64,
            "identity_context_hash": "5" * 64,
            "legacy_record": True,
        }
        with tempfile.TemporaryDirectory() as td:
            for field, value in conflicts.items():
                path = Path(td) / f"bad-{field}.jsonl"
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "candidate identity"):
                        CandidateStore(path).append({
                            "record_type": "candidate_trial_group_v1",
                            "action_indices": action,
                            "effective_action_indices": action,
                            "identity_context": context,
                            "fidelity": "F1",
                            "valid": True,
                            field: value,
                        })

    def test_mixed_store_random_offsets_pool_in_order_and_share_v1_duplicate_rules(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key
        from blb_stage2_rl.statistical_constraints import TrialSeries

        fixture = self._representative_compact_fixture()
        action = fixture["action"]
        context = fixture["contexts"]["F1"]
        second_trials = TrialSeries(
            loss=[0.401, 0.402], metric1=[0.701, 0.702], metric2=[0.601, 0.602],
            seeds=[301, 302],
        )
        second_metadata = {
            **fixture["metadata"]["F1"],
            "episode_index": 121,
        }
        with tempfile.TemporaryDirectory() as td:
            mixed = CandidateStore(Path(td) / "mixed.jsonl")
            legacy = CandidateStore(Path(td) / "legacy.jsonl")
            for store in (mixed, legacy):
                store.append_trial_group(
                    action, fixture["trials"]["F1"], fixture["metadata"]["F1"],
                )
            mixed.append_trial_group(action, second_trials, second_metadata, compact=True)
            legacy.append_trial_group(action, second_trials, second_metadata)

            mixed = CandidateStore(mixed.path)
            mixed_evidence = mixed.trial_evidence_for_action(action, context)
            legacy_evidence = legacy.trial_evidence_for_action(action, context)
            key = candidate_key(action, context)
            offset_records = list(mixed._trial_records_for_candidate_key(key))

            duplicate = TrialSeries(
                loss=[0.5, 0.6], metric1=[0.5, 0.4], metric2=[0.4, 0.3],
                seeds=[302, 303],
            )
            with self.assertRaises(ValueError) as mixed_error:
                mixed.append_trial_group(action, duplicate, second_metadata, compact=True)
            with self.assertRaises(ValueError) as legacy_error:
                legacy.append_trial_group(action, duplicate, second_metadata)

            f4_context = fixture["contexts"]["F4"]
            for store in (mixed, legacy):
                store.append_trial_group(
                    action, fixture["trials"]["F4"], fixture["metadata"]["F4"],
                )
                store.append({
                    "record_type": "candidate_promotion_status_v1",
                    "action_indices": action,
                    "identity_context": f4_context,
                    "promotion_status": "promoted",
                    "fidelity": "F4",
                    "valid": True,
                })
            mixed.append_promotion_status(
                action, f4_context, status="failed_probability_gate", metadata={},
            )
            legacy.append({
                "record_type": "candidate_promotion_status_v1",
                "action_indices": action,
                "identity_context": f4_context,
                "promotion_status": "failed_probability_gate",
                "fidelity": "F4",
                "valid": False,
            })

            mixed_f4 = mixed.trial_evidence_for_action(action, f4_context)
            legacy_f4 = legacy.trial_evidence_for_action(action, f4_context)
            active_types = {row["record_type"] for row in mixed.read_all()}

        self.assertEqual(
            self._logical_evidence_snapshot(mixed_evidence, "F1"),
            self._logical_evidence_snapshot(legacy_evidence, "F1"),
        )
        self.assertEqual(
            mixed_evidence.trials.seeds,
            fixture["trials"]["F1"].seeds + second_trials.seeds,
        )
        self.assertEqual(
            [row["record_type"] for row in offset_records],
            ["candidate_trial_group_v1", "candidate_trial_group_v2"],
        )
        self.assertTrue(all(row["identity_context"] == context for row in offset_records))
        self.assertEqual(str(mixed_error.exception), str(legacy_error.exception))
        self.assertEqual(mixed_f4.promotion_attempted, legacy_f4.promotion_attempted)
        self.assertEqual(mixed_f4.promotion_status, legacy_f4.promotion_status)
        self.assertEqual(
            active_types,
            {
                "candidate_trial_group_v1", "candidate_trial_group_v2",
                "candidate_promotion_status_v1", "candidate_promotion_status_v2",
            },
        )

    def test_mixed_store_recovery_preserves_compact_replay_fingerprints_and_offsets(self):
        from blb_stage2_rl.candidate_store import CandidateStore, candidate_key, sha256_json
        from blb_stage2_rl.layerwise_runner import checkpoint_file_fingerprints
        from blb_stage2_rl.statistical_constraints import TrialSeries

        fixture = self._representative_compact_fixture()
        action = fixture["action"]
        context = fixture["contexts"]["F1"]
        compact_trials = TrialSeries(
            loss=[0.41], metric1=[0.71], metric2=[0.61], seeds=[301],
        )
        compact_metadata = {
            **fixture["metadata"]["F1"], "episode_index": 121,
        }
        orphan_context = {**context, "profile": "orphan-context"}
        orphan_metadata = {
            **compact_metadata,
            "identity_context": orphan_context,
            "episode_index": 999,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mixed.jsonl"
            store = CandidateStore(path)
            store.append_trial_group(
                action, fixture["trials"]["F1"], fixture["metadata"]["F1"],
            )
            store.append_trial_group(
                action, compact_trials, compact_metadata, compact=True,
            )
            store.append_promotion_status(
                action, fixture["contexts"]["F4"], status="promoted",
                metadata=fixture["status_metadata"],
            )
            committed_size = path.stat().st_size
            expected_fingerprint = checkpoint_file_fingerprints({
                "candidate_store": (path, committed_size),
            })
            logical_before = store.read_all()
            evidence_before = store.trial_evidence_for_action(action, context)
            base_key = candidate_key(action, context)
            offsets_before = tuple(store._trial_offsets_by_candidate_key[base_key])

            store.append_trial_group(
                action,
                TrialSeries(loss=[0.9], metric1=[0.1], metric2=[0.1], seeds=[999]),
                orphan_metadata,
                compact=True,
            )
            store.recover_to_checkpoint_size(committed_size)
            resumed = CandidateStore(path)
            self.assertIsNone(
                resumed.trial_evidence_for_action(action, orphan_context),
            )
            logical_after = resumed.read_all()
            evidence_after = resumed.trial_evidence_for_action(action, context)
            offsets_after = tuple(resumed._trial_offsets_by_candidate_key[base_key])
            replay_size = path.stat().st_size
            replay = resumed.append_trial_group(
                action, compact_trials, compact_metadata, compact=True,
            )
            size_after_replay = path.stat().st_size

            resumed.append_trial_group(
                action,
                TrialSeries(loss=[0.8], metric1=[0.2], metric2=[0.2], seeds=[1000]),
                orphan_metadata,
                compact=True,
            )
            replacement_checkpoint = path.stat().st_size
            CandidateStore(path).recover_to_checkpoint_size(replacement_checkpoint)
            CandidateStore(path).recover_to_checkpoint_size(replacement_checkpoint)
            final_store = CandidateStore(path)
            final_base = final_store.trial_evidence_for_action(action, context)
            replacement = final_store.trial_evidence_for_action(action, orphan_context)
            final_records = final_store.read_all()
            final_fingerprint = checkpoint_file_fingerprints({
                "candidate_store": (path, committed_size),
            })
            physical_rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            orphan_hash = sha256_json(orphan_context)
            physical_orphan_contexts = [
                row for row in physical_rows
                if row.get("record_type") == "candidate_identity_context_v1"
                and row.get("identity_context_hash") == orphan_hash
            ]

        self.assertEqual(logical_after, logical_before)
        self.assertEqual(
            self._logical_evidence_snapshot(evidence_after, "F1"),
            self._logical_evidence_snapshot(evidence_before, "F1"),
        )
        self.assertEqual(offsets_after, offsets_before)
        self.assertTrue(replay["idempotent_replay"])
        self.assertEqual(size_after_replay, replay_size)
        self.assertEqual(final_fingerprint, expected_fingerprint)
        self.assertEqual(
            self._logical_evidence_snapshot(final_base, "F1"),
            self._logical_evidence_snapshot(evidence_before, "F1"),
        )
        self.assertEqual(replacement.trials.seeds, (1000,))
        self.assertEqual(len(physical_orphan_contexts), 2)
        self.assertNotIn(
            "candidate_identity_context_v1",
            {row["record_type"] for row in final_records},
        )

    def test_store_is_path_backed_and_reopens_ordinary_trial_evidence(self):
        from blb_stage2_rl.candidate_store import CandidateStore
        from blb_stage2_rl.statistical_constraints import TrialSeries

        action = [1, 2, 3]
        context = {"action_space_version": "layerwise-v1", "fidelity": "F4"}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            CandidateStore(path).append_trial_group(
                action,
                TrialSeries(
                    loss=[0.3], metric1=[0.9], metric2=[0.8], seeds=[101],
                ),
                {
                    "identity_context": context,
                    "fidelity": "F4",
                    "axis": "joint",
                    "bank": "A",
                    "group_index": 0,
                },
                compact=True,
            )
            evidence = CandidateStore(path).trial_evidence_for_action(
                action, context,
            )

        self.assertFalse(hasattr(CandidateStore, "from_bytes"))
        self.assertFalse(hasattr(CandidateStore, "append_physical_trial_event"))
        self.assertFalse(hasattr(CandidateStore, "physical_trial_accounting"))
        self.assertEqual(evidence.trials.seeds, (101,))
        self.assertEqual(evidence.trials.loss, (0.3,))
