import builtins
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class BLBCandidateStoreIdentityTests(unittest.TestCase):
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

    def test_append_streams_jsonl_rows_without_full_row_write(self):
        from blb_stage2_rl.candidate_store import CandidateStore

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate_store.jsonl"
            store = CandidateStore(path)
            fake_handle = mock.MagicMock()
            fake_handle.__enter__.return_value = fake_handle
            fake_handle.__exit__.return_value = None
            original_open = Path.open

            def guarded_open(open_path, *args, **kwargs):
                if Path(open_path) == path:
                    return fake_handle
                return original_open(open_path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                saved = store.append({
                    "action_indices": [1, 2, 3],
                    "fidelity": "F1",
                    "valid": True,
                })

        self.assertEqual(saved["action_indices"], [1, 2, 3])
        fake_handle.writelines.assert_called_once()
        newline_writes = [
            call for call in fake_handle.write.call_args_list
            if call.args == ("\n",)
        ]
        self.assertEqual(len(newline_writes), 1)

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
