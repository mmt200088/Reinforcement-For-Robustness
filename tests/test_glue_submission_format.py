import os
import tempfile
import unittest

import numpy as np

import generate_glue_submission as glue_submission


class GlueSubmissionFormatTests(unittest.TestCase):
    def test_generic_textattack_mnli_labels_use_textattack_order(self):
        task_config = dict(glue_submission.TASK_REGISTRY["mnli"])
        task_config["model_name"] = "textattack/bert-base-uncased-MNLI"
        logits = np.eye(3, dtype=np.float32)

        predictions = glue_submission.logits_to_predictions(
            logits,
            task_config,
            "mnli",
            {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        )

        self.assertEqual(predictions, ["contradiction", "entailment", "neutral"])

    def test_generic_mnli_labels_use_registry_order_for_non_textattack_models(self):
        task_config = dict(glue_submission.TASK_REGISTRY["mnli"])
        task_config["model_name"] = "example/mnli-model"
        logits = np.eye(3, dtype=np.float32)

        predictions = glue_submission.logits_to_predictions(
            logits,
            task_config,
            "mnli",
            {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
        )

        self.assertEqual(predictions, ["entailment", "neutral", "contradiction"])

    def test_generic_binary_labels_use_glue_submission_labels(self):
        logits = np.eye(2, dtype=np.float32)

        qnli_predictions = glue_submission.logits_to_predictions(
            logits,
            glue_submission.TASK_REGISTRY["qnli"],
            "qnli",
            {0: "LABEL_0", 1: "LABEL_1"},
        )
        sst2_predictions = glue_submission.logits_to_predictions(
            logits,
            glue_submission.TASK_REGISTRY["sst2"],
            "sst2",
            {0: "LABEL_0", 1: "LABEL_1"},
        )

        self.assertEqual(qnli_predictions, ["entailment", "not_entailment"])
        self.assertEqual(sst2_predictions, ["0", "1"])

    def test_validate_tsv_file_rejects_non_glue_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "QNLI.tsv")
            with open(path, "w") as f:
                f.write("index\tprediction\n")
                f.write("0\tlabel_0\n")

            actual_lines, errors = glue_submission.validate_tsv_file(path, "QNLI.tsv", 2)

        self.assertEqual(actual_lines, 2)
        self.assertTrue(errors)
        self.assertIn("not in", errors[0])

    def test_remove_stale_submission_files_keeps_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stale_tsv = os.path.join(tmpdir, "CoLA.tsv")
            stale_zip = os.path.join(tmpdir, "submission.zip")
            unrelated = os.path.join(tmpdir, "notes.txt")
            for path in (stale_tsv, stale_zip, unrelated):
                with open(path, "w") as f:
                    f.write("x")

            glue_submission.remove_stale_submission_files(tmpdir)

            self.assertFalse(os.path.exists(stale_tsv))
            self.assertFalse(os.path.exists(stale_zip))
            self.assertTrue(os.path.exists(unrelated))


if __name__ == "__main__":
    unittest.main()
