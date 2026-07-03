# Installed Inference MNLI Accuracy Helper Evidence

Source commit: `7be83af`

Purpose: verify that MNLI full-eval accuracy reuses the shared direct-count
accuracy helper instead of carrying a local `np.mean()` implementation.

Server runs:

- Red: `/hy-tmp/rfr_mnli_accuracy_helper_red_0f6834b_20260704_041930`
- Green: `/hy-tmp/rfr_mnli_accuracy_helper_green_0f6834b_20260704_042030`

Green command coverage:

- `python -m py_compile blb_stage2_rl/inference_eval.py`
- `python -m unittest tests.test_blb_inference_eval_shared -v`
- Source guard confirming the MNLI branch calls `accuracy_from_labels()`
