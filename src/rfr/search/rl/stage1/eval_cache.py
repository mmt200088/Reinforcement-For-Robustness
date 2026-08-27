"""Thread-safe shared cache for Stage-1 per-episode plaintext evaluations.

Stage-1 reward scoring is a deterministic function of the episode's
``(gelu_degrees, softmax_degrees, split)``: the BERT weights are frozen, the
model runs in eval mode under ``inference_mode``, the dataloader iterates with
``shuffle=False``, and the Stage-1 path never installs noise. The single-GPU
path has exploited this for a long time via ``LayerImportanceEvaluator.
_eval_cache``; the multi-GPU worker path (``_stage1_evaluate_on_model``)
deliberately skipped that dict because un-locked cross-thread writes are a
race risk. This module is the lock-protected replacement so worker episodes
that repeat an already-scored config skip the entire install + forward.

Correctness notes (why a cache hit cannot change results):

* The cached value is the exact tuple a worker previously computed — same
  floats, not a re-derivation. Whether episode ``g`` computes or hits, the
  reward stream is identical, so the GPU-count-independence contract
  (``rollout_sig`` byte-equality between 1-worker and N-worker runs) is
  preserved trivially.
* Two workers may race to compute the same key concurrently; both compute,
  one write wins. Values are identical by determinism, so this is benign —
  deliberately no in-flight dedup (a waiting worker's GPU would idle for
  exactly as long as its own recompute would take).
* The cache is Stage-1-only. Stage 2 samples noise and is not a pure function
  of this configuration key.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Hashable, Optional, Tuple


class Stage1EvalCache:
    """Lock-protected ``{config key -> eval result tuple}`` with hit stats."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[Hashable, Any] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(gelu_degrees, softmax_degrees, split_name) -> Tuple:
        return (
            tuple(int(d) for d in gelu_degrees),
            tuple(int(d) for d in softmax_degrees),
            str(split_name),
        )

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self.hits += 1
            return value

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._entries[key] = value
            self.misses += 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats_line(self) -> str:
        with self._lock:
            total = self.hits + self.misses
            rate = (self.hits / total) if total else 0.0
            return (
                f"eval_cache hits={self.hits} misses={self.misses} "
                f"distinct={len(self._entries)} hit_rate={rate:.1%}"
            )
