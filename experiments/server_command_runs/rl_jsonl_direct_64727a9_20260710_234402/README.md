# Direct RL JSONL Encoding Evidence

## Scope

`RLDataPointWriter` already reused a buffered `JSONEncoder`, but each row was
first recursively normalized with `to_jsonable(..., preserve_native=True)`.
The encoder was also configured with `default=json_default`, so that eager
walk duplicated conversion dispatch for every JSON-native field in every
Stage-1 step, Stage-1 episode, Stage-2 episode, and PPO-update row.

Production commit `64727a9` passes the original payload directly to the reused
encoder. Non-native NumPy, Path, dataclass, and optional Torch leaves are still
normalized by `json_default` when the encoder encounters them. Buffer size,
flush interval, key sorting, whitespace, newline, and artifact schemas are
unchanged.

## TDD Gate

- RED source: `4225543606ed4e5aba0efc6e7cc219837637cc18`
- RED command: `python -m unittest tests.test_rl_data_points.RLDataPointWriterTest.test_training_data_jsonl_writer_avoids_eager_payload_normalization -v`
- RED result: exit `1`; the old writer called the patched eager normalizer.
- GREEN source: `64727a94d863ccb3ae6f5cb3dbcccd6175327ad0`
- GREEN compile command: `python -m py_compile rl_data_points.py json_utils.py tests/test_rl_data_points.py`
- GREEN test command: `python -m unittest tests.test_rl_data_points -v`
- GREEN result: compile exit `0`; all 25 tests passed in `0.84s`.

The server sparse checkout temporarily included the tracked `reports/` path
because the full contract module has static report-source checks. The exact
original sparse path set was restored after the test, and the final server
worktree remained clean.

## Benchmark

The benchmark alternated old-first and direct-first order for seven measured
samples after warmup. Each sample serialized 30,000 rows through the same
reused `JSONEncoder`; encoder chunks were consumed without disk I/O so the
measurement isolates repeated normalization and encoding CPU time.

| Fixture | Old median | Direct median | Speedup | Byte parity |
| --- | ---: | ---: | ---: | --- |
| Stage-1 native step row | 1.811772s | 0.530645s | 3.414x | exact |
| Stage-2 NumPy-heavy episode row | 1.978704s | 1.799263s | 1.100x | exact |

For the Stage-1 fixture, scaling the measured serialization-only delta to
600,000 step rows (50,000 episodes times 12 decisions) gives about `25.62s` of
CPU time avoided. This is not claimed as an end-to-end training speedup. The
Stage-2 fixture projection is `3.59s` per 600,000 rows.

`green/benchmark.jsonl` contains all samples, row sizes, SHA-256 values, and
the projections. Both old and direct paths produced byte-identical serialized
rows before timing.

## Server

- Python `3.10.19`
- 20 logical CPUs, 125 GiB RAM
- NVIDIA RTX 4090 24 GiB (not used by this CPU serialization benchmark)

