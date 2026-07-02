"""Generate Phase-0 BLB optimization preflight reports."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REPORT_PHASE_DIRS = (
    "phase0_preflight",
    "phase1_registry",
    "phase2_eval",
    "phase3_sensitivity",
    "phase4_search",
    "phase5_rl",
    "phase6_confirm",
    "phase7_final",
)
ENTRYPOINT_PATTERNS = (
    "BLBStage2RLRunner",
    "stage2_rl_variant",
    "blb_v3",
    "RescaleOptimizer",
    "InProcessInvoker",
    "action_vector_to_cfgs",
    "make_all_max_action_vector",
)
CODE_CONFIG_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".conf",
    ".sh",
}
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "tmp_tests",
}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_repo_files(repo_root: Path) -> Iterable[Path]:
    root = Path(repo_root).resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        current = Path(dirpath)
        for filename in filenames:
            yield current / filename


def build_phase0_entrypoint_report(repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    launcher = root / "llama_7B_LayerImportance.sh"
    runner = root / "blb_stage2_rl" / "runner.py"
    action_space = root / "blb_stage2_rl" / "action_space.py"
    rescale = root / "Rescale_optimizer"
    preset = root / "presets" / "mrpc-blb-stage2-rl.conf"
    lines = [
        "# BLB Phase 0 Entrypoints",
        "",
        "1. Main training entrypoint: `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl --fresh`.",
        "2. Resume entrypoint: `bash llama_7B_LayerImportance.sh run rl --preset mrpc-blb-stage2-rl`.",
        "3. Stage-2 variant switch: `--stage2-rl-variant blb_v3` in `presets/mrpc-blb-stage2-rl.conf`.",
        "4. Runner implementation: `blb_stage2_rl/runner.py` (`BLBStage2RLRunner`).",
        "5. Action registry/decode implementation: `blb_stage2_rl/action_space.py`.",
        "6. Rescale optimizer path: `Rescale_optimizer`; BLB Stage-2 uses the in-process optimizer path.",
        "",
        "| artifact | exists |",
        "|---|---:|",
        f"| llama_7B_LayerImportance.sh | {str(launcher.exists()).lower()} |",
        f"| presets/mrpc-blb-stage2-rl.conf | {str(preset.exists()).lower()} |",
        f"| blb_stage2_rl/runner.py | {str(runner.exists()).lower()} |",
        f"| blb_stage2_rl/action_space.py | {str(action_space.exists()).lower()} |",
        f"| Rescale_optimizer | {str(rescale.exists()).lower()} |",
        "",
        "Current defaults from the preserved operator surface are resolved by the launcher and preset; this report is an audit artifact, not a replacement command.",
    ]
    return "\n".join(lines) + "\n"


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grep_entrypoint_file(repo_root: Path, path: Path, pattern: re.Pattern[str]) -> List[str]:
    if path.suffix.lower() not in CODE_CONFIG_SUFFIXES:
        return []
    rel = _relative(path, repo_root)
    try:
        handle = path.open("r", encoding="utf-8", errors="ignore")
    except OSError:
        return []
    matches: List[str] = []
    with handle:
        for lineno, line in enumerate(handle, start=1):
            if pattern.search(line):
                matches.append(f"{rel}:{lineno}:{line.strip()}")
    return matches


def _grep_entrypoint_paths(repo_root: Path, paths: Iterable[Path]) -> List[str]:
    pattern: re.Pattern[str] = re.compile("|".join(re.escape(item) for item in ENTRYPOINT_PATTERNS))
    matches: List[str] = []
    for path in paths:
        matches.extend(_grep_entrypoint_file(repo_root, path, pattern))
    return matches


def _grep_entrypoints(repo_root: Path) -> List[str]:
    return _grep_entrypoint_paths(repo_root, iter_repo_files(repo_root))


def write_phase0_reports(
        repo_root: os.PathLike[str] | str,
        *,
        reports_dir: os.PathLike[str] | str = "reports",
        ) -> dict:
    root = Path(repo_root).resolve()
    reports = root / reports_dir
    blb_root = reports / "blb_opt"
    for dirname in REPORT_PHASE_DIRS:
        (blb_root / dirname).mkdir(parents=True, exist_ok=True)

    pattern: re.Pattern[str] = re.compile("|".join(re.escape(item) for item in ENTRYPOINT_PATTERNS))
    files: List[str] = []
    code_config: List[str] = []
    grep_matches: List[str] = []
    for path in iter_repo_files(root):
        rel = _relative(path, root)
        files.append(rel)
        if path.suffix.lower() in CODE_CONFIG_SUFFIXES:
            code_config.append(rel)
            grep_matches.extend(_grep_entrypoint_file(root, path, pattern))
    files.sort()
    code_config.sort()
    _write_lines(reports / "repo_file_list.txt", files)
    _write_lines(reports / "repo_code_config_files.txt", code_config)
    _write_lines(reports / "blb_entrypoints_grep.txt", grep_matches)
    phase0_path = reports / "phase0_entrypoints.md"
    phase0_path.write_text(build_phase0_entrypoint_report(root), encoding="utf-8")
    return {
        "repo_file_list": str(reports / "repo_file_list.txt"),
        "repo_code_config_files": str(reports / "repo_code_config_files.txt"),
        "blb_entrypoints_grep": str(reports / "blb_entrypoints_grep.txt"),
        "phase0_entrypoints": str(phase0_path),
        "blb_opt": str(blb_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args(argv)
    paths = write_phase0_reports(args.repo_root, reports_dir=args.reports_dir)
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
