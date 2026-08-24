from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Type

from final_evaluation_module import (
    UnifiedFinalEvaluationModule,
    require_final_evaluation_protocol,
)
from glue_data_protocol import FINAL_EVAL_SPLIT

from .config import (
    DEFAULT_PRESET,
    FinalEvalSettings,
    default_config_for_algorithm,
    final_eval_results_dir,
    parse_final_eval_settings,
    resolve_repo_path,
)


def load_embedded_settings(
    *,
    preset_name: str = "",
    output_root: str = "",
    run_name: str = "",
) -> FinalEvalSettings:
    preset = preset_name or DEFAULT_PRESET
    settings = parse_final_eval_settings(
        ["--preset", preset],
        require_resume_for_search=False,
    )
    updates = {}
    if output_root:
        updates["output_root"] = output_root
    if run_name:
        updates["run_name"] = run_name
    if updates:
        settings = dataclasses.replace(settings, **updates)
    return settings


def run_embedded_final_eval(
    *,
    evaluator,
    search_best_stage1,
    search_best_stage2,
    baseline_stage1_gelu,
    baseline_stage1_softmax,
    baseline_noise_tot_c,
    limit_loss,
    limit_p,
    limit_s,
    preset_name: str = "",
    output_root: str = "",
    run_name: str = "",
    module_cls: Optional[Type[UnifiedFinalEvaluationModule]] = None,
):
    """Run final eval from training using the independent final_eval preset.

    The training path always evaluates the just-found search configs, so the
    embedded call forces ``config_source='search'``. The preset still controls
    repeat counts, random comparison counts, seed, output root, and fallback
    JSON path.
    """
    require_final_evaluation_protocol(
        evaluator,
        search_results=(search_best_stage1, search_best_stage2),
        requested_split=FINAL_EVAL_SPLIT,
    )

    settings = load_embedded_settings(
        preset_name=preset_name,
        output_root=output_root,
        run_name=run_name,
    )
    if (settings.preset or DEFAULT_PRESET) == DEFAULT_PRESET:
        algorithm = str(getattr(evaluator, "search_algorithm", "rl") or "rl")
        settings = dataclasses.replace(
            settings,
            algorithm=algorithm,
            config=resolve_repo_path(default_config_for_algorithm(algorithm)),
        )
    source_run_dir = str(getattr(evaluator, "run_output_dir", "") or "")
    results_dir = final_eval_results_dir(
        settings,
        source_run_dir=source_run_dir,
        timestamp_if_needed=False,
    )
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    log = getattr(evaluator, "log", None)
    if callable(log):
        log("\n" + "=" * 60)
        log("PHASE: INDEPENDENT FINAL_EVAL MODULE")
        log(f"final_eval preset={settings.preset or DEFAULT_PRESET}")
        log(f"final_eval output_dir={results_dir}")
        log(
            "final_eval args from preset: "
            f"repeat={settings.repeat}, seed={settings.random_seed}, "
            f"perm={settings.perm_trials}, cost={settings.cost_trials}, "
            f"budget={settings.budget_trials}, stage1_budget={settings.stage1_budget_trials}, "
            f"stage2_budget={settings.stage2_budget_trials}"
        )
        log("embedded training call uses config_source=search for the just-found configs")
        log("=" * 60)

    runner_type = module_cls or UnifiedFinalEvaluationModule
    should_run_blb_action_eval = (
        isinstance(search_best_stage2, dict)
        and search_best_stage2.get("blb_v3_best_action_vec") is not None
    ) or bool(settings.action_config or settings.action_ranges or settings.action_fixed)
    if should_run_blb_action_eval:
        from .blb_action_eval import BLBActionFinalEvaluationModule

        # The BLB path's "random_count" was previously a stand-in for the
        # legacy perm/cost/budget family counts; the SF-K-first comparison
        # uses --cost-match-count (default 50). If the user explicitly turned
        # on legacy random groups via --random + non-zero perm/cost/budget
        # trials, treat that as a request to also pull a same-cost random
        # group. ``cost_match_count`` is the source of truth from now on.
        random_enabled = bool(
            settings.random_enabled
            or settings.cost_match_count > 0
        )
        runner = BLBActionFinalEvaluationModule(
            evaluator=evaluator,
            config_source="search",
            config_path=settings.config,
            manual_stage1_gelu=None,
            manual_stage1_softmax=None,
            random_seed=settings.random_seed,
            random_enabled=random_enabled,
            random_count=(
                settings.perm_trials
                + settings.cost_trials
                + settings.budget_trials
                + settings.stage1_budget_trials
                + settings.stage2_budget_trials
            ),
            repeat_n=settings.repeat,
            results_dir=str(results_dir),
            action_config_path=settings.action_config,
            action_ranges=settings.action_ranges,
            action_fixed=settings.action_fixed,
            cost_match_count=settings.cost_match_count,
            cost_match_max_attempts=settings.cost_match_max_attempts,
            glue_submission_enabled=settings.glue_submission_enabled,
            glue_submission_seed=settings.glue_submission_seed,
        )
        return runner.run(
            search_best_stage1=search_best_stage1,
            search_best_stage2=search_best_stage2,
            baseline_stage1_gelu=baseline_stage1_gelu,
            baseline_stage1_softmax=baseline_stage1_softmax,
            baseline_noise_tot_c=float(baseline_noise_tot_c),
            limit_loss=float(limit_loss),
            limit_p=float(limit_p),
            limit_s=float(limit_s),
        )

    runner = runner_type(
        evaluator=evaluator,
        config_source="search",
        config_path=settings.config,
        manual_stage1_gelu=None,
        manual_stage1_softmax=None,
        manual_stage2_noise=None,
        random_seed=settings.random_seed,
        permutation_trials=settings.perm_trials,
        cost_equivalent_trials=settings.cost_trials,
        budget_equivalent_trials=settings.budget_trials,
        stage1_budget_trials=settings.stage1_budget_trials,
        stage2_budget_trials=settings.stage2_budget_trials,
        repeat_n=settings.repeat,
        results_dir=str(results_dir),
    )
    return runner.run(
        search_best_stage1=search_best_stage1,
        search_best_stage2=search_best_stage2,
        baseline_stage1_gelu=baseline_stage1_gelu,
        baseline_stage1_softmax=baseline_stage1_softmax,
        baseline_noise_tot_c=float(baseline_noise_tot_c),
        limit_loss=float(limit_loss),
        limit_p=float(limit_p),
        limit_s=float(limit_s),
    )
