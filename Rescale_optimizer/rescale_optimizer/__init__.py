"""
rescale_optimizer — HE 计算图 rescale 位置优化器

Four-stage pipeline:

    Stage 1  Feasibility-DAG construction
    Stage 2  Reachability analysis
    Stage 3  Backward Level-DP for cost-optimal placement
    Stage 4  Modulus-chain construction, repair, K-best fallback,
             headroom compression and final validation.

Primary entry point: :func:`optimize_rescale`.
"""

from .graph import (
    AmplitudeProfile,
    ComputeNode,
    CostParams,
    CutPoint,
    CUT_POINT_TYPES,
    MULTIPLICATION_TYPES,
    NodeType,
    NoiseLookupTable,
    RescaleGraph,
    SNRRequirement,
    StageEdge,
    TailEdge,
    propagate_scale,
)
from .feasibility import (
    build_feasibility_dag,
    find_min_sf,
)
from .reachability import (
    Reachability,
    compute_reachability,
)
from .backward_level_dp import (
    DPResult,
    build_dp_table,
    backtrack_from,
    deviate_at,
    run_backward_dp,
    stage_edge_cost,
    tail_edge_cost,
)
from .modulus_chain import (
    ChainResult,
    ModulusChain,
    best_first_repairable_skeleton,
    compress_headroom,
    construct_modulus_chain,
    repair_chain,
    validate_cut_points,
)
from .optimizer import (
    OptimizationConfig,
    OptimizationResult,
    optimize_rescale,
    print_result,
)
from .config_loader import (
    build_from_dict,
    load_graph_from_json,
)
from .replan import (
    FusionEvent,
    ReplanInputs,
    ReplanResult,
    replan_with_user_actions,
)
from .replan_interface import (
    BaselineRecord,
    CompactReplanResult,
    DEFAULT_FUSION_POLICY,
    FusionPair,
    ReplanSession,
    SUPPORTED_GELU_GRAPH_DEGREES,
    build_new_compact_config,
    build_replan_output_dict,
    default_allowed_fusion_pairs_for_graph,
    graph_key_for_stage1,
    iter_stage2_graph_targets,
    load_static_skeleton_baselines,
    replan_from_variables,
    resolve_allowed_fusion_pairs,
    split_replan_payload,
)

__all__ = [

    "AmplitudeProfile", "SNRRequirement", "NoiseLookupTable",
    "CostParams", "ComputeNode", "CutPoint",
    "NodeType", "CUT_POINT_TYPES", "MULTIPLICATION_TYPES", "RescaleGraph",
    "StageEdge", "TailEdge", "propagate_scale",

    "build_feasibility_dag", "find_min_sf",

    "Reachability", "compute_reachability",

    "DPResult", "build_dp_table", "backtrack_from", "deviate_at",
    "run_backward_dp", "stage_edge_cost", "tail_edge_cost",

    "ChainResult", "ModulusChain",
    "validate_cut_points", "repair_chain",
    "best_first_repairable_skeleton", "compress_headroom",
    "construct_modulus_chain",

    "OptimizationConfig", "OptimizationResult",
    "optimize_rescale", "print_result",

    "load_graph_from_json", "build_from_dict",

    "FusionEvent", "ReplanInputs", "ReplanResult", "replan_with_user_actions",

    "BaselineRecord", "CompactReplanResult", "DEFAULT_FUSION_POLICY",
    "FusionPair", "ReplanSession",
    "SUPPORTED_GELU_GRAPH_DEGREES", "build_new_compact_config",
    "build_replan_output_dict", "default_allowed_fusion_pairs_for_graph",
    "graph_key_for_stage1",
    "iter_stage2_graph_targets", "load_static_skeleton_baselines",
    "replan_from_variables", "resolve_allowed_fusion_pairs", "split_replan_payload",
]
