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

__all__ = [
    # graph
    "AmplitudeProfile", "SNRRequirement", "NoiseLookupTable",
    "CostParams", "ComputeNode", "CutPoint",
    "NodeType", "CUT_POINT_TYPES", "MULTIPLICATION_TYPES", "RescaleGraph",
    "StageEdge", "TailEdge", "propagate_scale",
    # feasibility
    "build_feasibility_dag", "find_min_sf",
    # reachability
    "Reachability", "compute_reachability",
    # backward DP
    "DPResult", "build_dp_table", "backtrack_from", "deviate_at",
    "run_backward_dp", "stage_edge_cost", "tail_edge_cost",
    # modulus chain
    "ChainResult", "ModulusChain",
    "validate_cut_points", "repair_chain",
    "best_first_repairable_skeleton", "compress_headroom",
    "construct_modulus_chain",
    # optimizer
    "OptimizationConfig", "OptimizationResult",
    "optimize_rescale", "print_result",
    # config loader
    "load_graph_from_json", "build_from_dict",
    # replan (what-if + fusion-tolerant feasibility)
    "FusionEvent", "ReplanInputs", "ReplanResult", "replan_with_user_actions",
]
