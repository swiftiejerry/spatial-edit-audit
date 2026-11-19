"""Controlled simulation and counterfactual audit toolkit for selective local spatial multi-omics edits."""

from .calibration import (
    betabinomial_tail_summary,
    fit_order_statistic_cutoffs,
    mondrian_stratum_cutoffs,
)
from .contrast import (
    aggregate_rule_scores,
    local_molecular_contrast,
    rank_scorable_candidates,
    score_candidate_pool,
)
from .evaluation import (
    compare_counterfactual_arms,
    summarize_fixed_budget_contrasts,
    summarize_quota_coupling,
)
from .partitioners import (
    build_grown_patch_candidates,
    evaluate_partition_coverage,
    partition_kmeans_components,
    partition_ward_graph,
)
from .preprocessing import (
    SectionBaseline,
    build_section_baseline,
    load_section_baseline,
    project_count_matrix_svd,
)
from .selective_gates import (
    GateConfig,
    apply_reference_intervention,
    evaluate_dual_calibrated_gate,
    evaluate_multigate_prototype,
)
from .simulation import (
    BLOCK_SEED_ROOT,
    generate_main_case,
    generate_secondary_case,
)
from .spatial_graph import (
    build_knn_graph,
    compute_two_hop_ring,
    partition_spatial_tiles,
)

__all__ = [
    "BLOCK_SEED_ROOT",
    "GateConfig",
    "SectionBaseline",
    "aggregate_rule_scores",
    "apply_reference_intervention",
    "betabinomial_tail_summary",
    "build_grown_patch_candidates",
    "build_knn_graph",
    "build_section_baseline",
    "compare_counterfactual_arms",
    "compute_two_hop_ring",
    "evaluate_dual_calibrated_gate",
    "evaluate_multigate_prototype",
    "evaluate_partition_coverage",
    "fit_order_statistic_cutoffs",
    "generate_main_case",
    "generate_secondary_case",
    "load_section_baseline",
    "local_molecular_contrast",
    "mondrian_stratum_cutoffs",
    "partition_kmeans_components",
    "partition_spatial_tiles",
    "partition_ward_graph",
    "project_count_matrix_svd",
    "rank_scorable_candidates",
    "score_candidate_pool",
    "summarize_fixed_budget_contrasts",
    "summarize_quota_coupling",
]
