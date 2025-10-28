"""Convenience exports for core K-Map logic helpers."""

from .logic import (
    get_variables,
    parse_human_expression,
    truth_minterms,
    minterms_to_expression,
    simplify_to_dnf,
    prime_format,
    validate_expression_variables,
    validate_minterm_range,
    simplify_from_minterms,
)
from .kmap_engine import (
    GroupRect,
    idx_to_rc,
    label_rect_4,
    map_dimensions,
    map_minterms_to_cells,
    rc_to_bits_4,
    rect_cells,
    expression_to_groups,
    select_group_rectangles,
)

__all__ = [
    "GroupRect",
    "get_variables",
    "idx_to_rc",
    "label_rect_4",
    "map_dimensions",
    "map_minterms_to_cells",
    "parse_human_expression",
    "truth_minterms",
    "minterms_to_expression",
    "simplify_to_dnf",
    "prime_format",
    "validate_expression_variables",
    "validate_minterm_range",
    "simplify_from_minterms",
    "rect_cells",
    "expression_to_groups",
    "rc_to_bits_4",
    "select_group_rectangles",
]
