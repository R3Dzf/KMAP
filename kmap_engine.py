"""Karnaugh map indexing and grouping helpers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Sequence, Set, Tuple

from sympy import And, Not, Or, Symbol

# Gray-code ordering for two-bit combinations
GRAY4: Sequence[Tuple[int, int]] = ((0, 0), (0, 1), (1, 1), (1, 0))


@dataclass(frozen=True)
class GroupRect:
    """Descriptor for a grouped rectangle on the K-map."""

    r0: int
    rows: int
    c0: int
    cols: int
    cells: FrozenSet[Tuple[int, int]]


def map_dimensions(nvars: int) -> Tuple[int, int]:
    """Return (rows, cols) for K-map based on variable count."""
    if nvars == 2:
        return 2, 2
    if nvars == 3:
        return 2, 4
    if nvars == 4:
        return 4, 4
    raise ValueError("K-map available for 2-4 variables.")


def idx_to_rc(nvars: int, idx: int, orient_cols_is_ab: bool = True) -> Tuple[int, int]:
    """Translate a minterm index to (row, col) coordinates."""
    if nvars == 2:
        a = (idx >> 1) & 1
        b = idx & 1
        return (a, b) if not orient_cols_is_ab else (b, a)
    if nvars == 3:
        a = (idx >> 2) & 1
        b = (idx >> 1) & 1
        c = idx & 1
        col = GRAY4.index((b, c))
        return a, col
    if nvars == 4:
        a = (idx >> 3) & 1
        b = (idx >> 2) & 1
        c = (idx >> 1) & 1
        d = idx & 1
        if orient_cols_is_ab:
            col = GRAY4.index((a, b))  # AB columns
            row = GRAY4.index((c, d))  # CD rows
        else:
            col = GRAY4.index((c, d))  # CD columns
            row = GRAY4.index((a, b))  # AB rows
        return row, col
    raise ValueError("K-map available for 2-4 variables.")


def rc_to_bits_4(r: int, c: int, orient_cols_is_ab: bool) -> Tuple[int, int, int, int]:
    """Return (A, B, C, D) bit tuple for a given grid cell when n=4."""
    if orient_cols_is_ab:
        a, b = GRAY4[c]
        c_bit, d = GRAY4[r]
    else:
        c_bit, d = GRAY4[c]
        a, b = GRAY4[r]
    return a, b, c_bit, d


def rect_cells(
    r0: int, rows: int, c0: int, cols: int, nrows: int, ncols: int
) -> Set[Tuple[int, int]]:
    """Return the set of cells covered by a rectangle (with wrap-around)."""
    cells: Set[Tuple[int, int]] = set()
    for dr in range(rows):
        for dc in range(cols):
            r = (r0 + dr) % nrows
            c = (c0 + dc) % ncols
            cells.add((r, c))
    return cells


def _power_sizes(limit: int) -> List[int]:
    sizes = [1]
    while sizes[-1] * 2 <= limit:
        sizes.append(sizes[-1] * 2)
    return sizes


def all_rects(nrows: int, ncols: int) -> List[Tuple[int, int, int, int]]:
    """Generate all rectangle placements sorted by descending area."""
    sizes_r = _power_sizes(nrows)
    sizes_c = _power_sizes(ncols)
    rects: List[Tuple[int, int, int, int]] = []
    for rows in reversed(sizes_r):
        for cols in reversed(sizes_c):
            for r0 in range(nrows):
                for c0 in range(ncols):
                    rects.append((r0, rows, c0, cols))
    rects.sort(key=lambda r: r[1] * r[3], reverse=True)
    return rects


def map_minterms_to_cells(
    nvars: int, minterms: Iterable[int], orient_cols_is_ab: bool = True
) -> Set[Tuple[int, int]]:
    """Convert minterm indices to a set of (row, col) cells."""
    return {idx_to_rc(nvars, m, orient_cols_is_ab) for m in minterms}


def select_group_rectangles(
    nrows: int, ncols: int, ones_cells: Set[Tuple[int, int]]
) -> List[GroupRect]:
    """Select grouped rectangles following the legacy heuristic."""
    if not ones_cells:
        return []

    rects = all_rects(nrows, ncols)
    candidates: List[GroupRect] = []
    for r0, rows, c0, cols in rects:
        full_cells = rect_cells(r0, rows, c0, cols, nrows, ncols)
        if full_cells and full_cells <= ones_cells:
            candidates.append(
                GroupRect(
                    r0=r0,
                    rows=rows,
                    c0=c0,
                    cols=cols,
                    cells=frozenset(full_cells),
                )
            )

    chosen: List[GroupRect] = []
    covered_all: Set[Tuple[int, int]] = set()

    while True:
        essentials: List[GroupRect] = []
        for cell in ones_cells - covered_all:
            containing = [g for g in candidates if cell in g.cells]
            if len(containing) == 1:
                essentials.append(containing[0])
        if not essentials:
            break
        for group in essentials:
            if group not in chosen:
                chosen.append(group)
                covered_all |= set(group.cells)

    remaining = ones_cells - covered_all
    while remaining:
        best = max(candidates, key=lambda g: len(g.cells & remaining))
        if not (best.cells & remaining):
            break
        chosen.append(best)
        covered_all |= set(best.cells)
        remaining = ones_cells - covered_all

    return chosen


def label_rect_4(group: GroupRect, orient_cols_is_ab: bool) -> str:
    """Return the SOP term label for a 4-variable group."""
    values = {"A": set(), "B": set(), "C": set(), "D": set()}
    for r, c in group.cells:
        a, b, c_bit, d = rc_to_bits_4(r, c, orient_cols_is_ab)
        values["A"].add(a)
        values["B"].add(b)
        values["C"].add(c_bit)
        values["D"].add(d)

    pieces: List[str] = []
    for var in ("A", "B", "C", "D"):
        bits = values[var]
        if bits == {0}:
            pieces.append(f"{var}'")
        elif bits == {1}:
            pieces.append(var)
    return "".join(pieces) if pieces else "1"


def _contiguous_span(coords: Set[int], size: int) -> Tuple[int, int]:
    unique = set(coords)
    length = len(unique)
    ordered = sorted(unique)
    for start in range(size):
        seq = {(start + offset) % size for offset in range(length)}
        if seq == unique:
            return start, length
    raise ValueError("Cells do not form a contiguous span on the map.")


def _term_to_minterms(term, vars_tuple) -> List[int]:
    factors = list(term.args) if isinstance(term, And) else [term]
    fixed = {}
    for factor in factors:
        if isinstance(factor, Symbol):
            fixed[factor] = 1
        elif isinstance(factor, Not) and isinstance(factor.args[0], Symbol):
            fixed[factor.args[0]] = 0
        else:
            raise ValueError("Unsupported factor within implicant.")

    mins = []
    free_vars = [v for v in vars_tuple if v not in fixed]
    for bits in itertools.product([0, 1], repeat=len(free_vars)):
        assignment = fixed.copy()
        assignment.update(dict(zip(free_vars, bits)))
        idx = 0
        for var in vars_tuple:
            idx = (idx << 1) | assignment[var]
        mins.append(idx)
    return mins


def expression_to_groups(expr, vars_tuple, orient_cols_is_ab: bool) -> List[GroupRect]:
    """Translate a simplified SymPy expression into explicit K-Map rectangles."""
    if expr in (False, None):
        return []
    nvars = len(vars_tuple)
    nrows, ncols = map_dimensions(nvars)

    if expr is True:
        cells = {
            idx_to_rc(nvars, idx, orient_cols_is_ab) for idx in range(2**nvars)
        }
        return [
            GroupRect(
                r0=0,
                rows=nrows,
                c0=0,
                cols=ncols,
                cells=frozenset(cells),
            )
        ]

    terms = list(expr.args) if isinstance(expr, Or) else [expr]
    groups: List[GroupRect] = []
    for term in terms:
        mins = _term_to_minterms(term, vars_tuple)
        cells = {idx_to_rc(nvars, idx, orient_cols_is_ab) for idx in mins}
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        r0, rows_len = _contiguous_span(rows, nrows)
        c0, cols_len = _contiguous_span(cols, ncols)
        groups.append(
            GroupRect(
                r0=r0,
                rows=rows_len,
                c0=c0,
                cols=cols_len,
                cells=frozenset(cells),
            )
        )
    return groups


__all__ = [
    "GroupRect",
    "GRAY4",
    "map_dimensions",
    "idx_to_rc",
    "rc_to_bits_4",
    "rect_cells",
    "all_rects",
    "map_minterms_to_cells",
    "expression_to_groups",
    "select_group_rectangles",
    "label_rect_4",
]
