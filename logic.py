"""Boolean logic utilities for the K-Map Simplifier UI."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

from sympy import And, Not, Or, Symbol, simplify_logic, symbols
from sympy.logic.boolalg import SOPform


def get_variables(n: int):
    """Return SymPy symbols (A, B, C, ...) for the requested variable count."""
    if n < 1:
        raise ValueError("Number of variables must be positive.")
    return symbols(" ".join(chr(65 + i) for i in range(n)))


def _beautify_expr(sympy_style: str, display_map=None) -> str:
    """Convert SymPy-friendly logical text into SOP-like notation."""
    parts: list[str] = []
    i = 0
    while i < len(sympy_style):
        ch = sympy_style[i]
        if ch == "~":
            if i + 1 < len(sympy_style) and sympy_style[i + 1].isalpha():
                symbol = sympy_style[i + 1]
                name = display_map.get(symbol, symbol) if display_map else symbol
                parts.append(f"{name}'")
                i += 2
                continue
            parts.append("~")
            i += 1
            continue
        if ch == "&":
            i += 1
            continue
        if ch == "|":
            parts.append(" + ")
            i += 1
            continue
        if ch.isalpha() and display_map:
            parts.append(display_map.get(ch, ch))
        else:
            parts.append(ch)
        i += 1
    return "".join(parts)


def parse_human_expression(raw: str, vars_tuple):
    """Parse algebraic Boolean input into a SymPy expression."""
    text = raw.replace("`", "'").replace(" ", "")
    if not text:
        raise ValueError("أدخل المعادلة أولًا.")

    allowed_names = [str(v) for v in vars_tuple]
    allowed_set = {name.lower() for name in allowed_names}
    display_map = {name: name for name in allowed_names}

    letters = [ch for ch in text if ch.isalpha()]
    unique_lower = []
    first_forms = {}
    for ch in letters:
        lower = ch.lower()
        if lower not in unique_lower:
            unique_lower.append(lower)
            first_forms[lower] = ch
    unique_set = set(unique_lower)

    mapping = {}
    if unique_set:
        if unique_set.issubset(allowed_set):
            for name in allowed_names:
                mapping[name.lower()] = name
                mapping[name] = name
                if name.lower() in first_forms:
                    display_map[name] = first_forms[name.lower()]
        elif unique_set.isdisjoint(allowed_set) and len(unique_set) <= len(vars_tuple):
            for alias, target in zip(unique_lower, allowed_names):
                mapping[alias] = target
                mapping[alias.upper()] = target
                display_map[target] = first_forms.get(alias, alias)
        else:
            allowed_display = ", ".join(allowed_names)
            raise ValueError(
                f"عدد المتغيرات المحدد يسمح فقط بالرموز: {allowed_display}"
            )
    else:
        mapping = {name: name for name in allowed_names}

    converted_chars = []
    for ch in text:
        if ch.isalpha():
            key = ch if ch in mapping else ch.lower()
            if key not in mapping:
                allowed_display = ", ".join(allowed_names)
                raise ValueError(
                    f"تم استخدام متغير غير مدعوم. استخدم فقط: {allowed_display}"
                )
            converted_chars.append(mapping[key])
        else:
            converted_chars.append(ch)
    stripped = "".join(converted_chars)

    for name in allowed_names:
        while f"{name}'" in stripped:
            stripped = stripped.replace(f"{name}'", f"~{name}")

    implicit = []
    for idx, ch in enumerate(stripped):
        implicit.append(ch)
        if idx < len(stripped) - 1:
            nxt = stripped[idx + 1]
            if (ch.isalpha() or ch == ")" or ch == "'") and (
                nxt.isalpha() or nxt in ("(", "~")
            ):
                implicit.append("&")
    py_expr = "".join(implicit).replace("+", "|")

    local = {str(v): v for v in vars_tuple}
    try:
        expr = eval(py_expr, {"__builtins__": {}}, local)
    except Exception as exc:
        raise ValueError("تعذر تحليل المعادلة. تأكد من الصياغة الصحيحة.") from exc
    return expr, _beautify_expr(py_expr, display_map), display_map


def truth_minterms(expr, vars_tuple) -> Sequence[int]:
    """Return indices whose assignments make the expression evaluate to True."""
    mins = []
    for idx, bits in enumerate(itertools.product([0, 1], repeat=len(vars_tuple))):
        subs = {var: bool(bit) for var, bit in zip(vars_tuple, bits)}
        if bool(expr.xreplace(subs)):
            mins.append(idx)
    return mins


def minterms_to_expression(minterms: Iterable[int], vars_tuple):
    """Build a SymPy expression representing the provided minterms."""
    mins = list(minterms)
    if not mins:
        return False

    n = len(vars_tuple)
    clauses = []
    for value in mins:
        literals = []
        for idx, var in enumerate(vars_tuple):
            bit = (value >> (n - 1 - idx)) & 1
            literals.append(var if bit else Not(var))
        clauses.append(And(*literals))
    return Or(*clauses)


def simplify_to_dnf(expr, dontcare=None):
    """Simplify expression using SymPy and return a DNF expression."""
    return simplify_logic(expr, form="dnf")


def prime_format(
    expr,
    var_order: Tuple[Symbol, ...],
    already_simplified: bool = False,
    display_map=None,
) -> str:
    """Format a (simplified) expression into SOP text following var_order."""
    if expr is False:
        return "0"
    if expr is True:
        return "1"

    if not already_simplified:
        expr = simplify_to_dnf(expr)

    def lit_to_str(lit):
        if isinstance(lit, Not) and isinstance(lit.args[0], Symbol):
            base = str(lit.args[0])
            name = display_map.get(base, base) if display_map else base
            return f"{name}'"
        if isinstance(lit, Symbol):
            name = str(lit)
            return display_map.get(name, name) if display_map else name
        return str(lit)

    terms = list(expr.args) if isinstance(expr, Or) else [expr]
    result = []
    for term in terms:
        literals = list(term.args) if isinstance(term, And) else [term]
        ordered = []
        for var in var_order:
            for lit in literals:
                if lit == var or (isinstance(lit, Not) and lit.args and lit.args[0] == var):
                    ordered.append(lit)
                    break
        result.append("".join(lit_to_str(item) for item in ordered) or "1")
    return " + ".join(result)


def validate_expression_variables(expr, vars_tuple):
    """Ensure the expression uses only symbols within vars_tuple."""
    allowed = set(vars_tuple)
    extras = expr.free_symbols - allowed
    if extras:
        names = ", ".join(sorted(str(sym) for sym in extras))
        raise ValueError(f"Expression contains variables outside the selected set: {names}")


def validate_minterm_range(minterms: Iterable[int], n: int) -> None:
    """Ensure all minterms are within the range for the current variable count."""
    max_valid = (1 << n) - 1
    invalid = [m for m in minterms if m < 0 or m > max_valid]
    if invalid:
        raise ValueError(
            f"Minterms out of range for {n} variables (0-{max_valid}): {sorted(set(invalid))}"
        )


def simplify_from_minterms(vars_tuple, minterms, dontcares=None):
    """Return simplified expression built from minterms and optional don't cares."""
    expr = SOPform(vars_tuple, minterms, dontcares or [])
    simplified = simplify_to_dnf(expr)
    sop_text = prime_format(simplified, vars_tuple, already_simplified=True)
    return simplified, sop_text
