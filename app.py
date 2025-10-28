import sys, os
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sympy.logic.boolalg import And, Or, Not, simplify_logic

from kmap_engine import (
    idx_to_rc,
    label_rect_4,
    map_dimensions,
    map_minterms_to_cells,
    expression_to_groups,
    select_group_rectangles,
)
from logic import (
    get_variables,
    parse_human_expression,
    truth_minterms,
    simplify_to_dnf,
    prime_format,
    validate_expression_variables,
    validate_minterm_range,
    simplify_from_minterms,
    minterms_to_expression,
)

# ------------------------------- إعداد الواجهة -------------------------------

COLOR_PALETTE = [
    "#e53935", "#1e88e5", "#43a047", "#f39c12",
    "#8e24aa", "#009688", "#6d4c41", "#2e86c1",
]

st.set_page_config(page_title="K-Map Simplifier", layout="wide")
st.title("🧮 K-Map Simplifier (by Ahmed Youssef)")
st.markdown("---")

mode = st.radio("اختيار نوع الإدخال:", ["إدخال المعادلة الجبرية", "إدخال Minterms"])
n = st.number_input("عدد المتغيرات:", min_value=2, max_value=4, value=4, step=1)

vars_tuple = get_variables(n)
display_map = {str(v): str(v) for v in vars_tuple}

if mode == "إدخال المعادلة الجبرية":
    raw_expr = st.text_input("أدخل المعادلة (مثال: A'B + A'B'C):")
else:
    raw_mins = st.text_input("أدخل أرقام الـ minterms (مثال: 1,3,5,7):")
    raw_dcs = st.text_input("أدخل أرقام don't care (اختياري):")

# ------------------------------- أدوات تنسيق POS -------------------------------
def lit_to_str_POS(lit, display_map=None):
    # في POS نحتاج OR داخل القوس: A + B' + ...
    if isinstance(lit, Not):
        base = str(lit.args[0])
        name = display_map.get(base, base) if display_map else base
        return f"{name}'"
    name = str(lit)
    return display_map.get(name, name) if display_map else name

def format_pos(expr, var_order, display_map=None):
    """
    يحوّل تعبير CNF إلى نص POS بالشكل:
    (A + B' + C)(A' + D) ...
    """
    if expr is True:
        return "1"
    if expr is False:
        return "0"

    # CNF يعني AND لمجاميع OR
    # لو كان OR واحد فقط بدون And:
    if isinstance(expr, Or):
        terms = [expr]
    elif isinstance(expr, And):
        terms = list(expr.args)
    else:
        # حالة حرف/نفي وحيد: (A) أو (A')
        terms = [expr]

    parts = []
    for term in terms:
        if isinstance(term, Or):
            literals = list(term.args)
        else:
            # لو term مفرد (رمز أو نفيه)، نغلفه كقوس واحد
            literals = [term]

        # نرتّب حسب ترتيب المتغيرات المطلوب
        ordered = []
        for var in var_order:
            for lit in literals:
                if lit == var or (isinstance(lit, Not) and lit.args and lit.args[0] == var):
                    ordered.append(lit)
                    break

        inside = " + ".join(lit_to_str_POS(l, display_map) for l in ordered) or "1"
        parts.append(f"({inside})")
    return "".join(parts)

# ------------------------------- عند الضغط على الزر -------------------------------
if st.button("حل المعادلة 🚀"):
    try:
        # =============== بناء الدالة والتبسيط ===============
        if mode == "إدخال المعادلة الجبرية":
            expr, parsed, display_map = parse_human_expression(raw_expr, vars_tuple)
            validate_expression_variables(expr, vars_tuple)
            mins = truth_minterms(expr, vars_tuple)
            simplified_sop = simplify_to_dnf(expr)                     # SOP مبسّط
            simplified_pos_expr = simplify_logic(expr, form="cnf")     # CNF → POS
            sop_text = prime_format(
                simplified_sop, vars_tuple, already_simplified=True, display_map=display_map
            )
            pos_text = format_pos(simplified_pos_expr, vars_tuple, display_map)

            dcs = []
        else:
            mins = [int(x.strip()) for x in raw_mins.split(",") if x.strip()]
            dcs = [int(x.strip()) for x in raw_dcs.split(",")] if raw_dcs else []
            validate_minterm_range(mins, len(vars_tuple))
            validate_minterm_range(dcs, len(vars_tuple))

            # SOP مبني من المينترمز (مع don't cares)
            simplified_sop, sop_text = simplify_from_minterms(vars_tuple, mins, dcs)

            # POS الصحيح: CNF لنفس الدالة (مع مراعاة الدونت كيرز)
            # نبني الدالة من المينترمز ثم نطلب CNF
            expr_from_mins = minterms_to_expression(mins, vars_tuple)
            simplified_pos_expr = simplify_logic(expr_from_mins, form="cnf")
            pos_text = format_pos(simplified_pos_expr, vars_tuple, display_map)

        # =============== عرض النتيجة ===============
        st.success(f"**SOP:**  \nF = {sop_text}")
        st.info(f"**POS:**  \nF = {pos_text}")

        steps = (
            f"• عدد المتغيرات: {n}\n"
            f"• minterms = {mins}\n"
            f"• don't cares = {dcs if dcs else '—'}\n"
            f"• الناتج (SOP): F = {sop_text}\n"
            f"• الناتج (POS): F = {pos_text}"
        )
        st.text_area("تفاصيل الحساب:", steps, height=180)

        # =============== رسم خريطة كارنوف داخل Card أنيقة ===============
        with st.container():
            st.markdown("### 🗺️ خريطة كارنوف (K-Map)")
            st.caption("تم توليد الخريطة تلقائيًا بناءً على المعادلة أو الـ minterms")

            size_map = {2: (3.2, 3.2), 3: (5.2, 3.4), 4: (5.2, 5.2)}
            fig, ax = plt.subplots(figsize=size_map.get(n, (4.2, 4.2)))

            nrows, ncols = map_dimensions(n)
            # وسّع الحدود شوية علشان عناوين الصفوف/الأعمدة ما تتقصّش
            ax.set_xlim(-0.6, ncols)
            ax.set_ylim(-0.6, nrows)
            ax.set_xticks(np.arange(0, ncols + 1))
            ax.set_yticks(np.arange(0, nrows + 1))
            ax.grid(True, color="#888", linewidth=1)
            ax.invert_yaxis()
            ax.set_facecolor("#fafafa")

            # ---- التسميات (Labels) ----
            if n == 2:
                col_labels, row_labels = ["A=0", "A=1"], ["B=0", "B=1"]
            elif n == 3:
                col_labels, row_labels = ["BC=00", "BC=01", "BC=11", "BC=10"], ["A=0", "A=1"]
            else:
                col_labels, row_labels = (
                    ["AB=00", "AB=01", "AB=11", "AB=10"],
                    ["CD=00", "CD=01", "CD=11", "CD=10"],
                )

            for j, lab in enumerate(col_labels):
                ax.text(j + 0.5, -0.25, lab, ha="center", va="center", fontsize=10, color="#333")
            for i, lab in enumerate(row_labels):
                ax.text(-0.25, i + 0.5, lab, ha="right", va="center", fontsize=10, color="#333")

            # ---- القيم داخل الخلايا ----
            ones = set(mins)
            dcs_set = set(dcs)

            for idx in range(2**n):
                r, c = idx_to_rc(n, idx)
                if idx in ones:
                    val, color = "1", "#1f3c88"
                elif idx in dcs_set:
                    val, color = "X", "#ff8c32"
                else:
                    val, color = "0", "#9aa7b7"
                ax.text(c + 0.5, r + 0.5, val, color=color,
                        fontsize=13, ha="center", va="center", weight="bold")
                ax.text(c + 0.05, r + 0.9, str(idx),
                        color="#777", fontsize=8, alpha=0.7)

            # ---- المجموعات (Grouping) بناءً على الـ SOP المبسّط ----
            ones_rc = map_minterms_to_cells(n, mins)
            groups = expression_to_groups(simplified_sop, vars_tuple, True) if simplified_sop is not None else []
            if not groups:
                groups = select_group_rectangles(nrows, ncols, ones_rc)

            for i, g in enumerate(groups):
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                rect = plt.Rectangle(
                    (g.c0, g.r0), g.cols, g.rows, fill=False,
                    color=color, lw=2.5, ls='-'
                )
                ax.add_patch(rect)
                label = label_rect_4(g, True) if n == 4 else ""
                if label:
                    ax.text(
                        g.c0 + g.cols / 2,
                        g.r0 + g.rows / 2,
                        label,
                        color=color,
                        fontsize=11,
                        ha="center",
                        va="center",
                        weight="bold",
                    )

            st.pyplot(fig)

    except Exception as e:
        st.error(f"حدث خطأ أثناء الحساب:\n{e}")
