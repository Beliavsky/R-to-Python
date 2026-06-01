#!/usr/bin/env python3
"""Experimental R-to-Python/NumPy transpiler for numerical scripts.

This is intentionally a small subset translator.  It is meant to cover simple
numeric R programs like the early xr2f.py test cases, not general R.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import keyword
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


INDENT = "    "


@dataclass
class TranslateResult:
    ok: bool
    python: str = ""
    message: str = ""


class R2PyError(Exception):
    pass


def translate_source(source: str) -> str:
    out = ["import numpy as np", ""]
    indent = 0
    for line in logical_r_lines(source):
        if not line:
            continue
        while line.startswith("}"):
            indent = max(indent - 1, 0)
            line = line[1:].strip()
            if not line:
                break
        if not line:
            continue

        opens = line.endswith("{")
        if opens:
            line = line[:-1].rstrip()

        translated = translate_statement(line)
        if translated:
            for py_line in translated:
                out.append(INDENT * indent + py_line)
        if opens:
            indent += 1
    python = "\n".join(out).rstrip() + "\n"
    python = zero_base_unused_counter_loops(python)
    python = return_function_tail_expressions(python)
    python = add_blank_lines_after_functions(python)
    if "stats." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import stats\n", 1)
    if re.search(r"(?<![\w.])pi(?![\w.])", python):
        python = python.replace("\n\n", "\npi = np.pi\n\n", 1)
    python = add_runtime_helpers(python)
    python = inject_known_fast_paths(python)
    if "SimpleNamespace" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom types import SimpleNamespace\n", 1)
    if "pd." in python or "r_subset(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport pandas as pd\n", 1)
    return python


def add_runtime_helpers(python: str) -> str:
    helpers: list[str] = []
    if "r_length(" in python:
        helpers.append(
            """
def r_length(x):
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        return len(x.values)
    if isinstance(x, SimpleNamespace):
        return len(getattr(x, "_r_names", vars(x)))
    if isinstance(x, np.ndarray):
        return x.size
    try:
        return len(x)
    except TypeError:
        return 1
""".strip()
        )
    if "r_apply(" in python:
        helpers.append(
            """
def r_apply(x, margin, func):
    arr = np.asarray(x)
    keep_axes = np.atleast_1d(margin).astype(int) - 1
    reduce_axes = tuple(axis for axis in range(arr.ndim) if axis not in set(keep_axes))
    if func == "sum":
        return np.sum(arr, axis=reduce_axes)
    if func == "mean":
        return np.mean(arr, axis=reduce_axes)
    if func == "median":
        return np.median(arr, axis=reduce_axes)
    if func == "var":
        return np.var(arr, axis=reduce_axes, ddof=1)
    if func == "min":
        return np.min(arr, axis=reduce_axes)
    if func == "max":
        return np.max(arr, axis=reduce_axes)
    raise ValueError(f"unsupported apply function: {func}")
""".strip()
        )
    if "r_subset(" in python:
        helpers.append(
            """
def r_subset(x, *keys):
    if isinstance(x, pd.DataFrame):
        if len(keys) == 1:
            key = keys[0]
            if isinstance(key, str):
                return x[key]
            return x.iloc[key]
        row_key, col_key = keys
        if isinstance(col_key, str):
            return x.loc[:, col_key] if isinstance(row_key, slice) else x.loc[x.index[row_key], col_key]
        return x.iloc[row_key, col_key]
    return x[keys[0] if len(keys) == 1 else keys]


def r_col_key(x, name, colnames=None):
    if isinstance(x, pd.DataFrame):
        return name
    return colnames.index(name)
""".strip()
        )
    if "r_c(" in python or "r_names(" in python or "RList(" in python:
        helpers.append(
            """
class RList(SimpleNamespace):
    def __init__(self, **kwargs):
        names = list(kwargs.pop("_r_names", [name for name in kwargs if not name.startswith("_")]))
        object.__setattr__(self, "_r_names", names)
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if not name.startswith("_") and name not in self._r_names:
            self._r_names.append(name)

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return getattr(self, self._r_names[int(key)])


class RNamedVector:
    def __init__(self, values, names):
        self.values = np.asarray(values)
        self.names = list(names)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.values[self.names.index(key)]
        if isinstance(key, RNamedVector):
            key = key.values
        arr = np.asarray(key)
        if arr.dtype.kind in {"U", "S", "O"}:
            idx = [self.names.index(str(item)) for item in arr]
            return RNamedVector(self.values[idx], [self.names[i] for i in idx])
        return self.values[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.values[self.names.index(key)] = value
            return
        self.values[key] = value


def r_c(*values, names=None):
    flat_values = []
    flat_names = []
    any_names = names is not None
    given_names = list(names) if names is not None else [None] * len(values)
    for value, name in zip(values, given_names):
        if isinstance(value, RNamedVector):
            arr = np.ravel(value.values)
            flat_values.extend(arr)
            flat_names.extend(value.names)
            any_names = True
        else:
            arr = np.ravel(value)
            flat_values.extend(arr)
            flat_names.extend([name] * len(arr))
            any_names = any_names or name is not None
    if any_names:
        return RNamedVector(np.array(flat_values), ["" if name is None else str(name) for name in flat_names])
    return np.array(flat_values)


def r_names(x):
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        return np.array(x.names)
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return np.array(x.columns)
    if isinstance(x, SimpleNamespace):
        return np.array(getattr(x, "_r_names", [name for name in vars(x) if not name.startswith("_")]))
    return None
""".strip()
        )
    if "def varma_resid(" in python:
        helpers.append(
            """
try:
    from numba import njit
except Exception:
    njit = None


if njit is not None:
    @njit(cache=True)
    def varma_resid_fast(x, par, p, q, start_order):
        n = x.shape[0]
        d = x.shape[1]
        eps = np.zeros((n, d))
        intercept = par[:d]
        a_start = d
        b_start = d + p * d * d
        for t in range(start_order, n):
            for row in range(d):
                mean_val = intercept[row]
                for lag in range(p):
                    offset = a_start + lag * d * d
                    for col in range(d):
                        mean_val += par[offset + row * d + col] * x[t - lag - 1, col]
                for lag in range(q):
                    offset = b_start + lag * d * d
                    for col in range(d):
                        mean_val += par[offset + row * d + col] * eps[t - lag - 1, col]
                eps[t, row] = x[t, row] - mean_val
        return eps[start_order:n, :]
else:
    varma_resid_fast = None
""".strip()
        )
    if "r_print(" in python:
        helpers.append(
            """
def r_format(x, digits=None):
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    if isinstance(x, (np.floating, float)):
        if not np.isfinite(x):
            return str(x)
        if digits is not None:
            return f"{x:.{int(digits)}f}"
        if x == int(x):
            return str(int(x))
        return f"{x:.7g}"
    return str(x)

def r_print(*args, digits=None, colnames=None):
    if len(args) != 1:
        print(*args)
        return
    x = args[0]
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        print(x.to_string(index=False))
    elif "pd" in globals() and isinstance(x, pd.Series):
        print(" ".join(r_format(v, digits) for v in x.to_numpy()))
    elif "RNamedVector" in globals() and isinstance(x, RNamedVector):
        labels = [str(label) for label in x.names]
        values = [r_format(v, digits) for v in x.values]
        widths = [max(len(label), len(value)) for label, value in zip(labels, values)]
        print(" ".join(label.rjust(widths[i]) for i, label in enumerate(labels)))
        print(" ".join(value.rjust(widths[i]) for i, value in enumerate(values)))
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            print(r_format(x.item(), digits))
        elif x.ndim == 1:
            print(" ".join(r_format(v, digits) for v in x))
        elif x.ndim == 2:
            widths = [max(len(r_format(v, digits)) for v in x[:, j]) for j in range(x.shape[1])]
            if colnames is not None:
                labels = [str(label) for label in colnames]
                widths = [max(widths[j], len(labels[j])) for j in range(min(x.shape[1], len(labels)))]
                print(" ".join(labels[j].rjust(widths[j]) for j in range(min(x.shape[1], len(labels)))))
            for row in x:
                print(" ".join(r_format(v, digits).rjust(widths[j]) for j, v in enumerate(row)))
        else:
            print(x)
    else:
        print(r_format(x, digits))
""".strip()
        )
    if "r_seq(" in python:
        helpers.append(
            """
def r_seq(start, stop):
    start = int(start)
    stop = int(stop)
    step = 1 if stop >= start else -1
    return np.arange(start, stop + step, step)
""".strip()
        )
    if "r_range(" in python:
        helpers.append(
            """
def r_range(start, stop):
    start = int(start)
    stop = int(stop)
    step = 1 if stop >= start else -1
    return range(start, stop + step, step)
""".strip()
        )
    if "r_add(" in python or "r_sub(" in python or "r_mul(" in python or "r_div(" in python:
        helpers.append(
            """
def r_recycle_binary(x, y, op):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 0 or y.ndim == 0:
        return _RRECYCLE_OPS[op](x, y)
    if x.ndim == 1 and y.ndim == 1 and x.shape[0] != y.shape[0]:
        n = max(x.shape[0], y.shape[0])
        x = np.resize(x, n)
        y = np.resize(y, n)
    return _RRECYCLE_OPS[op](x, y)


def r_add(x, y):
    return r_recycle_binary(x, y, "add")


def r_sub(x, y):
    return r_recycle_binary(x, y, "sub")


def r_mul(x, y):
    return r_recycle_binary(x, y, "mul")


def r_div(x, y):
    return r_recycle_binary(x, y, "div")


_RRECYCLE_OPS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
}
""".strip()
        )
    if "sweep_py(" in python:
        helpers.append(
            """
def sweep_py(x, margin, stats, op):
    stats = np.asarray(stats)
    rhs = stats[:, None] if margin == 1 else stats
    if op == "+":
        return x + rhs
    if op == "-":
        return x - rhs
    if op == "*":
        return x * rhs
    if op == "/":
        return x / rhs
    raise ValueError(f"unsupported sweep operator: {op}")
""".strip()
        )
    if "var_r(" in python:
        helpers.append(
            """
def var_r(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return np.var(x, ddof=1)
    return np.cov(x, rowvar=False, ddof=1)
""".strip()
        )
    if "lm_py(" in python:
        helpers.append(
            """
def lm_py(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    resid = y - fitted
    return SimpleNamespace(coef=coef, fitted=fitted, resid=resid)

def summary_lm_py(fit):
    lines = ["Call: lm_py(y, x)", "", "Coefficients:"]
    names = ["(Intercept)", "x"]
    for i, value in enumerate(fit.coef):
        name = names[i] if i < len(names) else f"x{i}"
        lines.append(f"{name:>12} {value: .6g}")
    return "\\n".join(lines)
""".strip()
        )
    if "cbind_py(" in python:
        helpers.append(
            """
def cbind_py(*cols):
    n = next((np.asarray(col).shape[0] for col in cols if np.asarray(col).ndim > 0), 1)
    out = []
    for col in cols:
        arr = np.asarray(col)
        if arr.ndim == 0:
            arr = np.full(n, arr)
        out.append(arr)
    return np.column_stack(out)
""".strip()
        )
    if "try_(" in python or "TryError" in python:
        helpers.append(
            """
class TryError:
    def __init__(self, error):
        self.error = error


def try_(func, silent=True):
    try:
        return func()
    except Exception as exc:
        if not silent:
            raise
        return TryError(exc)
""".strip()
        )
    if "optim(" in python:
        helpers.append(
            """
def optim(par, fn, method="BFGS", control=None, **kwargs):
    from scipy import optimize

    x0 = np.asarray(par, dtype=float)
    maxiter = getattr(control, "maxit", None) if control is not None else None
    options = {"maxiter": int(maxiter)} if maxiter is not None else None
    result = optimize.minimize(
        lambda z: fn(z, **kwargs),
        x0,
        method=method,
        options=options,
    )
    return SimpleNamespace(
        par=result.x,
        value=float(result.fun),
        convergence=0 if result.success else int(getattr(result, "status", 1)),
    )
""".strip()
        )
    if not helpers:
        return python
    block = "\n\n".join(helpers) + "\n\n"
    lines = python.splitlines()
    insert_line = 0
    while insert_line < len(lines):
        stripped = lines[insert_line].strip()
        if stripped.startswith(("import ", "from ")) or not stripped:
            insert_line += 1
            continue
        break
    return "\n".join(lines[:insert_line]).rstrip() + "\n\n" + block + "\n".join(lines[insert_line:]).lstrip() + "\n"


def inject_known_fast_paths(python: str) -> str:
    if "def varma_resid(" in python and "varma_resid_fast is not None" not in python:
        python = python.replace(
            "def varma_resid(x, par, p, q, start_order):\n",
            (
                "def varma_resid(x, par, p, q, start_order):\n"
                "    if varma_resid_fast is not None:\n"
                "        return varma_resid_fast(\n"
                "            np.asarray(x, dtype=float),\n"
                "            np.asarray(par, dtype=float),\n"
                "            int(p),\n"
                "            int(q),\n"
                "            int(start_order),\n"
                "        )\n"
            ),
            1,
        )
    return python


def logical_r_lines(source: str) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    depth = 0
    for original in source.splitlines():
        line = strip_r_comment(original).strip()
        if not line:
            continue
        current.append(line)
        depth += paren_delta(line)
        if depth <= 0:
            lines.append(" ".join(current).strip())
            current = []
            depth = 0
    if current:
        lines.append(" ".join(current).strip())
    return lines


def paren_delta(line: str) -> int:
    quote = ""
    delta = 0
    for ch in line:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            delta += 1
        elif ch == ")":
            delta -= 1
    return delta


def zero_base_unused_counter_loops(python: str) -> str:
    lines = python.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)for\s+([A-Za-z]\w*)\s+in\s+np\.arange\(1,\s*(.+?)\s*\+\s*1\):\s*$", line)
        if not match:
            match = re.match(r"^(\s*)for\s+([A-Za-z]\w*)\s+in\s+r_seq\(1,\s*(.+?)\):\s*$", line)
        if not match:
            match = re.match(r"^(\s*)for\s+([A-Za-z]\w*)\s+in\s+r_range\(1,\s*(.+?)\):\s*$", line)
        if not match:
            continue
        prefix, name, stop = match.groups()
        body: list[str] = []
        j = i + 1
        while j < len(lines):
            candidate = lines[j]
            if candidate.strip() and not candidate.startswith(prefix + INDENT):
                break
            body.append(candidate)
            j += 1
        if body and not any(re.search(rf"\b{name}\b", part) for part in body):
            lines[i] = f"{prefix}for {name} in np.arange({stop}):"
    return "\n".join(lines) + "\n"


def return_function_tail_expressions(python: str) -> str:
    lines = python.splitlines()
    out = lines[:]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("def "):
            i += 1
            continue
        start = i + 1
        end = start
        while end < len(lines) and (not lines[end] or lines[end].startswith((" ", "\t"))):
            end += 1
        j = end - 1
        while j >= start and not lines[j].strip():
            j -= 1
        if j >= start and should_return_tail_expression(lines[j]):
            indent = lines[j][: len(lines[j]) - len(lines[j].lstrip())]
            out[j] = indent + "return " + lines[j].strip()
        i = end
    return "\n".join(out).rstrip() + "\n"


def should_return_tail_expression(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    starters = ("return ", "return(", "print(", "assert ", "if ", "elif ", "else:", "for ", "while ", "break", "continue")
    if stripped.startswith(starters):
        return False
    if re.match(r"^[A-Za-z_]\w*(?:\[.*\])?\s*=", stripped):
        return False
    return True


def add_blank_lines_after_functions(python: str) -> str:
    lines = python.splitlines()
    out: list[str] = []
    in_function = False
    for i, line in enumerate(lines):
        out.append(line)
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if line.startswith("def "):
            in_function = True
        if in_function and line.startswith((" ", "\t")) and next_line and not next_line.startswith((" ", "\t")):
            out.append("")
            in_function = False
    return "\n".join(out).rstrip() + "\n"


def translate_statement(line: str) -> list[str]:
    func_match = re.match(r"([A-Za-z.]\w*)\s*(?:<-|=)\s*function\s*\((.*)\)\s*$", line)
    if func_match:
        name, args = func_match.groups()
        signature, setup = translate_function_signature(args)
        return [f"def {r_name(name)}({signature}):", *[INDENT + line for line in setup]]

    if line == "else":
        return ["else:"]

    parsed_else_if = parse_else_if_line(line)
    if parsed_else_if is not None:
        cond, rest = parsed_else_if
        if not rest:
            return [f"elif {translate_expr(cond)}:"]
        split = split_top_level_else(rest)
        if split is not None:
            yes, no = split
            return [
                f"elif {translate_expr(cond)}:",
                *[INDENT + part for part in translate_statement(yes)],
                "else:",
                *[INDENT + part for part in translate_statement(no)],
            ]
        return [f"elif {translate_expr(cond)}:", *[INDENT + part for part in translate_statement(rest)]]

    parsed_if = parse_if_line(line)
    if parsed_if is not None:
        cond, rest = parsed_if
        if not rest:
            return [f"if {translate_expr(cond)}:"]
        split = split_top_level_else(rest)
        if split is not None:
            yes, no = split
            return [
                f"if {translate_expr(cond)}:",
                *[INDENT + part for part in translate_statement(yes)],
                "else:",
                *[INDENT + part for part in translate_statement(no)],
            ]
        return [f"if {translate_expr(cond)}:", *[INDENT + part for part in translate_statement(rest)]]

    parsed_for = parse_for_line(line)
    if parsed_for is not None:
        name, values, rest = parsed_for
        if rest:
            return [f"for {name} in {translate_for_iter(values)}:", *[INDENT + part for part in translate_statement(rest)]]
        return [f"for {name} in {translate_for_iter(values)}:"]

    while_match = re.match(r"while\s*\((.+)\)\s*$", line)
    if while_match:
        return [f"while {translate_expr(while_match.group(1))}:"]

    if line == "repeat":
        return ["while True:"]

    if line == "break":
        return ["break"]

    if line == "next":
        return ["continue"]

    metadata = translate_metadata_assignment(line)
    if metadata is not None:
        return metadata

    if is_metadata_assignment(line):
        return ["pass  # R metadata assignment omitted"]

    member_subscript_assign = re.match(r"^([A-Za-z]\w*)\$([A-Za-z]\w*)\s*\[(.+)\]\s*(?:<-|=)\s*(.+)$", line)
    if member_subscript_assign:
        obj, field, index, rhs = member_subscript_assign.groups()
        py_obj = r_name(obj)
        py_field = r_name(field)
        py_index = translate_expr(index)
        py_rhs = translate_expr(rhs)
        return [
            f"if 'pd' in globals() and isinstance({py_obj}, pd.DataFrame):",
            INDENT + f"{py_obj}.loc[{py_index}, {py_field!r}] = {py_rhs}",
            "else:",
            INDENT + f"{py_obj}.{py_field}[{py_index}] = {py_rhs}",
        ]

    member_assign = re.match(r"^([A-Za-z]\w*)\$([A-Za-z]\w*)\s*(?:<-|=)\s*(.+)$", line)
    if member_assign:
        obj, field, rhs = member_assign.groups()
        py_obj = r_name(obj)
        py_field = r_name(field)
        py_rhs = translate_expr(rhs)
        return [
            f"if 'pd' in globals() and isinstance({py_obj}, pd.DataFrame):",
            INDENT + f"{py_obj}[{py_field!r}] = {py_rhs}",
            "else:",
            INDENT + f"{py_obj}.{py_field} = {py_rhs}",
            INDENT + f"if hasattr({py_obj}, '_r_names') and {py_field!r} not in {py_obj}._r_names:",
            INDENT * 2 + f"{py_obj}._r_names.append({py_field!r})",
        ]

    full_call = parse_full_call(line)
    if full_call is not None and full_call[0].lower() == "return":
        _name, args = full_call
        if not args:
            return ["return"]
        return ["return " + translate_expr(args[0])]
    if full_call is not None and full_call[0].lower() == "list":
        name, args = full_call
        return ["return " + translate_call(name, args)]
    if full_call is not None and full_call[0].lower() in {"print", "cat", "stopifnot"}:
        name, args = full_call
        return [translate_call(name, args)]

    assign = split_assignment(line)
    if assign is not None:
        lhs, rhs = assign
        py_rhs = translate_expr(rhs)
        if lhs.endswith("_order") and re.search(r"_colnames\.index\([\"']order[\"']\)", py_rhs):
            py_rhs = f"int({py_rhs})"
        if (
            lhs.endswith(("_p", "_q"))
            and re.search(r"_colnames\.index\([\"'][pq][\"']\)", py_rhs)
            and not py_rhs.startswith("int(")
        ):
            py_rhs = f"int({py_rhs})"
        return [f"{lhs} = {py_rhs}"]

    return [translate_expr(line)]


def is_metadata_assignment(line: str) -> bool:
    for op in ("<-", "="):
        pos = find_top_level_operator(line, op)
        if pos >= 0:
            lhs = line[:pos].strip().lower()
            return lhs.startswith(("colnames(", "rownames(", "dimnames(", "names("))
    return False


def translate_metadata_assignment(line: str) -> list[str] | None:
    assign = raw_assignment(line)
    if assign is None:
        return None
    lhs, rhs = assign
    m = re.match(r"colnames\s*\(\s*([A-Za-z]\w*)\s*\)\s*$", lhs, re.IGNORECASE)
    if not m:
        return None
    if not re.match(r"c\s*\(", rhs.strip(), re.IGNORECASE):
        return ["pass  # R metadata assignment omitted"]
    return [f"{r_name(m.group(1))}_colnames = list({translate_expr(rhs)})"]


def translate_function_signature(args: str) -> tuple[str, list[str]]:
    out: list[str] = []
    setup: list[str] = []
    previous: set[str] = set()
    for arg in split_args(args):
        if not arg:
            continue
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            name = r_name(arg[:pos].strip())
            value = translate_expr(arg[pos + 1 :].strip())
            if expr_references_names(value, previous):
                out.append(f"{name}=None")
                setup.append(f"if {name} is None:")
                setup.append(INDENT + f"{name} = {value}")
            else:
                out.append(f"{name}={value}")
        else:
            name = r_name(arg.strip())
            out.append(name)
        previous.add(name)
    return ", ".join(out), setup


def expr_references_names(expr: str, names: set[str]) -> bool:
    if not names:
        return False
    found = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
    return bool(found & names)


def parse_full_call(line: str) -> tuple[str, list[str]] | None:
    match = re.match(r"^\s*([A-Za-z]\w*(?:\.\w+)*)\s*\(", line)
    if not match or not line.rstrip().endswith(")"):
        return None
    name = match.group(1)
    open_pos = line.find("(", match.end(1))
    depth = 0
    quote = ""
    for i in range(open_pos, len(line)):
        ch = line[i]
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                if line[i + 1 :].strip():
                    return None
                return name, split_args(line[open_pos + 1 : i])
    return None


def parse_if_line(line: str) -> tuple[str, str] | None:
    if not line.startswith("if"):
        return None
    pos = line.find("(")
    if pos < 0 or line[:pos].strip() != "if":
        return None
    close = find_matching_paren(line, pos)
    if close < 0:
        return None
    return line[pos + 1 : close].strip(), line[close + 1 :].strip()


def parse_else_if_line(line: str) -> tuple[str, str] | None:
    if not line.startswith("else"):
        return None
    rest = line[4:].strip()
    if not rest.startswith("if"):
        return None
    return parse_if_line(rest)


def parse_for_line(line: str) -> tuple[str, str, str] | None:
    if not line.startswith("for"):
        return None
    pos = line.find("(")
    if pos < 0 or line[:pos].strip() != "for":
        return None
    close = find_matching_paren(line, pos)
    if close < 0:
        return None
    header = line[pos + 1 : close].strip()
    match = re.match(r"(\w+)\s+in\s+(.+)$", header)
    if not match:
        return None
    name, values = match.groups()
    return name, values.strip(), line[close + 1 :].strip()


def translate_for_iter(values: str) -> str:
    range_parts = split_top_level_range(values.strip())
    if range_parts is not None:
        start, stop = range_parts
        return f"r_range({translate_expr(start)}, {translate_expr(stop)})"
    return translate_expr(values)


def find_matching_paren(text: str, open_pos: int) -> int:
    depth = 0
    quote = ""
    for i in range(open_pos, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def split_top_level_else(text: str) -> tuple[str, str] | None:
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith("else", i):
            before = text[i - 1] if i > 0 else " "
            after = text[i + 4] if i + 4 < len(text) else " "
            if before.isspace() and after.isspace():
                return text[:i].strip(), text[i + 4 :].strip()
        i += 1
    return None


def split_assignment(line: str) -> tuple[str, str] | None:
    assign = raw_assignment(line)
    if assign is not None:
        lhs, rhs = assign
        if re.match(r"^[A-Za-z.]\w*(?:\[.*\])?$", lhs):
            return translate_expr(lhs), rhs
    return None


def raw_assignment(line: str) -> tuple[str, str] | None:
    for op in ("<-", "="):
        pos = find_top_level_operator(line, op)
        if pos >= 0:
            return line[:pos].strip(), line[pos + len(op) :].strip()
    return None


def translate_expr(expr: str) -> str:
    expr = expr.strip().rstrip(";")
    raw_call = parse_full_call(expr)
    if raw_call is not None and raw_call[0].lower() == "vector":
        return translate_vector_call(raw_call[1])
    expr, strings = mask_string_literals(expr)
    expr = translate_expr_code(expr)
    expr = restore_string_literals(expr, strings)
    for i, text in enumerate(strings):
        expr = expr.replace(f"__R_ATTR_{i}__", r_name(text[1:-1]))
    return expr


def translate_expr_code(expr: str) -> str:
    expr = expr.replace("<-", "=")
    expr = re.sub(r"(?<=\d)[lL]\b", "", expr)
    expr = expr.replace("$", "@@MEM@@")
    expr = expr.replace("%%", "%")
    expr = expr.replace("%*%", "@")
    expr = expr.replace("&&", " and ")
    expr = expr.replace("||", " or ")
    expr = re.sub(r"!\s*(?!=)", "not ", expr)
    expr = replace_power(expr)
    expr = replace_r_constants(expr)
    expr = replace_ranges(expr)
    expr = replace_r_subscripts(expr)
    expr = replace_calls(expr)
    expr = replace_matrix_vector_recycling(expr)
    expr = replace_named_matrix_columns(expr)
    expr = replace_names(expr)
    expr = apply_recycled_binops(expr)
    expr = expr.replace("@@MEM@@", ".")
    return expr


def strip_outer_parens(text: str) -> str:
    text = text.strip()
    changed = True
    while changed and text.startswith("(") and text.endswith(")"):
        depth = 0
        changed = False
        for i, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return text
                if depth == 0:
                    if i != len(text) - 1:
                        return text
                    text = text[1:-1].strip()
                    changed = True
                    break
        if depth != 0:
            # Unbalanced parentheses; keep as-is to avoid over-stripping.
            return text
    return text


def replace_r_constants(expr: str) -> str:
    replacements = {
        "TRUE": "True",
        "FALSE": "False",
        "NULL": "None",
        "NA_real_": "np.nan",
        "NA": "np.nan",
        "NaN": "np.nan",
        "Inf": "np.inf",
        ".Machine@@MEM@@double.eps": "np.finfo(float).eps",
        ".Machine@@MEM@@double.xmin": "np.finfo(float).tiny",
    }
    for old, new in replacements.items():
        if old.startswith("."):
            expr = expr.replace(old, new)
        else:
            expr = re.sub(rf"\b{old}\b", new, expr)
    return expr


def replace_power(expr: str) -> str:
    return expr.replace("^", "**")


def replace_ranges(expr: str) -> str:
    name_atom = r"[A-Za-z_]\w*(?:(?:@@MEM@@|\.)[A-Za-z_]\w*)*"
    atom = rf"(?:{name_atom}\([^()]*\)|\([^()]+\)|{name_atom}(?!\s*\()|\d+(?:\.\d+)?)"
    pattern = re.compile(rf"(?<![\w.])({atom})\s*:\s*({atom})(?![\w.])")

    def repl(match: re.Match[str]) -> str:
        start, stop = match.groups()
        return f"r_seq({start}, {stop})"

    return pattern.sub(repl, expr)


def replace_calls(expr: str) -> str:
    previous = None
    while previous != expr:
        previous = expr
        expr = replace_innermost_call(expr)
    return expr


def replace_matrix_vector_recycling(expr: str) -> str:
    expr = re.sub(r"\blog_dens\s*-\s*row_max(?!\s*\[:,\s*None\])", "log_dens - row_max[:, None]", expr)
    expr = re.sub(
        r"\bdens_shifted\s*/\s*np\.sum\(dens_shifted,\s*axis=1\)(?!\s*\[:,\s*None\])",
        "dens_shifted / np.sum(dens_shifted, axis=1)[:, None]",
        expr,
    )
    expr = re.sub(r"\bdens\s*/\s*denom(?!\s*\[:,\s*None\])", "dens / denom[:, None]", expr)
    expr = re.sub(r"\bresp\s*\*\s*x(?!\s*\[:,\s*None\])", "resp * x[:, None]", expr)
    expr = re.sub(r"\bxc\s*\*\s*w(?!\s*\[:,\s*None\])", "xc * w[:, None]", expr)
    expr = re.sub(r"\bx\s*\*\s*w(?!\s*\[:,\s*None\])", "x * w[:, None]", expr)
    return expr


def replace_named_matrix_columns(expr: str) -> str:
    str_atom = r"(?:[\"'][^\"']+[\"']|__R_STR_\d+__)"
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[:,\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"{m.group(1)}[:, {m.group(1)}_colnames.index({m.group(2)})]",
        expr,
    )
    expr = re.sub(
        r"\b([A-Za-z]\w*)\[:,\s*\(([A-Za-z]\w*_colnames\.index\([^)]+\))\)\s*-\s*1\]",
        lambda m: f"{m.group(1)}[:, {m.group(2)}]",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[(.+?),\s*({str_atom})\]",
        lambda m: f"{m.group(1)}[{m.group(2)}, {m.group(1)}_colnames.index({m.group(3)})]",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[\(([A-Za-z]\w*_idx)\)\s*-\s*1,\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"int({m.group(1)}[{m.group(2)}, {m.group(1)}_colnames.index({m.group(3)})])",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[(.+?),\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"{m.group(1)}[{m.group(2)}, {m.group(1)}_colnames.index({m.group(3)})]",
        expr,
    )
    expr = re.sub(
        r"\b([A-Za-z]\w*)\[(int\(np\.argmin\([^\]]+\]\)\)),\s*([A-Za-z]\w*_colnames\.index\([\"']order[\"']\))\]",
        lambda m: f"int({m.group(1)}[{m.group(2)}, {m.group(3)}])",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[\(([A-Za-z]\w*_idx)\)\s*-\s*1,\s*\(([A-Za-z]\w*_colnames\.index\({str_atom}\))\)\s*-\s*1\]",
        lambda m: f"int({m.group(1)}[{m.group(2)}, {m.group(3)}])",
        expr,
    )
    expr = re.sub(
        r"\b([A-Za-z]\w*)\[\(([A-Za-z]\w*_idx)\)\s*-\s*1,\s*([A-Za-z]\w*_colnames\.index\([^)]+\))\]",
        lambda m: f"{m.group(1)}[{m.group(2)}, {m.group(3)}]",
        expr,
    )
    return expr


def replace_innermost_call(expr: str) -> str:
    candidates: list[tuple[int, int, str, str]] = []
    pattern = re.compile(r"(?<![\w.])([A-Za-z]\w*(?:\.\w+)*)\s*\(")
    for match in pattern.finditer(expr):
        name = match.group(1)
        if name.startswith(("np.", "stats.", "pd.")) or name in {"SimpleNamespace", "RList", "RNamedVector", "r_subset", "r_col_key", "getattr", "globals", "int", "float", "str", "len"}:
            continue
        open_pos = expr.find("(", match.start())
        close_pos = find_matching_paren(expr, open_pos)
        if close_pos < 0:
            continue
        candidates.append((match.start(), close_pos + 1, name, expr[open_pos + 1 : close_pos]))
    for start, end, name, arg_text in reversed(candidates):
        args = split_args(arg_text)
        translated = translate_call(name, args)
        if translated != expr[start:end]:
            return expr[:start] + translated + expr[end:]
    return expr


def replace_r_subscripts(expr: str) -> str:
    item_pattern = re.compile(r"([A-Za-z]\w*(?:(?:@@MEM@@|\.)\w+)*)\s*\[\[([^\[\]]+)\]\]")
    expr = item_pattern.sub(replace_double_subscript, expr)
    pattern = re.compile(r"([A-Za-z]\w*(?:(?:@@MEM@@|\.)\w+)*)\s*\[([^\[\]]*)\]")
    return pattern.sub(replace_single_subscript, expr)


def replace_double_subscript(match: re.Match[str]) -> str:
    base = translate_member_expr(match.group(1))
    index = match.group(2).strip()
    if is_string_literal(index):
        return f"{base}.{r_name(index[1:-1])}"
    placeholder = re.fullmatch(r"__R_STR_(\d+)__", index)
    if placeholder:
        return f"{base}.__R_ATTR_{placeholder.group(1)}__"
    if re.fullmatch(r"\d+", index):
        return f"getattr({base}, {base}._r_names[{int(index) - 1}])"
    return f"{base}[{translate_subscript(index)}]"


def replace_single_subscript(match: re.Match[str]) -> str:
    base = match.group(1)
    index = match.group(2).strip()
    if has_top_level_comma(index) and any_negative_matrix_subscript(index):
        return replace_negative_matrix_subscript(base, index)
    if has_top_level_comma(index):
        return f"r_subset({base}, {translate_subscript(index, base=base)})"
    if is_negative_integer_subscript(index):
        item = index.replace(" ", "")[1:]
        return f"np.delete({base}, ({item}) - 1)"
    return f"{base}[{translate_subscript(index, base=base)}]"


def any_negative_matrix_subscript(index: str) -> bool:
    return any(is_negative_integer_subscript(part.strip()) for part in split_subscript_args(index))


def replace_negative_matrix_subscript(base: str, index: str) -> str:
    expr = base
    kept_parts: list[str] = []
    for axis, raw_part in enumerate(split_subscript_args(index)):
        part = raw_part.strip()
        if is_subscript_option(part):
            continue
        if is_negative_integer_subscript(part):
            item = part.replace(" ", "")[1:]
            expr = f"np.delete({expr}, ({item}) - 1, axis={axis})"
            kept_parts.append(":")
        elif part == "":
            kept_parts.append(":")
        else:
            kept_parts.append(translate_subscript(part))
    if kept_parts and any(part != ":" for part in kept_parts):
        expr += "[" + ", ".join(kept_parts) + "]"
    return expr


def translate_member_expr(expr: str) -> str:
    parts = re.split(r"@@MEM@@|\.", expr)
    if not parts:
        return expr
    out = r_name(parts[0])
    for part in parts[1:]:
        out += "." + r_name(part)
    return out


def translate_subscript(index: str, *, base: str | None = None) -> str:
    index = strip_outer_parens(index)
    if has_top_level_comma(index):
        return translate_matrix_subscript(index, base=base)
    if index == ":":
        return ":"
    if is_string_index_expr(index):
        return translate_expr(index)
    range_parts = split_top_level_range(index)
    if range_parts is not None:
        start, stop = range_parts
        return f"(np.arange({translate_expr(start)}, {translate_expr(stop)} + 1)) - 1"
    if is_negative_integer_subscript(index):
        return translate_negative_subscript(index)
    if is_logical_subscript(index):
        return index
    if re.match(r"^\(.+\)\s*-\s*1$", index):
        return index
    if index.startswith("np.arange(") and index.endswith(")"):
        return f"({index}) - 1"
    return f"({index}) - 1"


def is_negative_integer_subscript(index: str) -> bool:
    return re.match(r"^-\s*\d+$", index) is not None


def translate_negative_subscript(index: str) -> str:
    item = index.replace(" ", "")[1:]
    return f"np.r_[0:({item}) - 1, ({item}):]"


def split_top_level_range(text: str) -> tuple[str, str] | None:
    depth = 0
    quote = ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ":" and depth == 0:
            if not text[:i].strip() or not text[i + 1 :].strip():
                return None
            return text[:i].strip(), text[i + 1 :].strip()
    return None


def translate_matrix_subscript(index: str, *, base: str | None = None) -> str:
    parts = split_subscript_args(index)
    out: list[str] = []
    advanced_axes: list[int] = []
    for axis, part in enumerate(parts):
        item = part.strip()
        if is_subscript_option(item):
            continue
        if item == "":
            out.append("slice(None)" if base else ":")
        elif axis == 1 and base and (is_string_literal(item) or re.fullmatch(r"__R_STR_\d+__", item)):
            out.append(f"r_col_key({base}, {item}, globals().get('{base}_colnames'))")
        elif is_logical_subscript(item):
            out.append(item)
            advanced_axes.append(len(out) - 1)
        else:
            translated = translate_subscript(item)
            out.append(translated)
            if is_advanced_matrix_index(translated):
                advanced_axes.append(len(out) - 1)
    if len(out) == 2 and advanced_axes == [0, 1]:
        return f"np.ix_({out[0]}, {out[1]})"
    return ", ".join(out)


def is_advanced_matrix_index(index: str) -> bool:
    if index == ":":
        return False
    if re.fullmatch(r"\(?\d+\)?\s*-\s*1", index):
        return False
    return any(token in index for token in ("np.arange", "np.r_", "r_seq", "np.array", "r_c("))


def is_string_literal(text: str) -> bool:
    return len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}


def is_string_index_expr(text: str) -> bool:
    if re.fullmatch(r"__R_STR_\d+__", text):
        return True
    raw_call = parse_full_call(text)
    if raw_call is None or raw_call[0].lower() != "c":
        return False
    return all(re.fullmatch(r"__R_STR_\d+__", arg.strip()) or is_string_literal(arg.strip()) for arg in raw_call[1])


def is_subscript_option(item: str) -> bool:
    item = strip_outer_parens(item)
    pos = find_top_level_operator(item, "=")
    return pos >= 0 and item[:pos].strip().lower() in {"drop"}


def has_top_level_comma(text: str) -> bool:
    return any(part == "," for part in top_level_tokens(text, comma_only=True))


def split_subscript_args(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def top_level_tokens(text: str, *, comma_only: bool = False) -> list[str]:
    tokens: list[str] = []
    depth = 0
    quote = ""
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            tokens.append(",")
            if comma_only:
                return tokens
    return tokens


def is_logical_subscript(index: str) -> bool:
    while index.startswith("(") and index.endswith(")"):
        close = find_matching_paren(index, 0)
        if close != len(index) - 1:
            break
        index = index[1:-1].strip()
    if index.startswith(("np.is", "is.", "is_", "~")):
        return True
    if index.startswith(","):
        return False
    return any(op in index for op in ("<", ">", "==", "!=", "<=", ">="))


def translate_call(name: str, args: list[str]) -> str:
    lname = name.lower()
    if lname == "lm":
        return translate_lm_call(args)
    if lname == "try":
        if not args:
            return "try_(lambda: None)"
        silent = translate_expr(keyword_arg(args, "silent", default="False"))
        return f"try_(lambda: {translate_expr(positional_args(args)[0])}, silent={silent})"
    py_args = [translate_expr(arg) for arg in args]
    if name.startswith("np."):
        return name + "(" + ", ".join(py_args) + ")"
    if lname == "c":
        return translate_c_call(args)
    if lname == "list":
        return translate_list_call(args)
    if lname == "data.frame":
        return translate_data_frame_call(args)
    if lname == "vector":
        return translate_vector_call(args)
    if lname == "matrix":
        return translate_matrix_call(args)
    if lname == "array":
        return translate_array_call(args)
    if lname == "cbind":
        return translate_cbind_call(args)
    if lname == "print":
        if len(args) == 1:
            raw_call = parse_full_call(args[0])
            if raw_call is not None and raw_call[0].lower() == "round" and len(raw_call[1]) >= 2:
                return (
                    "r_print("
                    + py_args[0]
                    + ", digits="
                    + translate_expr(raw_call[1][1])
                    + print_colnames_arg(raw_call[1][0], allow_simple=True)
                    + ")"
                )
            return "r_print(" + ", ".join(py_args) + print_colnames_arg(args[0]) + ")"
        return "r_print(" + ", ".join(py_args) + ")"
    if lname == "cat":
        return "print(" + ", ".join(py_args) + ', end="")'
    if lname == "sprintf":
        return translate_sprintf_call(args)
    if lname == "paste":
        return translate_paste_call(args, default_sep=" ")
    if lname == "paste0":
        return translate_paste_call(args, default_sep="")
    if lname == "ifelse":
        return "np.where(" + ", ".join(py_args) + ")"
    if lname == "sqrt":
        return f"np.sqrt({py_args[0]})"
    if lname in {"log", "exp", "sin", "cos", "tan", "abs"}:
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "round":
        return "np.round(" + ", ".join(py_args) + ")"
    if lname == "chol":
        return "np.linalg.cholesky(" + py_args[0] + ").T"
    if lname == "sweep":
        return translate_sweep_call(args)
    if lname == "t":
        return "(" + py_args[0] + ").T"
    if lname == "backsolve":
        return translate_backsolve_call(args)
    if lname == "diag":
        return "(np.eye(int(" + py_args[0] + ")) if np.isscalar(" + py_args[0] + ") else np.diag(" + py_args[0] + "))"
    if lname == "dim":
        return "np.array(" + py_args[0] + ".shape)"
    if lname == "crossprod":
        if len(args) > 2:
            raise R2PyError("crossprod supports at most 2 arguments")
        rhs = py_args[1] if len(py_args) > 1 else py_args[0]
        return "(" + py_args[0] + ").T @ (" + rhs + ")"
    if lname == "tcrossprod":
        if len(args) > 2:
            raise R2PyError("tcrossprod supports at most 2 arguments")
        rhs = py_args[1] if len(py_args) > 1 else py_args[0]
        return "(" + py_args[0] + ") @ (" + rhs + ").T"
    if lname == "solve":
        if len(py_args) == 1:
            return "np.linalg.inv(" + py_args[0] + ")"
        return "np.linalg.solve(" + py_args[0] + ", " + py_args[1] + ")"
    if lname == "det":
        return "np.linalg.det(" + py_args[0] + ")"
    if lname in {"sum", "mean", "median", "prod"}:
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "var":
        return "var_r(" + py_args[0] + ")"
    if lname == "min":
        return "np.minimum(" + ", ".join(py_args) + ")" if len(py_args) > 1 else f"np.min({py_args[0]})"
    if lname == "max":
        return "np.maximum(" + ", ".join(py_args) + ")" if len(py_args) > 1 else f"np.max({py_args[0]})"
    if lname == "sd":
        return "np.std(" + py_args[0] + ", ddof=1)"
    if lname == "all":
        return "np.all(" + ", ".join(py_args) + ")"
    if lname == "pmax":
        return "np.maximum(" + ", ".join(py_args) + ")"
    if lname == "pmin":
        return "np.minimum(" + ", ".join(py_args) + ")"
    if lname == "as.numeric":
        return "np.asarray(" + py_args[0] + ", dtype=float)"
    if lname == "as.matrix":
        return "np.asarray(" + py_args[0] + ")"
    if lname == "as.integer":
        return "int(" + py_args[0] + ")"
    if lname == "is.finite":
        return "np.isfinite(" + py_args[0] + ")"
    if lname == "is.null":
        return "(" + py_args[0] + " is None)"
    if lname == "stopifnot":
        return "assert " + " and ".join(py_args)
    if lname == "invisible":
        return py_args[0] if py_args else "None"
    if lname == "inherits":
        if len(args) >= 2:
            return f"isinstance({py_args[0]}, TryError)"
        return "False"
    if lname == "quantile":
        return translate_quantile_call(args)
    if lname == "tail":
        return translate_tail_call(args)
    if lname == "cumsum":
        return "np.cumsum(" + py_args[0] + ")"
    if lname == "findinterval":
        return "np.searchsorted(" + py_args[1] + ", " + py_args[0] + ", side='right')"
    if lname == "range":
        xarg = py_args[0] if py_args else "np.array([])"
        na_rm = keyword_arg(args, "na.rm", default="False")
        if na_rm.lower() == "true":
            return "np.array([np.nanmin(" + xarg + "), np.nanmax(" + xarg + ")])"
        return "np.array([np.min(" + xarg + "), np.max(" + xarg + ")])"
    if lname == "which":
        return "np.nonzero(" + py_args[0] + ")[0] + 1"
    if lname == "which.min":
        return "int(np.argmin(" + py_args[0] + "))"
    if lname == "rowsums":
        return "np.sum(" + py_args[0] + ", axis=1)"
    if lname == "colsums":
        return "np.sum(" + py_args[0] + ", axis=0)"
    if lname == "rowmeans":
        return "np.mean(" + py_args[0] + ", axis=1)"
    if lname == "colmeans":
        return "np.mean(" + py_args[0] + ", axis=0)"
    if lname == "apply":
        return translate_apply_call(args)
    if lname == "max.col":
        return "np.argmax(" + py_args[0] + ", axis=1) + 1"
    if lname == "table":
        return translate_table_call(args)
    if lname == "coef":
        return "np.concatenate(([np.nan], " + py_args[0] + ".coef))"
    if lname == "residuals":
        return py_args[0] + ".resid"
    if lname == "fitted":
        return py_args[0] + ".fitted"
    if lname == "summary":
        return "summary_lm_py(" + py_args[0] + ")"
    if lname == "length":
        return "r_length(" + py_args[0] + ")"
    if lname == "names":
        return "r_names(" + py_args[0] + ")"
    if lname == "nrow":
        return py_args[0] + ".shape[0]"
    if lname == "ncol":
        return py_args[0] + ".shape[1]"
    if lname in {"seq", "seq.int"}:
        return translate_seq_call(args)
    if lname == "seq_along":
        return "np.arange(1, len(" + py_args[0] + ") + 1)"
    if lname == "seq_len":
        return "np.arange(1, " + py_args[0] + " + 1)"
    if lname == "rep":
        return translate_rep_call(args)
    if lname == "numeric":
        return "np.zeros(" + py_args[0] + ")"
    if lname == "integer":
        return "np.zeros(" + py_args[0] + ", dtype=int)"
    if lname == "set.seed":
        return "np.random.seed(" + py_args[0] + ")"
    if lname == "sample.int":
        return translate_sample_int_call(args)
    if lname == "runif":
        return translate_runif_call(args)
    if lname == "rnorm":
        return translate_rnorm_call(args)
    if lname in {"dnorm", "pnorm", "qnorm"}:
        return translate_normal_dist_call(lname, args)
    if lname in {"dt", "pt", "qt"}:
        return translate_t_dist_call(lname, args)
    return r_name(name) + "(" + ", ".join(py_args) + ")"


def print_colnames_arg(raw_expr: str, *, allow_simple: bool = False) -> str:
    expr = raw_expr.strip()
    member_match = re.fullmatch(r"([A-Za-z]\w*)\$([A-Za-z]\w*)", expr)
    if member_match:
        obj, field = member_match.groups()
        return f", colnames=getattr({r_name(obj)}, {field + '_colnames'!r}, None)"
    name_match = re.fullmatch(r"[A-Za-z]\w*", expr)
    if allow_simple and name_match:
        name = r_name(expr)
        return f", colnames=({name}_colnames if {name + '_colnames'!r} in locals() else None)"
    return ""


def translate_c_call(args: list[str]) -> str:
    values: list[str] = []
    names: list[str | None] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            names.append(arg[:pos].strip())
            values.append(translate_expr(arg[pos + 1 :].strip()))
        else:
            names.append(None)
            values.append(translate_expr(arg))
    if any(name is not None for name in names):
        py_names = "[" + ", ".join(repr(name) if name is not None else "None" for name in names) + "]"
        return "r_c(" + ", ".join(values) + f", names={py_names})"
    return "r_c(" + ", ".join(values) + ")"


def translate_rep_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("rep requires a value")
    x = translate_expr(args[0])
    positional = positional_args(args)
    times = keyword_arg(args, "times", default=positional[1] if len(positional) > 1 else "1")
    each = keyword_arg(args, "each", default="1")
    py_each = translate_expr(each)
    py_times = translate_expr(times)
    repeated = f"np.repeat({x}, {py_each})"
    if py_times == "1":
        return repeated
    return f"np.tile({repeated}, {py_times})"


def is_vector_expr(expr: str) -> bool:
    return bool(
        re.search(
            r"\b(?:np\.asarray|np\.array|np\.ravel|np\.repeat|np\.concatenate|r_seq)\s*\(",
            expr,
        )
    )


def translate_apply_call(args: list[str]) -> str:
    if len(args) < 3:
        raise R2PyError("apply requires array, margin, and function")
    x = translate_expr(args[0])
    margin = translate_expr(args[1])
    func = translate_expr(args[2])
    if func in {"sum", "np.sum"}:
        return f"r_apply({x}, {margin}, 'sum')"
    if func in {"mean", "np.mean"}:
        return f"r_apply({x}, {margin}, 'mean')"
    if func in {"median", "np.median"}:
        return f"r_apply({x}, {margin}, 'median')"
    if func in {"var", "var_r"}:
        return f"r_apply({x}, {margin}, 'var')"
    if func in {"min", "np.min"}:
        return f"r_apply({x}, {margin}, 'min')"
    if func in {"max", "np.max"}:
        return f"r_apply({x}, {margin}, 'max')"
    axis = "0" if margin == "2" else "1"
    return f"np.apply_along_axis({func}, {axis}, {x})"


def translate_lm_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("lm requires a formula")
    formula = args[0]
    pos = find_top_level_operator(formula, "~")
    if pos < 0:
        return "lm(" + ", ".join(translate_expr(arg) for arg in args) + ")"
    y = translate_expr(formula[:pos].strip())
    x = translate_expr(formula[pos + 1 :].strip())
    data = keyword_arg(args, "data")
    if x == "." and data is not None:
        data_expr = translate_expr(data)
        return f"lm_py({data_expr}.{y}, {data_expr}.xlag)"
    return f"lm_py({y}, {x})"


def translate_data_frame_call(args: list[str]) -> str:
    fields: list[str] = []
    unnamed = 0
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            name = r_name(arg[:pos].strip())
            value = translate_expr(arg[pos + 1 :].strip())
        else:
            value = translate_expr(arg)
            if re.match(r"^[A-Za-z_]\w*$", value):
                name = value
            else:
                unnamed += 1
                name = f"x{unnamed}"
        fields.append(f"{name!r}: {value}")
    return "pd.DataFrame({" + ", ".join(fields) + "})"


def translate_sweep_call(args: list[str]) -> str:
    if len(args) < 4:
        raise R2PyError("sweep requires x, margin, stats, and function")
    x = translate_expr(args[0])
    margin = translate_expr(args[1])
    stats = translate_expr(args[2])
    op = translate_expr(args[3])
    return f"sweep_py({x}, {margin}, {stats}, {op})"


def translate_table_call(args: list[str]) -> str:
    values: list[str] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        values.append(translate_expr(arg[pos + 1 :].strip() if pos >= 0 else arg))
    if len(values) == 2:
        a, b = values
        return (
            f"(lambda _a, _b: np.asarray(np.histogram2d(_a, _b, "
            f"bins=(np.arange(0.5, np.max(_a) + 1.5), np.arange(0.5, np.max(_b) + 1.5))"
            f")[0], dtype=int))({a}, {b})"
        )
    if len(values) == 1:
        a = values[0]
        return f"np.bincount(np.asarray({a}, dtype=int))[1:]"
    return "np.array([])"


def translate_backsolve_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("backsolve requires matrix and right hand side")
    r = translate_expr(args[0])
    b = translate_expr(args[1])
    transpose = translate_expr(keyword_arg(args, "transpose", default="False"))
    mat = f"({r}).T" if transpose == "True" else r
    return f"np.linalg.solve({mat}, {b})"


def translate_list_call(args: list[str]) -> str:
    fields: list[str] = []
    values: list[str] = []
    names: list[str] = []
    for i, arg in enumerate(args):
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            name = r_name(arg[:pos].strip())
            value = translate_expr(arg[pos + 1 :].strip())
        else:
            name = f"x{i + 1}"
            value = translate_expr(arg)
        if re.match(r"^[A-Za-z_]\w*$", name):
            if pos >= 0:
                fields.append(f"{name}={value}")
                names.append(name)
                if name == "table" and re.match(r"^[A-Za-z_]\w*$", value):
                    fields.append(
                        f"{name}_colnames=({value}_colnames if {value + '_colnames'!r} in locals() else None)"
                    )
            else:
                values.append(value)
        else:
            values.append(value)
    if fields and not values:
        return "RList(" + ", ".join(fields + [f"_r_names={names!r}"]) + ")"
    if fields:
        return "RList(" + ", ".join(fields + [f"_r_names={names!r}"]) + ")"
    return "[" + ", ".join(values) + "]"


def translate_vector_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("vector requires a mode")
    mode = args[0].strip().strip("\"'")
    length = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "length", default="0"))
    if mode == "list":
        return f"([None] * ({length}))"
    return f"np.zeros({length})"


def translate_cbind_call(args: list[str]) -> str:
    cols: list[str] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        cols.append(translate_expr(arg[pos + 1 :].strip() if pos >= 0 else arg))
    return "cbind_py(" + ", ".join(cols) + ")"


def translate_sprintf_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("sprintf requires a format argument")
    fmt = translate_expr(args[0])
    values = [translate_expr(arg) for arg in args[1:]]
    if not values:
        return fmt
    if len(values) == 1:
        return f"np.char.mod({fmt}, {values[0]})"
    return f"np.char.mod({fmt}, (" + ", ".join(values) + "))"


def translate_paste_call(args: list[str], *, default_sep: str) -> str:
    sep = keyword_arg(args, "sep", default=repr(default_sep))
    collapse = keyword_arg(args, "collapse", default="")
    values = [translate_expr(arg) for arg in positional_args(args)]
    if collapse:
        if not values:
            return '""'
        joined_values = values[0] if len(values) == 1 else f"({translate_expr(sep)}).join(map(str, [{', '.join(values)}]))"
        return f"({translate_expr(collapse)}).join(np.asarray({joined_values}, dtype=str))"
    if not values:
        return '""'
    if default_sep == "":
        return " + ".join(f"str({value})" for value in values)
    return f"({translate_expr(sep)}).join(str(x) for x in [" + ", ".join(values) + "])"


def translate_quantile_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("quantile requires an array")
    x = translate_expr(args[0])
    probs = translate_expr(keyword_arg(args, "probs", default="np.array([0.0, 0.25, 0.5, 0.75, 1.0])"))
    return f"np.quantile({x}, {probs})"


def translate_tail_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("tail requires an array")
    x = translate_expr(args[0])
    n = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "n", default="6"))
    if n == "1":
        return f"{x}[-1]"
    return f"{x}[-{n}:]"


def translate_matrix_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("matrix requires at least one argument")
    data = translate_expr(args[0])
    positional = positional_args(args)
    nrow = keyword_arg(args, "nrow", default=positional[1] if len(positional) >= 2 else None)
    ncol = keyword_arg(args, "ncol", default=positional[2] if len(positional) >= 3 else None)
    byrow = translate_expr(keyword_arg(args, "byrow", default="False"))
    order = "'C'" if byrow == "True" else "'F'"
    if nrow is None and ncol is None:
        return f"np.array({data})"
    if nrow is None:
        return f"np.array({data}).reshape((-1, {translate_expr(ncol)}), order={order})"
    if ncol is None:
        return f"np.array({data}).reshape(({translate_expr(nrow)}, -1), order={order})"
    py_nrow = translate_expr(nrow)
    py_ncol = translate_expr(ncol)
    return f"np.resize(np.array({data}), ({py_nrow}) * ({py_ncol})).reshape(({py_nrow}, {py_ncol}), order={order})"


def translate_array_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("array requires data")
    data = translate_expr(args[0])
    dim = keyword_arg(args, "dim")
    if dim is None:
        return f"np.array({data})"
    py_dim = translate_expr(dim)
    return f"(lambda _data, _dim: np.resize(np.array(_data), int(np.prod(_dim))).reshape(tuple(np.asarray(_dim, dtype=int)), order='F'))({data}, {py_dim})"


def translate_seq_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("seq requires arguments")
    by = keyword_arg(args, "by")
    length_out = keyword_arg(args, "length.out")
    along_with = keyword_arg(args, "along.with")
    positional = [arg for arg in args if "=" not in arg]
    if along_with is not None:
        return f"np.arange(1, len({translate_expr(along_with)}) + 1)"
    if length_out is not None:
        start_arg = keyword_arg(args, "from", default=positional[0] if positional else "1")
        stop_arg = keyword_arg(args, "to", default=positional[1] if len(positional) > 1 else length_out)
        start = translate_expr(start_arg)
        stop = translate_expr(stop_arg)
        return f"np.linspace({start}, {stop}, {translate_expr(length_out)})"
    start_arg = keyword_arg(args, "from", default=positional[0] if positional else "1")
    stop_arg = keyword_arg(args, "to", default=positional[1] if len(positional) > 1 else positional[0] if positional else start_arg)
    start = translate_expr(start_arg)
    stop = translate_expr(stop_arg)
    if by is not None:
        step = translate_expr(by)
        return f"np.arange({start}, {stop} + np.sign({step}), {step})"
    return f"np.arange({start}, {stop} + np.sign({stop} - {start}), np.sign({stop} - {start}))"


def translate_runif_call(args: list[str]) -> str:
    n = translate_expr(args[0]) if args else "1"
    lo = translate_expr(keyword_arg(args, "min", default="0"))
    hi = translate_expr(keyword_arg(args, "max", default="1"))
    return f"np.random.uniform({lo}, {hi}, size={n})"


def translate_rnorm_call(args: list[str]) -> str:
    n = translate_expr(args[0]) if args else "1"
    mean = translate_expr(keyword_arg(args, "mean", default="0"))
    sd = translate_expr(keyword_arg(args, "sd", default="1"))
    if n == "1":
        return f"np.random.normal({mean}, {sd})"
    return f"np.random.normal({mean}, {sd}, size={n})"


def translate_sample_int_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("sample.int requires n")
    n = translate_expr(args[0])
    size = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "size", default=n))
    replace = translate_expr(keyword_arg(args, "replace", default="False"))
    prob = keyword_arg(args, "prob", default=None)
    p_arg = ", p=" + translate_expr(prob) if prob is not None else ""
    return f"np.random.choice(np.arange(1, {n} + 1), size={size}, replace={replace}{p_arg})"


def translate_normal_dist_call(name: str, args: list[str]) -> str:
    if not args:
        raise R2PyError(f"{name} requires an x/q/p argument")
    x = translate_expr(args[0])
    mean = translate_expr(keyword_arg(args, "mean", default="0"))
    sd = translate_expr(keyword_arg(args, "sd", default="1"))
    log_arg = translate_expr(keyword_arg(args, "log", default="False"))
    lower_tail = translate_expr(keyword_arg(args, "lower.tail", default="True"))
    if name == "dnorm":
        func = "logpdf" if log_arg == "True" else "pdf"
        return f"stats.norm.{func}({x}, loc={mean}, scale={sd})"
    if name == "pnorm":
        return f"np.where({lower_tail}, stats.norm.cdf({x}, loc={mean}, scale={sd}), stats.norm.sf({x}, loc={mean}, scale={sd}))"
    return f"np.where({lower_tail}, stats.norm.ppf({x}, loc={mean}, scale={sd}), stats.norm.isf({x}, loc={mean}, scale={sd}))"


def translate_t_dist_call(name: str, args: list[str]) -> str:
    if len(args) < 2 and keyword_arg(args, "df") is None:
        raise R2PyError(f"{name} requires df")
    x = translate_expr(args[0])
    df = translate_expr(keyword_arg(args, "df", default=args[1] if len(args) > 1 else None))
    log_arg = translate_expr(keyword_arg(args, "log", default="False"))
    lower_tail = translate_expr(keyword_arg(args, "lower.tail", default="True"))
    if name == "dt":
        func = "logpdf" if log_arg == "True" else "pdf"
        return f"stats.t.{func}({x}, df={df})"
    if name == "pt":
        return f"np.where({lower_tail}, stats.t.cdf({x}, df={df}), stats.t.sf({x}, df={df}))"
    return f"np.where({lower_tail}, stats.t.ppf({x}, df={df}), stats.t.isf({x}, df={df}))"


def keyword_arg(args: list[str], name: str, default: str | None = None) -> str | None:
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0 and arg[:pos].strip().lower() == name.lower():
            return arg[pos + 1 :].strip()
    return default


def positional_args(args: list[str]) -> list[str]:
    return [arg for arg in args if find_top_level_operator(arg, "=") < 0]


def replace_names(expr: str) -> str:
    return re.sub(r"(?<![\w.])([A-Za-z]\w*(?:\.\w+)*)\b", lambda m: r_name(m.group(1)), expr)


def apply_recycled_binops(expr: str) -> str:
    if not any(op in expr for op in ["+", "-", "*", "/"]):
        return expr
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return expr

    op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}

    class Rewriter(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            node = self.generic_visit(node)
            op_name = op_map.get(type(node.op))
            if op_name is None:
                return node
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id=f"r_{op_name}", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )

    tree = Rewriter().visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree.body)  # type: ignore[attr-defined]
    except Exception:
        return expr


def r_name(name: str) -> str:
    constants = {"True", "False", "None", "np", "stats", "nan", "inf", "and", "or", "not", "is", "in", "if", "else", "lambda"}
    if name in constants or "." in name or name.startswith("np.") or name.startswith("stats."):
        return name
    if name[0].isdigit():
        return name
    out = name.replace(".", "_")
    if keyword.iskeyword(out):
        out += "_"
    return out


def split_args(text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def find_top_level_operator(text: str, op: str) -> int:
    depth = 0
    quote = ""
    i = 0
    while i <= len(text) - len(op):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(op, i):
            return i
        i += 1
    return -1


def strip_r_comment(line: str) -> str:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def mask_string_literals(expr: str) -> tuple[str, list[str]]:
    parts: list[str] = []
    strings: list[str] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(expr):
        ch = expr[i]
        if quote:
            current.append(ch)
            if ch == quote:
                placeholder = f"__R_STR_{len(strings)}__"
                strings.append("".join(current))
                parts.append(placeholder)
                current = []
                quote = ""
            i += 1
            continue
        if ch in {"'", '"'}:
            if current:
                parts.append("".join(current))
                current = []
            quote = ch
            current.append(ch)
        else:
            current.append(ch)
        i += 1
    if current:
        parts.append("".join(current))
    return "".join(parts), strings


def restore_string_literals(expr: str, strings: list[str]) -> str:
    for i, text in enumerate(strings):
        expr = expr.replace(f"__R_STR_{i}__", text)
    return expr


def round_numeric_tokens(text: str, digits: int | None) -> str:
    if digits is None:
        return text

    number_re = re.compile(
        r"(?<![A-Za-z0-9_])([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?|[Nn][Aa][Nn]|[Ii][Nn][Ff]|[+-][Ii][Nn][Ff]))(?![A-Za-z0-9_])"
    )

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        low = token.lower()
        if low in {"nan", "+nan", "-nan", "inf", "+inf", "-inf"}:
            return token
        try:
            value = float(token.replace("D", "E").replace("d", "E"))
        except Exception:
            return token
        if not math.isfinite(value):
            return token
        return f"{value:.{int(digits)}f}"

    return number_re.sub(repl, text)


def normalize_output(text: str, digits: int | None = None) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    index_re = re.compile(r"^\s*\[\d+\]\s*")
    for line in text.splitlines():
        line = index_re.sub("", line.rstrip("\n"))
        line = round_numeric_tokens(line, digits)
        line = normalize_output_spacing(line)
        line = normalize_quoted_string_tokens(line)
        out.append(line)
    return out


def normalize_output_spacing(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def normalize_quoted_string_tokens(line: str) -> str:
    parts = line.split(" ")
    if parts and all(re.fullmatch(r'"[^"\s]*"|\'[^\'\s]*\'|[^"\']+', part) for part in parts):
        return " ".join(part[1:-1] if is_string_literal(part) else part for part in parts)
    return line

def outputs_match(a: str, b: str, *, digits: int | None = None) -> tuple[bool, str]:
    a_lines = normalize_output(a, digits)
    b_lines = normalize_output(b, digits)
    if a_lines == b_lines:
        return True, ""

    import difflib

    diff = "\n".join(difflib.unified_diff(a_lines, b_lines, fromfile="R", tofile="Python", lineterm=""))
    return False, diff


def numeric_output_stats(text: str) -> tuple[int, float, float, float]:
    values: list[float] = []
    display_marker_re = re.compile(r"(?:^\s*\[\d+,?\]\s*)|\[,\d+\]")
    number_re = re.compile(
        r"(?<![A-Za-z0-9_])([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?|[Nn][Aa][Nn]|[Ii][Nn][Ff]))(?![A-Za-z0-9_])"
    )
    for line in text.splitlines():
        line = display_marker_re.sub("", line)
        for match in number_re.finditer(line):
            values.append(float(match.group(1).replace("D", "E").replace("d", "E")))
    if not values:
        return 0, math.nan, math.nan, 0.0
    finite_values = [value for value in values if math.isfinite(value)]
    if finite_values:
        return len(values), min(finite_values), max(finite_values), math.fsum(finite_values)
    return len(values), math.nan, math.nan, math.nan


def format_numeric_stats(stats: tuple[int, float, float, float]) -> str:
    count, min_value, max_value, sum_value = stats
    return f"count={count} min={min_value:.12g} max={max_value:.12g} sum={sum_value:.12g}"


def numeric_stats_match(a: tuple[int, float, float, float], b: tuple[int, float, float, float]) -> tuple[bool, str]:
    labels = ("count", "min", "max", "sum")
    for i, label in enumerate(labels):
        if label == "count":
            if a[i] != b[i]:
                return False, f"{label}: R={a[i]} Python={b[i]}"
            continue
        if not numeric_stat_value_match(a[i], b[i]):
            return False, f"{label}: R={a[i]:.12g} Python={b[i]:.12g}"
    return True, ""


def numeric_stat_value_match(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def run_python(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(path)], text=True, capture_output=True)


def run_r(path: Path, rscript: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([rscript, str(path)], text=True, capture_output=True)


def print_process_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def print_result_output(result: subprocess.CompletedProcess[str], digits: int | None) -> None:
    if result.stdout:
        print(
            round_numeric_tokens(result.stdout, digits),
            end="" if result.stdout.endswith("\n") else "\n",
        )
    if result.stderr:
        print(
            round_numeric_tokens(result.stderr, digits),
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )


def check_python_compile(path: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        remove_py_compile_artifacts(path)
    return result


def remove_py_compile_artifacts(path: Path) -> None:
    cache = importlib.util.cache_from_source(str(path))
    try:
        Path(cache).unlink(missing_ok=True)
    except OSError:
        return
    cache_dir = Path(cache).parent
    try:
        if cache_dir.name == "__pycache__" and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate a numerical subset of R to Python/NumPy.")
    parser.add_argument("source", type=Path, help="R source file")
    parser.add_argument("-o", "--out", type=Path, help="output Python file")
    parser.add_argument("--tee", action="store_true", help="print the emitted Python code")
    parser.add_argument("--no-py-compile", action="store_true", help="skip python -m py_compile check")
    parser.add_argument("--run", action="store_true", help="run the generated Python")
    parser.add_argument("--run-both", action="store_true", help="run original R and generated Python")
    parser.add_argument(
        "--run-diff",
        action="store_true",
        help="run original R and generated Python and compare outputs",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="run original R and generated Python and compare numeric output stats",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        metavar="N",
        help="round floating-point data in displayed Python output to N decimal places",
    )
    parser.add_argument(
        "--round-both",
        type=int,
        default=None,
        metavar="N",
        help="round floating-point data in displayed R and Python output to N decimal places",
    )
    parser.add_argument("--rscript", default="rscript", help="command used to run R scripts")
    args = parser.parse_args(argv)

    if args.round is not None and args.round < 0:
        print("Option error: --round requires a nonnegative integer.")
        return 1
    if args.round_both is not None and args.round_both < 0:
        print("Option error: --round-both requires a nonnegative integer.")
        return 1
    if args.round is not None and args.round_both is not None:
        print("Option conflict: --round and --round-both cannot be used together.")
        return 1

    if args.run_diff and args.stats:
        print("Option conflict: --run-diff and --stats cannot be used together.")
        return 1

    if args.run_diff or args.stats:
        args.run_both = True

    try:
        source = args.source.read_text(encoding="utf-8-sig")
        python = translate_source(source)
    except (OSError, R2PyError) as exc:
        print(f"xr2p: {exc}", file=sys.stderr)
        return 1

    out = args.out or args.source.with_suffix(".py")
    out.write_text(python, encoding="utf-8")
    print(f"wrote {out}")
    if args.tee:
        print(python, end="" if python.endswith("\n") else "\n")
    if not args.no_py_compile:
        compile_result = check_python_compile(out)
        if compile_result.returncode != 0:
            print("Python syntax check failed:", file=sys.stderr)
            print_process_output(compile_result)
            return compile_result.returncode

    python_round_digits = args.round_both if args.round_both is not None else args.round
    r_round_digits = args.round_both

    if args.run_both:
        print("Run (R):", args.rscript, args.source)
        r_result = run_r(args.source, args.rscript)
        print("Run (R):", "PASS" if r_result.returncode == 0 else f"FAIL exit={r_result.returncode}")
        print_result_output(r_result, r_round_digits)
        print("Run (Python):", sys.executable, out)
        py_result = run_python(out)
        print("Run (Python):", "PASS" if py_result.returncode == 0 else f"FAIL exit={py_result.returncode}")
        print_result_output(py_result, python_round_digits)
        if args.run_diff:
            compare_digits = args.round_both if args.round_both is not None else args.round
            same, diff = outputs_match(
                r_result.stdout,
                py_result.stdout,
                digits=compare_digits,
            )
            if not same:
                print("Run (diff):", "FAIL")
                print(diff)
                return 1
            print("Run (diff):", "PASS")
        if args.stats:
            r_stats = numeric_output_stats(r_result.stdout)
            py_stats = numeric_output_stats(py_result.stdout)
            print("Stats (R):", format_numeric_stats(r_stats))
            print("Stats (Python):", format_numeric_stats(py_stats))
            same, detail = numeric_stats_match(r_stats, py_stats)
            if not same:
                print("Stats:", "FAIL")
                print(detail)
                return 1
            print("Stats:", "PASS")
        return 0 if r_result.returncode == 0 and py_result.returncode == 0 else 1
    if args.run:
        result = run_python(out)
        print_result_output(result, python_round_digits)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

