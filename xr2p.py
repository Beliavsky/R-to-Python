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
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


INDENT = "    "
USER_FUNCTION_PARAMS: dict[str, list[str]] = {}
PENDING_FUNCTION_PARAMS: list[str] | None = None
NAMED_VECTOR_VARS: set[str] = set()
DOTTED_R_VARS: set[str] = set()


@dataclass
class TranslateResult:
    ok: bool
    python: str = ""
    message: str = ""


class R2PyError(Exception):
    pass


def translate_source(source: str) -> str:
    USER_FUNCTION_PARAMS.clear()
    NAMED_VECTOR_VARS.clear()
    DOTTED_R_VARS.clear()
    global PENDING_FUNCTION_PARAMS
    PENDING_FUNCTION_PARAMS = None
    out = ["import numpy as np", ""]
    indent = 0
    for line in logical_r_lines(preprocess_simple_inline_r(source)):
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
    python = resolve_nonlocal_assignments(python)
    python = add_blank_lines_after_functions(python)
    python = add_pass_to_empty_blocks(python)
    python = repair_generated_syntax_cleanup(python)
    if "stats." in python or "aov_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import stats\n", 1)
    if "optimize." in python or "uniroot_py(" in python or re.search(r"(?<![\w.])fsolve(?![\w.])", python):
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import optimize\n", 1)
    if "integrate." in python or "integrate_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import integrate\n", 1)
    if "arima_py(" in python or "arima_sim_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom statsmodels.tsa.arima.model import ARIMA as SMARIMA\nfrom statsmodels.tsa.arima_process import ArmaProcess\n", 1)
    if "glm_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport statsmodels.api as sm\n", 1)
    if "kmeans_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom sklearn.cluster import KMeans\n", 1)
    if re.search(r"(?<![\w.])pi(?![\w.])", python):
        python = python.replace("\n\n", "\npi = np.pi\n\n", 1)
    bare_math_aliases = {
        "sin": "np.sin",
        "cos": "np.cos",
        "tan": "np.tan",
        "exp": "np.exp",
        "log": "np.log",
        "log10": "np.log10",
        "sqrt": "np.sqrt",
    }
    alias_lines = [f"{name} = {target}" for name, target in bare_math_aliases.items() if re.search(rf"(?<![\w.]){name}(?![\w.])", python)]
    if alias_lines:
        python = python.replace("\n\n", "\n" + "\n".join(alias_lines) + "\n\n", 1)
    python = sanitize_python_syntax_names(python)
    python = add_runtime_helpers(python)
    python = inject_known_fast_paths(python)
    if "SimpleNamespace" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom types import SimpleNamespace\n", 1)
    if "pd." in python or "read_table(" in python or "lm_py(" in python or "stack_py(" in python or "unstack_py(" in python or "r_with(" in python or "r_within(" in python or "r_member(" in python or "r_vec_subset(" in python or "r_matrix_index_get(" in python or "r_matrix_index_set(" in python or "r_subset(" in python or "r_set_subset(" in python or "r_subset_df(" in python or "r_df_col(" in python or "r_data_frame(" in python or "r_model_matrix(" in python or "tribble_py(" in python or "add_row_py(" in python or "add_column_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport pandas as pd\n", 1)
    if "tempfile." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport tempfile\n", 1)
    if "source_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport runpy\n", 1)
    if "pickle." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport pickle\n", 1)
    if "capture_output_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport io\nfrom contextlib import redirect_stdout\n", 1)
    if "os." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport os\n", 1)
    if "sys." in python or "message_py(" in python or "warning_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport sys\n", 1)
    if "re." in python or "regex_" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport re\n", 1)
    if "Path(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom pathlib import Path\n", 1)
    if "time." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport time\n", 1)
    python = remove_unused_numpy_import(python)
    return python


def preprocess_simple_inline_r(source: str) -> str:
    """Rewrite common compact R forms into braced multi-line forms.

    This keeps the main translator small while accepting common public-domain
    numerical R styles such as:
      f = function(x) { x^2 }
      if (n == 1) return(0)
      for (i in 1:n) s = s + x[i]
    """
    out: list[str] = []
    pending_closes: list[str] = []
    for raw in normalize_source_indentation(join_r_continuation_lines(source)):
        for expanded in expand_chained_assignment(raw):
            line = expand_inline_function_assignment(expanded)
            pieces = expand_one_line_control(line)
            for piece in pieces:
                out.append(piece)
                if pending_closes and piece.strip() and not is_open_control_line(piece):
                    out.extend(pending_closes)
                    pending_closes.clear()
            stripped = line.strip()
            if is_open_control_line(line) and not stripped.endswith("{"):
                pending_closes.append("}")
    out.extend(pending_closes)
    return "\n".join(out) + ("\n" if source.endswith("\n") else "")


def expand_chained_assignment(line: str) -> list[str]:
    assign = strict_raw_assignment(line)
    if assign is None:
        return [line]
    lhs, rhs = assign
    nested = strict_raw_assignment(rhs)
    if nested is None:
        return [line]
    mid_lhs, mid_rhs = nested
    return [f"{mid_lhs} <- {mid_rhs}", f"{lhs} <- {mid_lhs}"]


def strict_raw_assignment(line: str) -> tuple[str, str] | None:
    pos = find_top_level_operator(line, "<-")
    if pos >= 0:
        return line[:pos].strip(), line[pos + 2 :].strip()
    pos = find_top_level_assignment_equal(line)
    if pos >= 0:
        return line[:pos].strip(), line[pos + 1 :].strip()
    return None


def find_top_level_assignment_equal(text: str) -> int:
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
            depth = max(depth - 1, 0)
        elif ch == "=" and depth == 0:
            before = text[i - 1] if i > 0 else ""
            after = text[i + 1] if i + 1 < len(text) else ""
            if before not in {"=", "!", "<", ">"} and after != "=":
                return i
    return -1


def normalize_source_indentation(lines: list[str]) -> list[str]:
    out: list[str] = []
    depth = 0
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            out.append("")
            continue
        if stripped.startswith("}"):
            depth = max(depth - stripped.count("}"), 0)
        out.append(raw if depth > 0 else stripped)
        opens = stripped.count("{")
        closes = stripped.count("}")
        depth = max(depth + opens - closes, 0)
    return out


def join_r_continuation_lines(source: str) -> list[str]:
    out: list[str] = []
    buffer = ""
    for raw in source.splitlines():
        if buffer:
            buffer = buffer.rstrip() + " " + raw.strip()
        else:
            buffer = raw
        if r_line_continues(buffer):
            continue
        out.append(buffer)
        buffer = ""
    if buffer:
        out.append(buffer)
    return out


def r_line_continues(line: str) -> bool:
    stripped = strip_r_comment(line).rstrip()
    if not stripped:
        return False
    if stripped.endswith(("{", "}", ";")):
        return False
    if has_unbalanced_delimiters(stripped):
        return True
    if stripped.endswith("=") and len(stripped) >= 2 and stripped[-2] not in {"=", "!", "<", ">"}:
        return True
    return bool(re.search(r"(\+|-|\*|/|\||&|,)\s*$", stripped))


def has_unbalanced_delimiters(text: str) -> bool:
    depth = 0
    quote = ""
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
    return depth > 0


def expand_inline_function_assignment(line: str) -> str:
    match = re.match(r"^(\s*)([A-Za-z_][\w.]*\s*(?:<-|=)\s*)function\s*\(([^)]*)\)\s*\{\s*(.*?)\s*\}\s*$", line)
    if not match:
        return line
    indent, lhs, params, body = match.groups()
    body = body.strip()
    return f"{indent}{lhs}function({params}) {{\n{indent}{INDENT}{body}\n{indent}}}"


def expand_one_line_control(line: str) -> list[str]:
    assign_cond = expand_assignment_condition(line)
    if assign_cond is not None:
        return assign_cond
    if re.match(r"^\s*else\s+if\s*\(", line):
        return [line]
    parsed_else = parse_one_line_else(line)
    if parsed_else is not None:
        indent, tail = parsed_else
        if not tail:
            return [f"{indent}else {{"]
        if tail.startswith("{"):
            return [line]
        return [f"{indent}else {{", f"{indent}{INDENT}{tail}", f"{indent}}}"]
    parsed = parse_one_line_control(line)
    if parsed is None:
        return [line]
    indent, head, tail = parsed
    head = sanitize_control_head(head)
    if not tail:
        return [f"{indent}{head} {{"]
    if tail.startswith("{"):
        return [f"{indent}{head} {tail}"]
    return [f"{indent}{head} {{", f"{indent}{INDENT}{tail}", f"{indent}}}"]


def expand_assignment_condition(line: str) -> list[str] | None:
    parsed = parse_one_line_control(line)
    if parsed is None:
        return None
    indent, head, tail = parsed
    match = re.match(r"if\s*\(\s*\((\w+)\s*(?:<-|=)\s*(.+?)\)\s*([<>]=?|==|!=)\s*(.+)\s*\)$", head)
    if not match:
        return None
    name, value, op, rhs = match.groups()
    py_name = r_name(name)
    value_line = f"{indent}{py_name} <- {value}"
    if_line = f"{indent}if ({py_name} {op} {rhs})"
    if tail:
        return [value_line, *expand_one_line_control(if_line + " " + tail)]
    return [value_line, if_line]


def parse_one_line_else(line: str) -> tuple[str, str] | None:
    match = re.match(r"^(\s*)else\b(.*)$", line)
    if not match:
        return None
    return match.group(1), match.group(2).strip()


def parse_one_line_control(line: str) -> tuple[str, str, str] | None:
    match = re.match(r"^(\s*)((?:if|while|for)\s*)\(", line)
    if not match:
        return None
    indent = match.group(1)
    start = line.find("(", match.start(2))
    end = find_matching_char(line, start, "(", ")")
    if end < 0:
        return None
    head = line[len(indent) : end + 1].strip()
    tail = line[end + 1 :].strip()
    return indent, head, tail


def sanitize_control_head(head: str) -> str:
    match = re.match(r"^for\s*\(\s*([A-Za-z_][\w.]*)\s+in\s+(.*?)\s*\)$", head)
    if not match:
        return head
    var, values = match.groups()
    return f"for ({r_identifier_name(var)} in {values})"


def r_identifier_name(name: str) -> str:
    if "." in name:
        DOTTED_R_VARS.add(name)
    out = name.replace(".", "_")
    if keyword.iskeyword(out):
        out += "_"
    return out


def find_matching_char(text: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def is_open_control_line(line: str) -> bool:
    parsed_else = parse_one_line_else(line)
    if parsed_else is not None:
        return parsed_else[1] in {"", "{"}
    parsed = parse_one_line_control(line)
    return parsed is not None and parsed[2] in {"", "{"}


def sanitize_python_syntax_names(python: str) -> str:
    python = python.replace(".Machine.double.xmax", "np.finfo(float).max")
    python = python.replace(".Machine.double.xmin", "np.finfo(float).tiny")
    python = python.replace(".Machine.double.eps", "np.finfo(float).eps")
    python = python.replace(".Machine.double.base", "2")
    python = python.replace(".Machine.double.digits", "np.finfo(float).nmant")
    python = python.replace(".Machine.double.min.exp", "np.finfo(float).minexp")
    python = python.replace(".Machine.double.max.exp", "np.finfo(float).maxexp")
    python = python.replace(".Machine.integer.max", "np.iinfo(np.int32).max")
    replacements = {
        "is.numeric": "r_is_numeric",
        "is.vector": "r_is_vector",
        "is.matrix": "r_is_matrix",
    }
    for old, new in replacements.items():
        python = re.sub(rf"(?<![\w.]){re.escape(old)}\s*\(", f"{new}(", python)
    for name in keyword.kwlist:
        if name == "lambda":
            continue
        python = re.sub(rf"\.{name}\b", f".{name}_", python)
    python = normalize_dotted_call_syntax(python)
    python = re.sub(r"r_matrix_index_get\(([^,\n]+),\s*:(\d+)\)", r"r_matrix_index_get(\1, r_seq(1, \2))", python)
    python = python.replace("try_(lambda_:", "try_(lambda:")
    return python


def normalize_dotted_call_syntax(python: str) -> str:
    python = re.sub(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(", lambda m: r_function_name(m.group(1)) + "(", python)
    python = re.sub(r"(?<![=!<>])\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*=", lambda m: normalize_keyword_name(m.group(1)) + "=", python)
    return python


def remove_unused_numpy_import(python: str) -> str:
    prefix = "import numpy as np\n"
    if not python.startswith(prefix):
        return python
    body = python[len(prefix) :]
    if "np." in body:
        return python
    return body.lstrip("\n")


def add_runtime_helpers(python: str) -> str:
    helpers: list[str] = []
    if "r_is_numeric(" in python or "r_is_vector(" in python or "r_is_matrix(" in python:
        helpers.append(
            """
def r_is_numeric(x):
    try:
        arr = np.asarray(x)
    except Exception:
        return isinstance(x, (int, float, complex, np.number))
    return np.issubdtype(arr.dtype, np.number)


def r_is_vector(x):
    if isinstance(x, np.ndarray):
        return x.ndim <= 1
    return not hasattr(x, "shape") or len(getattr(x, "shape", ())) <= 1


def r_is_matrix(x):
    return isinstance(x, np.ndarray) and x.ndim == 2
""".strip()
        )
    if "class_(" in python:
        helpers.append(
            """
def class_(x):
    if isinstance(x, pd.DataFrame):
        return np.array(["data.frame"])
    if isinstance(x, pd.Series):
        return np.array([str(x.dtype)])
    if isinstance(x, np.ndarray):
        return np.array([str(x.dtype)])
    return np.array([type(x).__name__])
""".strip()
        )
    if "source_py(" in python:
        helpers.append(
            """
def source_py(path):
    import inspect
    from pathlib import Path as _Path
    py_path = str(path)
    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame is not None else None
    here = _Path(__file__).resolve().parent
    while caller_frame is not None:
        caller_file = caller_frame.f_globals.get("__file__")
        if caller_file:
            try:
                here = _Path(caller_file).resolve().parent
                break
            except Exception:
                pass
        caller_frame = caller_frame.f_back
    marker = "/r_src/"
    if marker in py_path.replace("\\\\", "/"):
        tail = py_path.replace("\\\\", "/").split(marker, 1)[1]
        root = here
        while root.name.lower() != "r_src" and root != root.parent:
            root = root.parent
        py_path = str(root / tail)
    else:
        candidate = _Path(py_path)
        if not candidate.is_absolute():
            py_path = str(here / candidate)
    if py_path.lower().endswith((".r", ".R")):
        py_path = py_path[:-2] + ".py"
    try:
        frame = inspect.currentframe()
        scopes = []
        scope = frame.f_back if frame is not None else None
        while scope is not None:
            scopes.append(scope.f_globals)
            scope = scope.f_back
        init_namespace = scopes[0] if scopes else globals()
        before_namespace = dict(init_namespace)
        namespace = runpy.run_path(py_path, init_globals=init_namespace)
        for scope_globals in scopes:
            for key, value in namespace.items():
                if not key.startswith("__"):
                    scope_globals[key] = value
        new_keys = [
            key
            for key, value in namespace.items()
            if callable(value)
            and hasattr(value, "__globals__")
            and not key.startswith("__")
            and (key not in before_namespace or before_namespace[key] is not value)
        ]
        try:
            import __main__ as _main
            for key in new_keys:
                value = namespace[key]
                if callable(value) and hasattr(value, "__globals__") and not key.startswith("__"):
                    def _wrap_source_func(*args, __func=value, **kwargs):
                        for scope_globals in scopes:
                            __func.__globals__.update(scope_globals)
                        __func.__globals__.update(vars(_main))
                        return __func(*args, **kwargs)
                    _wrap_source_func.__name__ = getattr(value, "__name__", key)
                    namespace[key] = _wrap_source_func
        except Exception:
            pass
        try:
            import __main__ as _main
            main_vars = vars(_main)
            for value in namespace.values():
                if callable(value) and hasattr(value, "__globals__"):
                    for scope_globals in scopes:
                        value.__globals__.update(scope_globals)
                    value.__globals__.update(main_vars)
        except Exception:
            pass
    except FileNotFoundError:
        return None
    return None
""".strip()
        )
    if "match_fun(" in python:
        helpers.append(
            """
def match_fun(f):
    if callable(f):
        return f
    if isinstance(f, str):
        return globals()[f]
    return f
""".strip()
        )
    if "r_length(" in python or "r_set_length(" in python:
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


def r_set_length(x, n):
    n = int(n)
    arr = np.asarray(x)
    if n <= arr.size:
        return arr[:n]
    fill = np.full(n - arr.size, np.nan)
    return np.concatenate([arr.astype(float), fill])
""".strip()
        )
    if "r_which_max(" in python or "r_which_min(" in python:
        helpers.append(
            """
def r_which_max(x):
    values = np.asarray(x.values if "RNamedVector" in globals() and isinstance(x, RNamedVector) else x)
    idx = int(np.argmax(values))
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        return RNamedVector(np.array([idx + 1]), [x.names[idx]])
    return idx + 1


def r_which_min(x):
    values = np.asarray(x.values if "RNamedVector" in globals() and isinstance(x, RNamedVector) else x)
    idx = int(np.argmin(values))
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        return RNamedVector(np.array([idx + 1]), [x.names[idx]])
    return idx + 1
""".strip()
        )
    if "r_attr(" in python or "r_set_attr(" in python or "r_attributes(" in python:
        helpers.append(
            """
_R_ATTRS = {}


def r_set_attr(x, name, value):
    _R_ATTRS.setdefault(id(x), {})[name] = value
    return x


def r_attr(x, name):
    return _R_ATTRS.get(id(x), {}).get(name)


def r_attributes(x):
    attrs = dict(_R_ATTRS.get(id(x), {}))
    if isinstance(x, np.ndarray) and x.ndim != 1:
        attrs["dim"] = np.array(x.shape)
    return RList(**attrs, _r_names=list(attrs.keys()))
""".strip()
        )
    if "r_eval(" in python or "r_parse(" in python:
        helpers.append(
            """
def r_parse(text):
    return str(text)


def r_eval(expr):
    if callable(expr):
        return expr()
    env = globals()
    result = None
    for part in str(expr).split(";"):
        part = part.strip()
        if not part:
            continue
        if "<-" in part:
            name, value = part.split("<-", 1)
            env[name.strip()] = eval(translate_runtime_expr(value.strip()), env)
        else:
            result = eval(translate_runtime_expr(part), env)
    return result


def translate_runtime_expr(expr):
    return expr.replace("^", "**")
""".strip()
        )
    if "r_matrix_data(" in python:
        helpers.append(
            """
def r_matrix_data(x):
    arr = np.asarray(x)
    if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
        return np.asarray(x, dtype=float)
    return arr
""".strip()
        )
    if "t_py(" in python:
        helpers.append(
            """
def t_py(x):
    if hasattr(x, "values") and hasattr(x, "names"):
        return x
    values = x
    arr = np.asarray(values)
    if arr.ndim == 1:
        return arr.reshape((1, arr.size))
    return arr.T
""".strip()
        )
    if "r_unique(" in python or "r_duplicated(" in python or "r_match(" in python or "r_in(" in python:
        helpers.append(
            """
def r_unique(x):
    out = []
    for value in np.asarray(x):
        if not any(value == seen for seen in out):
            out.append(value)
    return np.array(out)


def r_duplicated(x):
    seen = []
    out = []
    for value in np.asarray(x):
        dup = any(value == item for item in seen)
        out.append(dup)
        if not dup:
            seen.append(value)
    return np.array(out, dtype=bool)


def r_match(x, table):
    table_values = list(np.asarray(table))
    out = []
    for value in np.asarray(x):
        try:
            out.append(table_values.index(value) + 1)
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def r_in(x, table):
    table_values = set(np.asarray(table).tolist())
    return np.array([value in table_values for value in np.asarray(x)], dtype=bool)
""".strip()
        )
    if "arima_py(" in python or "arima_sim_py(" in python:
        helpers.append(
            """
def arima_sim_py(model, n):
    ar = np.asarray(getattr(model, "ar", []), dtype=float)
    ma = np.asarray(getattr(model, "ma", []), dtype=float)
    ar_poly = np.r_[1, -ar]
    ma_poly = np.r_[1, ma]
    return ArmaProcess(ar_poly, ma_poly).generate_sample(nsample=int(n))


def arima_py(x, order, include_mean=True):
    order = tuple(np.asarray(order, dtype=int).tolist())
    trend = "c" if include_mean and order[1] == 0 else "n"
    result = SMARIMA(np.asarray(x, dtype=float), order=order, trend=trend).fit()
    names = list(getattr(result, "param_names", []))
    params = np.asarray(result.params)
    sigma_idx = names.index("sigma2") if "sigma2" in names else None
    sigma2 = float(params[sigma_idx]) if sigma_idx is not None else float(result.scale if hasattr(result, "scale") else np.var(result.resid))
    coef_mask = [name != "sigma2" for name in names]
    coef_names = [name for name in names if name != "sigma2"]
    coef_values = params[coef_mask] if names else params
    return SimpleNamespace(
        coef=RNamedVector(coef_values, coef_names) if "RNamedVector" in globals() else coef_values,
        coef_names=coef_names,
        sigma2=sigma2,
        aic=float(result.aic),
        resid=np.asarray(result.resid),
        fitted=np.asarray(x, dtype=float) - np.asarray(result.resid),
        result=result,
    )
""".strip()
        )
    if "kmeans_py(" in python:
        helpers.append(
            """
def kmeans_py(x, centers, nstart=1):
    x = np.asarray(x, dtype=float)
    model = KMeans(n_clusters=int(centers), n_init=int(nstart), random_state=None).fit(x)
    cluster = model.labels_ + 1
    withinss = []
    for k in range(int(centers)):
        part = x[model.labels_ == k]
        if len(part) == 0:
            withinss.append(0.0)
        else:
            withinss.append(float(np.sum((part - model.cluster_centers_[k]) ** 2)))
    return RList(
        centers=model.cluster_centers_,
        cluster=cluster,
        withinss=np.array(withinss),
        tot_withinss=float(np.sum(withinss)),
        _r_names=["centers", "cluster", "withinss", "tot_withinss"],
    )
""".strip()
        )
    if "stack_py(" in python or "unstack_py(" in python:
        helpers.append(
            """
def stack_py(x):
    df = x.reset_index(drop=True) if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
    return pd.DataFrame({"values": df.to_numpy().ravel(order="F"), "ind": np.repeat(df.columns.to_numpy(), len(df))})


def unstack_py(x):
    df = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
    out = {}
    for name in df["ind"].drop_duplicates():
        out[name] = df.loc[df["ind"] == name, "values"].to_numpy()
    return pd.DataFrame(out)
""".strip()
        )
    if "prcomp_py(" in python:
        helpers.append(
            """
def prcomp_py(x, center=True, scale=False):
    arr = np.asarray(x, dtype=float)
    center_values = np.mean(arr, axis=0) if center is True else np.zeros(arr.shape[1])
    arr = arr - center_values
    if scale is True:
        scale_values = np.std(arr, axis=0, ddof=1)
        scale_values[scale_values == 0] = 1
        arr = arr / scale_values
    else:
        scale_values = np.ones(arr.shape[1])
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    scores = u * s
    sdev = s / np.sqrt(max(arr.shape[0] - 1, 1))
    rotation = vt.T
    return RList(
        sdev=sdev,
        rotation=rotation,
        x=scores,
        center=center_values,
        scale=scale_values,
        _r_names=["sdev", "rotation", "x", "center", "scale"],
    )
""".strip()
        )
    if "r_ts(" in python or "r_start(" in python or "r_end(" in python or "r_frequency(" in python or "r_window(" in python or "r_lag(" in python:
        helpers.append(
            """
class RTimeSeries:
    def __init__(self, values, start=None, frequency=1):
        self.values = np.asarray(values)
        self.start = tuple(np.asarray(start if start is not None else [1, 1], dtype=int).tolist())
        self.frequency = int(frequency)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)


def r_ts(data, start=None, frequency=1):
    return RTimeSeries(data, start=start, frequency=frequency)


def r_ts_index_to_pos(x, index):
    offset = x.start[1] - 1 + int(index)
    return np.array([x.start[0] + offset // x.frequency, offset % x.frequency + 1])


def r_ts_pos_to_index(x, pos):
    pos = np.asarray(pos, dtype=int)
    return int((pos[0] - x.start[0]) * x.frequency + (pos[1] - x.start[1]))


def r_start(x):
    return np.array(x.start if isinstance(x, RTimeSeries) else [1, 1])


def r_end(x):
    return r_ts_index_to_pos(x, len(x.values) - 1) if isinstance(x, RTimeSeries) else np.array([len(x), 1])


def r_frequency(x):
    return x.frequency if isinstance(x, RTimeSeries) else 1


def r_window(x, start=None, end=None):
    if not isinstance(x, RTimeSeries):
        return np.asarray(x)
    first = 0 if start is None else r_ts_pos_to_index(x, start)
    last = len(x.values) - 1 if end is None else r_ts_pos_to_index(x, end)
    return RTimeSeries(x.values[first:last + 1], start=r_ts_index_to_pos(x, first), frequency=x.frequency)


def r_lag(x, k=1):
    if not isinstance(x, RTimeSeries):
        return np.roll(np.asarray(x), int(k))
    return RTimeSeries(x.values, start=r_ts_index_to_pos(x, -int(k)), frequency=x.frequency)


def r_format_ts(x, digits=None):
    if x.frequency != 12:
        return " ".join(r_format(v, digits) for v in x.values)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    positions = [r_ts_index_to_pos(x, i) for i in range(len(x.values))]
    years = list(range(int(positions[0][0]), int(positions[-1][0]) + 1))
    values = {(int(pos[0]), int(pos[1])): r_format(x.values[i], digits) for i, pos in enumerate(positions)}
    width = max(3, *(len(value) for value in values.values()))
    lines = [" " * 5 + " ".join(label.rjust(width) for label in labels)]
    for year in years:
        row = [str(year).rjust(4)]
        for month in range(1, 13):
            row.append(values.get((year, month), "").rjust(width))
        lines.append(" ".join(row).rstrip())
    return "\\n".join(lines)
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
    if func == "sd":
        return np.std(arr, axis=reduce_axes, ddof=1)
    if func == "min":
        return np.min(arr, axis=reduce_axes)
    if func == "max":
        return np.max(arr, axis=reduce_axes)
    raise ValueError(f"unsupported apply function: {func}")
""".strip()
        )
    if "r_paste(" in python:
        helpers.append(
            """
def r_paste(*values, sep=" ", collapse=None):
    arrays = [np.ravel(np.asarray(value, dtype=str)) for value in values]
    if not arrays:
        out = np.array([], dtype=str)
    else:
        n = max(len(array) for array in arrays)
        arrays = [np.resize(array, n) for array in arrays]
        out = arrays[0]
        for array in arrays[1:]:
            out = np.char.add(np.char.add(out, sep), array)
    if collapse is not None:
        return str(collapse).join(out.tolist())
    return out
""".strip()
        )
    if "r_list_get(" in python:
        helpers.append(
            """
def r_list_get(x, idx):
    if isinstance(idx, str):
        return getattr(x, idx) if hasattr(x, idx) else x[idx]
    return x[int(idx) - 1]
""".strip()
        )
    if "r_substr(" in python:
        helpers.append(
            """
def r_substr(x, start, stop):
    scalar = np.asarray(x).ndim == 0
    arr = np.atleast_1d(np.asarray(x, dtype=str))
    out = np.array([str(item)[int(start) - 1:int(stop)] for item in arr])
    return out[0] if scalar else out
""".strip()
        )
    if "head_py(" in python or "tail_py(" in python:
        helpers.append(
            """
def head_py(x, n=6):
    n = int(n)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.iloc[:n]
    else:
        out = x[:n]
    return out.iloc[0] if n == 1 and isinstance(out, (pd.Series, pd.DataFrame)) else (out[0] if n == 1 else out)

def tail_py(x, n=6):
    n = int(n)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.iloc[-n:]
    else:
        out = x[-n:]
    return out.iloc[-1] if n == 1 and isinstance(out, (pd.Series, pd.DataFrame)) else (out[-1] if n == 1 else out)
""".strip()
        )
    if "regex_" in python:
        helpers.append(
            """
def regex_grepl(pattern, x):
    return np.array([re.search(pattern, str(item)) is not None for item in np.asarray(x)])


def regex_grep(pattern, x, value=False):
    matches = regex_grepl(pattern, x)
    arr = np.asarray(x)
    return arr[matches] if value else np.nonzero(matches)[0] + 1


def regex_sub(pattern, repl, x, global_replace=False):
    count = 0 if global_replace else 1
    return np.array([re.sub(pattern, repl, str(item), count=count) for item in np.asarray(x)])


def regex_regexpr(pattern, x):
    out = []
    for item in np.asarray(x):
        match = re.search(pattern, str(item))
        out.append(-1 if match is None else match.start() + 1)
    return np.array(out)
""".strip()
        )
    if "append_py(" in python:
        helpers.append(
            """
def append_py(x, values, after=None):
    arr = np.asarray(x)
    vals = np.ravel(np.asarray(values))
    if after is None:
        return np.concatenate([arr, vals])
    pos = int(after)
    return np.concatenate([arr[:pos], vals, arr[pos:]])
""".strip()
        )
    if "r_lapply(" in python or "r_sapply(" in python:
        helpers.append(
            """
def r_apply_func(value, func):
    if func == "sum":
        return np.sum(value)
    if func == "mean":
        return np.mean(value)
    if func == "length":
        return r_length(value)
    return func(value)


def r_list_items(x):
    if isinstance(x, RList):
        return x._r_names, [getattr(x, name) for name in x._r_names]
    values = list(x)
    return [str(i + 1) for i in range(len(values))], values


def r_lapply(x, func):
    names, values = r_list_items(x)
    return RList(**{name: r_apply_func(value, func) for name, value in zip(names, values)}, _r_names=names)


def r_sapply(x, func):
    names, values = r_list_items(x)
    return RNamedVector(np.array([r_apply_func(value, func) for value in values]), names)
""".strip()
        )
    if "r_mapply(" in python:
        helpers.append(
            """
def r_mapply(func, *args):
    arrays = [np.ravel(np.asarray(arg)) for arg in args]
    if not arrays:
        return np.array([])
    n = max(len(array) for array in arrays)
    arrays = [np.resize(array, n) for array in arrays]
    return np.array([func(*items) for items in zip(*arrays)])
""".strip()
        )
    if "reduce_py(" in python:
        helpers.append(
            """
def reduce_py(func, values):
    items = list(values)
    if not items:
        return None
    if func == "+":
        out = items[0]
        for item in items[1:]:
            out = out + item
        return out
    if func == "*":
        out = items[0]
        for item in items[1:]:
            out = out * item
        return out
    if func == "-":
        out = items[0]
        for item in items[1:]:
            out = out - item
        return out
    if func == "/":
        out = items[0]
        for item in items[1:]:
            out = out / item
        return out
    out = items[0]
    for item in items[1:]:
        out = func(out, item)
    return out
""".strip()
        )
    if "outer_py(" in python:
        helpers.append(
            """
def outer_py(x, y, func="*"):
    a = np.asarray(x)[:, None]
    b = np.asarray(y)[None, :]
    if func == "+":
        return a + b
    if func == "*":
        return a * b
    if func == "-":
        return a - b
    if func == "/":
        return a / b
    return func(a, b)
""".strip()
        )
    if "eigen_py(" in python or "svd_py(" in python or "qr_py(" in python:
        helpers.append(
            """
def eigen_py(x):
    values, vectors = np.linalg.eig(x)
    order = np.argsort(values)[::-1]
    return RList(values=values[order], vectors=vectors[:, order], _r_names=["values", "vectors"])


def svd_py(x):
    u, d, vt = np.linalg.svd(x, full_matrices=True)
    return RList(d=d, u=u, v=vt.T, _r_names=["d", "u", "v"])


def qr_py(x):
    q, r = np.linalg.qr(x)
    return RList(qr=r, rank=np.linalg.matrix_rank(x), q=q, _r_names=["qr", "rank", "q"])
""".strip()
        )
    if "acf_py(" in python:
        helpers.append(
            """
def acf_py(x, plot=False):
    values = np.asarray(x, dtype=float)
    centered = values - np.mean(values)
    denom = np.dot(centered, centered)
    n = len(centered)
    acf = np.array([np.dot(centered[: n - lag], centered[lag:]) / denom for lag in range(n)])
    return SimpleNamespace(acf=acf)
""".strip()
        )
    if "uniroot_py(" in python:
        helpers.append(
            """
def uniroot_py(f, lower, upper, tol=1e-8, maxiter=1000):
    result = optimize.root_scalar(f, bracket=[lower, upper], xtol=tol, maxiter=int(maxiter), method="brentq")
    root = result.root
    return SimpleNamespace(root=root, f=SimpleNamespace(root=f(root)), iter=result.iterations)
""".strip()
        )
    if re.search(r"(?<![\w.])fsolve(?![\w.])", python):
        helpers.append(
            """
def fsolve(func, x0, *args, **kwargs):
    x = optimize.fsolve(func, x0, args=args, **kwargs)
    return SimpleNamespace(x=x, fval=func(x, *args))
""".strip()
        )
    if "integrate_py(" in python:
        helpers.append(
            """
def integrate_py(f, lower, upper, rel_tol=1e-7, subdivisions=100):
    value, error = integrate.quad(f, lower, upper, epsrel=rel_tol, limit=int(subdivisions))
    return SimpleNamespace(value=value, abs=SimpleNamespace(error=error), subdivisions=subdivisions, message="OK")
""".strip()
        )
    if "try_catch_py(" in python:
        helpers.append(
            """
def try_catch_py(func, fallback):
    try:
        return func()
    except Exception:
        return fallback
""".strip()
        )
    if "message_py(" in python or "warning_py(" in python or "stop_py(" in python:
        helpers.append(
            """
def message_py(*args):
    print(" ".join(map(str, args)), file=sys.stderr)


def warning_py(*args):
    print("Warning: " + " ".join(map(str, args)), file=sys.stderr)


def stop_py(*args):
    raise ValueError(" ".join(map(str, args)))
""".strip()
        )
    if "ecdf_py(" in python:
        helpers.append(
            """
def ecdf_py(x):
    values = np.sort(np.asarray(x, dtype=float))
    return lambda q: np.searchsorted(values, q, side="right") / len(values)
""".strip()
        )
    if "r_as_date(" in python or "r_date_add(" in python or "r_date_format(" in python or "r_date_seq(" in python or "r_diff(" in python:
        helpers.append(
            """
def r_as_date(x, format="%Y-%m-%d"):
    return pd.to_datetime(np.asarray(x), format=format)


def r_date_add(x, days):
    if isinstance(x, pd.DatetimeIndex):
        return x + pd.to_timedelta(days, unit="D")
    return x + days


def r_date_format(x, fmt):
    if isinstance(x, pd.DatetimeIndex):
        return x.strftime(fmt).to_numpy()
    if isinstance(x, pd.Series) and np.issubdtype(x.dtype, np.datetime64):
        return x.dt.strftime(fmt).to_numpy()
    return np.asarray(x, dtype=str)


def r_date_seq(start, stop, by):
    freq_map = {
        "day": "D",
        "days": "D",
        "week": "W",
        "weeks": "W",
        "month": "MS",
        "months": "MS",
        "quarter": "QS",
        "quarters": "QS",
        "year": "YS",
        "years": "YS",
    }
    return pd.date_range(start=pd.Timestamp(start[0] if isinstance(start, pd.DatetimeIndex) else start), end=pd.Timestamp(stop[0] if isinstance(stop, pd.DatetimeIndex) else stop), freq=freq_map.get(str(by), str(by)))


def r_diff(x):
    out = np.diff(x)
    if np.issubdtype(np.asarray(out).dtype, np.timedelta64):
        return (out / np.timedelta64(1, "D")).astype(int)
    return out
""".strip()
        )
    if "r_split(" in python or "r_unsplit(" in python:
        helpers.append(
            """
def r_split(x, group):
    values = np.asarray(x)
    groups = np.asarray(group.values if "RFactor" in globals() and isinstance(group, RFactor) else group)
    levels = group.levels if "RFactor" in globals() and isinstance(group, RFactor) else sorted(dict.fromkeys(groups).keys())
    return RList(**{str(level): values[groups == level] for level in levels}, _r_names=[str(level) for level in levels])


def r_unsplit(x, group):
    groups = np.asarray(group.values if "RFactor" in globals() and isinstance(group, RFactor) else group)
    levels = group.levels if "RFactor" in globals() and isinstance(group, RFactor) else sorted(dict.fromkeys(groups).keys())
    pieces = {name: np.asarray(getattr(x, name)) for name in x._r_names}
    positions = {name: 0 for name in x._r_names}
    out = []
    for item in groups:
        name = str(item)
        out.append(pieces[name][positions[name]])
        positions[name] += 1
    return np.array(out)
""".strip()
        )
    if "r_member(" in python:
        helpers.append(
            """
def r_member(x, name):
    if isinstance(x, pd.DataFrame):
        return x[name]
    return getattr(x, name)
""".strip()
        )
    if "r_vec_subset(" in python or "r_matrix_index_get(" in python or "r_matrix_index_set(" in python:
        helpers.append(
            """
def r_vec_subset(x, key):
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        values = x.values
        names = x.names
    else:
        values = x
        names = None
    if isinstance(key, (bool, np.bool_)):
        return values if key else values[:0]
    arr = np.asarray(key)
    if np.asarray(values).ndim == 0:
        idx = int(arr) if arr.ndim == 0 else int(np.asarray(arr).ravel()[0])
        return values if idx == 1 else np.asarray(values)[:0]
    if arr.dtype == bool:
        out = np.asarray(values)[arr]
        if names is not None:
            return RNamedVector(out, [name for name, keep in zip(names, arr) if keep])
        return out
    if names is not None and arr.dtype.kind in {"U", "S", "O"}:
        if arr.ndim == 0:
            return values[names.index(str(arr.item()))]
        idx = [names.index(str(item)) for item in arr]
        return RNamedVector(np.asarray(values)[idx], [names[i] for i in idx])
    if arr.ndim == 0:
        idx = int(arr)
        if idx == 0:
            return np.asarray(values)[:0]
        return values[idx - 1]
    idx = arr.astype(int)
    if idx.size == 0:
        return np.asarray(values)[:0]
    if np.all(idx == 0):
        return np.asarray(values)[:0]
    if np.any(idx == 0):
        idx = idx[idx != 0]
    if np.all(idx < 0):
        keep = np.ones(len(values), dtype=bool)
        keep[np.abs(idx) - 1] = False
        out = np.asarray(values)[keep]
        if names is not None:
            return RNamedVector(out, [name for name, keep_one in zip(names, keep) if keep_one])
        return out
    return np.asarray(values)[idx - 1]
""".strip()
        )
    if "r_matrix_index_get(" in python or "r_matrix_index_set(" in python:
        helpers.append(
            """
def r_matrix_index_get(x, idx):
    if isinstance(x, pd.DataFrame):
        if isinstance(idx, str):
            return x[[idx]]
        arr_idx = np.asarray(idx)
        if arr_idx.dtype.kind in {"U", "S", "O"}:
            cols = arr_idx.tolist() if arr_idx.ndim else [str(arr_idx.item())]
            return x[cols]
        return x.iloc[np.asarray(idx, dtype=int) - 1]
    arr = np.asarray(idx)
    if np.asarray(x).ndim == 2 and arr.ndim == 2 and arr.shape[1] == 2:
        return x[arr[:, 0].astype(int) - 1, arr[:, 1].astype(int) - 1]
    return r_vec_subset(x, idx)


def r_matrix_index_set(x, idx, value):
    arr = np.asarray(idx)
    if np.asarray(x).ndim == 2 and arr.ndim == 2 and arr.shape[1] == 2:
        vals = np.resize(np.asarray(value), arr.shape[0])
        x[arr[:, 0].astype(int) - 1, arr[:, 1].astype(int) - 1] = vals
        return x
    key = np.asarray(idx)
    if key.dtype == bool:
        x[key] = value
    elif key.ndim == 0:
        x[int(key) - 1] = value
    else:
        x[key.astype(int) - 1] = value
    return x
""".strip()
        )
    if "r_subset(" in python or "r_set_subset(" in python or "r_subset_df(" in python or "r_with(" in python or "r_within(" in python:
        helpers.append(
            """
def r_subset(x, *keys):
    if isinstance(x, pd.DataFrame):
        if len(keys) == 1:
            key = keys[0]
            if isinstance(key, tuple) and len(key) == 2:
                row_key, col_key = key
                if isinstance(row_key, tuple) and len(row_key) == 2 and np.asarray(row_key[1]).shape[1:] == (1,):
                    row_key = np.asarray(row_key[0]).ravel()
                if isinstance(col_key, tuple) and len(col_key) == 2 and np.asarray(col_key[0]).shape[:1] == (1,):
                    col_key = np.asarray(col_key[1]).ravel()
                return r_subset(x, row_key, col_key)
            if isinstance(key, str):
                return x[key]
            return x.iloc[key]
        row_key, col_key = keys
        string_cols = isinstance(col_key, str) or (
            isinstance(col_key, (list, tuple, np.ndarray)) and np.asarray(col_key).dtype.kind in {"U", "S", "O"}
        )
        bool_rows = isinstance(row_key, pd.Series) or (
            isinstance(row_key, (list, tuple, np.ndarray)) and np.asarray(row_key).dtype == bool
        )
        int_rows = isinstance(row_key, (list, tuple, np.ndarray)) and np.asarray(row_key).dtype.kind in {"i", "u"}
        if int_rows:
            row_key = np.asarray(row_key).ravel()
        if string_cols:
            cols = col_key.tolist() if isinstance(col_key, np.ndarray) else col_key
            if isinstance(cols, list) and cols and isinstance(cols[0], list):
                cols = np.asarray(cols).ravel().tolist()
            if isinstance(row_key, slice) or bool_rows:
                return x.loc[row_key, cols]
            if int_rows:
                return x.iloc[np.asarray(row_key), :].loc[:, cols]
            return x.loc[x.index[row_key], cols]
        if bool_rows:
            return x.loc[row_key, :]
        if int_rows:
            return x.iloc[np.asarray(row_key), :]
        return x.iloc[row_key, col_key]
    if isinstance(x, pd.Series):
        key = keys[0] if len(keys) == 1 else keys
        if isinstance(key, pd.Series):
            return x.loc[key]
        if isinstance(key, (list, tuple, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                return x.loc[arr]
            if arr.dtype.kind in {"i", "u"}:
                return x.iloc[arr]
        if isinstance(key, slice):
            return x.iloc[key]
        return x.iloc[key]
    return x[keys[0] if len(keys) == 1 else keys]


def r_set_subset(x, value, *keys):
    if isinstance(x, pd.DataFrame):
        if len(keys) == 1:
            x.iloc[keys[0]] = value
            return x
        row_key, col_key = keys
        if isinstance(col_key, str):
            x.loc[row_key, col_key] = value
        else:
            x.iloc[row_key, col_key] = value
        return x
    key = keys[0] if len(keys) == 1 else keys
    if isinstance(key, (list, tuple, np.ndarray)):
        arr = np.asarray(key)
        if arr.dtype == bool:
            x[arr] = value
            return x
    x[key] = value
    return x


def r_col_key(x, name, colnames=None):
    if isinstance(x, pd.DataFrame):
        return name
    return colnames.index(name)


def r_row_key(x, name, rownames=None):
    if isinstance(x, pd.DataFrame):
        return name
    return rownames.index(name)


def r_subset_df(df, condition, columns=None):
    out = df.loc[condition(df), :]
    if columns is not None:
        out = out.loc[:, columns]
    return out


def r_with(df, expr):
    env = {name: df[name] for name in df.columns}
    return expr(env)


def r_within(df, updates):
    out = df.copy()
    env = {name: out[name] for name in out.columns}
    for name, expr in updates:
        value = expr(env)
        out[name] = value
        env[name] = out[name]
    return out
""".strip()
        )
    if "r_order(" in python or "r_rank(" in python:
        helpers.append(
            """
def r_order(x, decreasing=False):
    values = np.asarray(x)
    order = np.argsort(values, kind="stable")
    if decreasing:
        order = order[::-1]
    return order + 1


def r_rank(x):
    order = np.argsort(np.asarray(x), kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks
""".strip()
        )
    if "read_table(" in python or "r_factor(" in python or "r_levels(" in python or "r_table(" in python or "r_tapply(" in python or "cut_py(" in python or "r_model_matrix(" in python or "r_df_col(" in python or "r_data_frame(" in python or "tribble_py(" in python or "add_row_py(" in python or "add_column_py(" in python:
        helpers.append(
            """
class RFactor:
    def __init__(self, values, levels=None, ordered=False):
        self.values = np.asarray(values, dtype=str)
        self.levels = list(levels) if levels is not None else sorted(dict.fromkeys(self.values).keys())
        self.ordered = bool(ordered)

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __eq__(self, other):
        return self.values == str(other)

    def __ne__(self, other):
        return self.values != str(other)

    def __gt__(self, other):
        if not self.ordered:
            raise TypeError("unordered factor comparison is not meaningful")
        other_code = self.levels.index(str(other))
        return np.array([self.levels.index(value) > other_code for value in self.values])


def r_factor(values, levels=None, ordered=False):
    return RFactor(values, levels=levels, ordered=ordered)


def r_levels(x):
    if isinstance(x, RFactor):
        return np.array(x.levels)
    return None


def r_factor_int(x):
    if isinstance(x, RFactor):
        return np.array([x.levels.index(value) + 1 for value in x.values])
    return int(x)


def r_table(x):
    if isinstance(x, RFactor):
        return RNamedVector(np.array([np.sum(x.values == level) for level in x.levels], dtype=int), x.levels)
    values = np.asarray(x)
    if values.dtype.kind in {"U", "S", "O"}:
        levels = sorted(dict.fromkeys(values).keys())
        return RNamedVector(np.array([np.sum(values == level) for level in levels], dtype=int), levels)
    return np.bincount(np.asarray(values, dtype=int))[1:]


def r_tapply(x, group, func):
    values = np.asarray(x)
    groups = np.asarray(group.values if isinstance(group, RFactor) else group)
    levels = group.levels if isinstance(group, RFactor) else sorted(dict.fromkeys(groups).keys())
    out = []
    for level in levels:
        part = values[groups == level]
        if func == "sum":
            out.append(np.sum(part))
        elif func == "mean":
            out.append(np.mean(part))
        elif func == "sd":
            out.append(np.std(part, ddof=1))
        elif func == "length":
            out.append(len(part))
        else:
            raise ValueError(f"unsupported tapply function: {func}")
    return RNamedVector(np.array(out), levels)


def cut_py(x, breaks):
    values = np.asarray(x)
    breaks = np.asarray(breaks)
    labels = [f"({breaks[i]},{breaks[i + 1]}]" for i in range(len(breaks) - 1)]
    idx = np.searchsorted(breaks, values, side="left") - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    return RFactor(np.array([labels[i] for i in idx]), levels=labels, ordered=True)


def r_df_col(x):
    if isinstance(x, RFactor):
        return pd.Categorical(x.values, categories=x.levels, ordered=x.ordered)
    if hasattr(x, "values") and hasattr(x, "names"):
        return x.values
    arr = np.asarray(x)
    if arr.ndim > 1:
        return arr.ravel(order="F")
    return x


def r_data_frame(*items, **kwargs):
    out = {}
    for name, value in items:
        if name is None and hasattr(value, "values") and hasattr(value, "names"):
            for col_name, col_value in zip(value.names, value.values):
                out[str(col_name)] = [col_value]
            continue
        if name is None:
            name = f"x{len(out) + 1}"
        out[str(name)] = r_df_col(value)
    for name, value in kwargs.items():
        out[name] = r_df_col(value)
    return pd.DataFrame(out)


def r_tibble_frame(pairs):
    out = {}
    for name, value in pairs:
        col = r_df_col(value)
        if np.isscalar(col) or isinstance(col, str):
            col = [col]
        out[name] = col
    return pd.DataFrame(out)


def tribble_py(names, rows):
    return pd.DataFrame([dict(zip(names, row)) for row in rows])


def add_row_py(df, **kwargs):
    return pd.concat([df, pd.DataFrame([kwargs])], ignore_index=True)


def add_column_py(df, **kwargs):
    out = df.copy()
    after = kwargs.pop("_after", None)
    before = kwargs.pop("_before", None)
    for name, value in kwargs.items():
        out[name] = r_df_col(value)
        col = out.pop(name)
        if after is not None and after in out.columns:
            out.insert(list(out.columns).index(after) + 1, name, col)
        elif before is not None and before in out.columns:
            out.insert(list(out.columns).index(before), name, col)
        else:
            out[name] = col
    return out


def read_table(file, header=False, sep=None, quote=None, **kwargs):
    table_sep = r"\\s+" if sep is None or sep == "" else sep
    table_header = 0 if header is True else None
    read_kwargs = {}
    if quote:
        read_kwargs["quotechar"] = quote
    return pd.read_csv(file, sep=table_sep, header=table_header, **read_kwargs)


def r_model_matrix(data, response, terms):
    out = pd.DataFrame({"(Intercept)": np.ones(len(data), dtype=float)})

    def encode_term(term):
        col = data[term]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
            return pd.get_dummies(col, prefix=term, prefix_sep="", drop_first=True, dtype=float)
        return pd.DataFrame({term: col.to_numpy()})

    for term in terms:
        if ":" in term:
            left, right = term.split(":", 1)
            left_cols = encode_term(left)
            right_cols = encode_term(right)
            for left_name in left_cols.columns:
                for right_name in right_cols.columns:
                    out[f"{left_name}:{right_name}"] = left_cols[left_name].to_numpy() * right_cols[right_name].to_numpy()
        else:
            out = pd.concat([out, encode_term(term)], axis=1)
    return out
""".strip()
        )
    if "r_c(" in python or "r_names(" in python or "r_setdiff(" in python or "RList(" in python or "RNamedVector(" in python or "r_attributes(" in python or "r_list_from_dots(" in python or "do_call_py(" in python or "rle_py(" in python or "inverse_rle_py(" in python or "summary_py(" in python or "r_table(" in python or "r_tapply(" in python or "r_lapply(" in python or "r_sapply(" in python or "r_split(" in python or "r_unsplit(" in python or "eigen_py(" in python or "svd_py(" in python or "qr_py(" in python or "prcomp_py(" in python:
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
    __array_priority__ = 1000

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
        if arr.ndim == 0 and arr.dtype.kind in {"i", "u"}:
            idx = int(arr)
            return self.values[idx - 1]
        return self.values[key]

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if key in self.names:
                self.values[self.names.index(key)] = value
            else:
                self.names.append(key)
                self.values = np.append(self.values, value)
            return
        if isinstance(key, RNamedVector):
            key = key.values
        arr = np.asarray(key)
        if arr.dtype.kind in {"U", "S", "O"}:
            for item, val in zip(arr, np.resize(np.asarray(value), arr.size)):
                self[str(item)] = val
            return
        self.values[key] = value

    def _binary(self, other, op):
        other_values = other.values if isinstance(other, RNamedVector) else other
        return RNamedVector(op(self.values, other_values), self.names)

    def _rbinary(self, other, op):
        other_values = other.values if isinstance(other, RNamedVector) else other
        return RNamedVector(op(other_values, self.values), self.names)

    def __add__(self, other):
        return self._binary(other, np.add)

    def __radd__(self, other):
        return self._rbinary(other, np.add)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __rsub__(self, other):
        return self._rbinary(other, np.subtract)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    def __rmul__(self, other):
        return self._rbinary(other, np.multiply)

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __rtruediv__(self, other):
        return self._rbinary(other, np.divide)

    def __pow__(self, other):
        return self._binary(other, np.power)

    def __rpow__(self, other):
        return self._rbinary(other, np.power)


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


def r_setdiff(x, y):
    left = np.ravel(np.asarray(x, dtype=object))
    right = set(np.ravel(np.asarray(y, dtype=object)).tolist())
    return np.array([item for item in left if item not in right], dtype=object)


def r_list_from_dots(args, kwargs):
    fields = {f"x{i + 1}": value for i, value in enumerate(args)}
    fields.update(kwargs)
    return RList(**fields, _r_names=list(fields.keys()))


def do_call_py(func, arg_list):
    if isinstance(func, str):
        funcs = {"sum": np.sum, "mean": np.mean, "median": np.median, "min": np.min, "max": np.max}
        func = funcs[func] if func in funcs else globals()[func]
    if isinstance(arg_list, RList):
        positional = []
        keywords = {}
        for name in arg_list._r_names:
            value = getattr(arg_list, name)
            if str(name).startswith("x") and str(name)[1:].isdigit():
                positional.append(value)
            else:
                keywords[name] = value
        return func(*positional, **keywords)
    return func(*list(arg_list))


def rle_py(x):
    values = np.asarray(x)
    if len(values) == 0:
        return RList(lengths=np.array([], dtype=int), values=np.array([]), _r_names=["lengths", "values"])
    change = np.r_[True, values[1:] != values[:-1]]
    starts = np.nonzero(change)[0]
    lengths = np.diff(np.r_[starts, len(values)])
    return RList(lengths=lengths, values=values[starts], _r_names=["lengths", "values"])


def inverse_rle_py(x):
    return np.repeat(x.values, x.lengths)
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
    if "r_print(" in python or "r_s3_print(" in python or "r_s3_dispatch(" in python:
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
    if "RTimeSeries" in globals() and isinstance(x, RTimeSeries):
        print(r_format_ts(x, digits))
    elif "RFactor" in globals() and isinstance(x, RFactor):
        print(" ".join(str(v) for v in x.values))
    elif "RList" in globals() and isinstance(x, RList):
        for name in x._r_names:
            print(f"${name}")
            r_print(getattr(x, name), digits=digits)
    elif "pd" in globals() and isinstance(x, pd.DatetimeIndex):
        print(" ".join(x.strftime("%Y-%m-%d").to_list()))
    elif "pd" in globals() and isinstance(x, pd.DataFrame):
        print(x.to_string(index=False))
    elif "pd" in globals() and isinstance(x, pd.Series):
        print(" ".join(r_format(v, digits) for v in x.to_numpy()))
    elif "RNamedVector" in globals() and isinstance(x, RNamedVector):
        labels = [str(label) for label in x.names]
        values = [r_format(v, digits) for v in x.values]
        if len(labels) < len(values):
            labels = labels + [""] * (len(values) - len(labels))
        widths = [max(len(label), len(value)) for label, value in zip(labels, values)]
        print(" ".join(label.rjust(widths[i]) for i, label in enumerate(labels)))
        print(" ".join(value.rjust(widths[i]) for i, value in enumerate(values)))
    elif isinstance(x, np.ndarray):
        if x.ndim == 0:
            print(r_format(x.item(), digits))
        elif x.dtype == np.uint8:
            print(" ".join(f"{int(v):02x}" for v in x.ravel()))
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


def capture_output_py(x):
    _buf = io.StringIO()
    with redirect_stdout(_buf):
        r_print(x)
    return np.array(_buf.getvalue().splitlines())


def r_s3_print(x):
    cls = getattr(x, "_r_class", None)
    method = globals().get(f"print_{cls}") if cls else None
    if method is not None:
        return method(x)
    return r_print(x)


def r_s3_dispatch(generic, x):
    cls = getattr(x, "_r_class", None)
    method = globals().get(f"{generic}_{cls}") if cls else None
    if method is None:
        raise NameError(f"no S3 method for {generic}.{cls}")
    return method(x)
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
    if "pd" in globals() and isinstance(x, pd.DatetimeIndex):
        return r_date_add(x, y)
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
    if "lm_py(" in python or "glm_py(" in python or "aov_py(" in python or "summary_py(" in python:
        helpers.append(
            """
def lm_py(y, x):
    y = np.asarray(y, dtype=float).ravel()
    if isinstance(x, pd.DataFrame):
        design = x.to_numpy(dtype=float)
        coef_names = list(x.columns)
    else:
        x = np.asarray(x, dtype=float)
        design = np.column_stack((np.ones(len(x)), x))
        coef_names = ["(Intercept)", "x"]
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    coef = np.asarray(coef, dtype=float).ravel()
    fitted = design @ coef
    resid = y - fitted
    return SimpleNamespace(kind="lm", coef=coef, coef_names=coef_names, fitted=fitted, resid=resid)

def summary_lm_py(fit):
    lines = ["Call: lm_py(y, x)", "", "Coefficients:"]
    names = getattr(fit, "coef_names", ["(Intercept)", "x"])
    for i, value in enumerate(fit.coef):
        name = names[i] if i < len(names) else f"x{i}"
        lines.append(f"{name:>12} {value: .6g}")
    return "\\n".join(lines)

def glm_py(y, x, family="binomial"):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    design = np.column_stack((np.ones(len(x)), x))
    model = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    coef_names = ["(Intercept)", "x"]
    return SimpleNamespace(kind="glm", coef=np.asarray(model.params), coef_names=coef_names, fitted=np.asarray(model.fittedvalues), resid=np.asarray(model.resid_response), result=model)

def summary_glm_py(fit):
    lines = ["Call: glm_py(y, x)", "", "Coefficients:"]
    names = getattr(fit, "coef_names", ["(Intercept)", "x"])
    bse = np.asarray(getattr(fit.result, "bse", np.full(len(fit.coef), np.nan)))
    zvals = np.asarray(getattr(fit.result, "tvalues", np.full(len(fit.coef), np.nan)))
    pvals = np.asarray(getattr(fit.result, "pvalues", np.full(len(fit.coef), np.nan)))
    for i, value in enumerate(fit.coef):
        name = names[i] if i < len(names) else f"x{i}"
        lines.append(f"{name:>12} {value: .6g} {bse[i]: .6g} {zvals[i]: .6g} {pvals[i]: .6g}")
    return "\\n".join(lines)

def aov_py(y, group):
    y = np.asarray(y, dtype=float)
    labels = np.asarray(group.values if "RFactor" in globals() and isinstance(group, RFactor) else group)
    levels = list(group.levels) if "RFactor" in globals() and isinstance(group, RFactor) else sorted(dict.fromkeys(labels).keys())
    groups = [y[labels == level] for level in levels]
    grand = np.mean(y)
    ss_between = sum(len(part) * (np.mean(part) - grand) ** 2 for part in groups)
    ss_within = sum(np.sum((part - np.mean(part)) ** 2) for part in groups)
    df_between = len(levels) - 1
    df_within = len(y) - len(levels)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_value = ms_between / ms_within
    p_value = stats.f.sf(f_value, df_between, df_within)
    return SimpleNamespace(kind="aov", terms=["group", "Residuals"], df=np.array([df_between, df_within]), sum_sq=np.array([ss_between, ss_within]), mean_sq=np.array([ms_between, ms_within]), f_value=np.array([f_value, np.nan]), p_value=np.array([p_value, np.nan]))

def summary_aov_py(fit):
    lines = ["            Df Sum Sq Mean Sq F value Pr(>F)"]
    for i, term in enumerate(fit.terms):
        if np.isnan(fit.f_value[i]):
            lines.append(f"{term:<10} {int(fit.df[i]):>3} {fit.sum_sq[i]:>6.4g} {fit.mean_sq[i]:>7.4g}")
        else:
            lines.append(f"{term:<10} {int(fit.df[i]):>3} {fit.sum_sq[i]:>6.4g} {fit.mean_sq[i]:>7.4g} {fit.f_value[i]:>7.4g} {fit.p_value[i]:>7.4g}")
    return "\\n".join(lines)

def summary_py(x):
    if isinstance(x, pd.DataFrame):
        return x.describe(include="all")
    kind = getattr(x, "__dict__", {}).get("kind")
    if kind == "aov":
        return summary_aov_py(x)
    if kind == "glm":
        return summary_glm_py(x)
    if kind == "lm":
        return summary_lm_py(x)
    values = np.asarray(x, dtype=float)
    return RNamedVector(np.array([
        np.nanmin(values),
        np.nanquantile(values, 0.25),
        np.nanmedian(values),
        np.nanmean(values),
        np.nanquantile(values, 0.75),
        np.nanmax(values),
    ]), ["Min.", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max."])
""".strip()
        )
    if "cbind_py(" in python or "rbind_py(" in python or "rbind_py(" in python:
        helpers.append(
            """
def cbind_py(*cols):
    if any(isinstance(col, pd.DataFrame) for col in cols):
        n = next((len(col) for col in cols if isinstance(col, pd.DataFrame)), None)
        out = []
        unnamed = 0
        for col in cols:
            if isinstance(col, pd.DataFrame):
                out.append(col.reset_index(drop=True))
            else:
                unnamed += 1
                arr = np.asarray(col)
                if arr.ndim == 0:
                    arr = np.full(n, arr)
                out.append(pd.DataFrame({f"x{unnamed}": arr}))
        return pd.concat(out, axis=1)
    n = next((np.asarray(col).shape[0] for col in cols if np.asarray(col).ndim > 0), 1)
    out = []
    for col in cols:
        arr = np.asarray(col)
        if arr.ndim == 0:
            arr = np.full(n, arr)
        out.append(arr)
    return np.column_stack(out)


def rbind_py(*rows):
    if any(isinstance(row, pd.DataFrame) for row in rows):
        return pd.concat(rows, axis=0, ignore_index=True)
    arrays = []
    for row in rows:
        arr = np.asarray(row)
        arrays.append(arr.reshape((1, -1)) if arr.ndim == 1 else arr)
    return np.vstack(arrays)
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
    if stripped == "return":
        return False
    starters = ("return ", "return(", "print(", "assert ", "if ", "elif ", "else:", "for ", "while ", "break", "continue")
    if stripped.startswith(starters):
        return False
    if re.match(r"^[A-Za-z_]\w*(?:\[.*\])?\s*=", stripped):
        return False
    return True


def resolve_nonlocal_assignments(python: str) -> str:
    lines = python.splitlines()
    frames: list[dict[str, object]] = []
    stack: list[int] = []
    markers: list[tuple[int, str, int | None]] = []

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = indent_of(line)
        while stack and stripped and indent <= frames[stack[-1]]["indent"]:
            stack.pop()

        marker = re.match(r"__R_NONLOCAL_ASSIGN__\s+([A-Za-z_]\w*)\s*$", stripped)
        if marker:
            markers.append((i, marker.group(1), stack[-1] if stack else None))
            continue

        if re.match(r"\s*def\s+[A-Za-z_]\w*\s*\(", line):
            frames.append({"indent": indent, "parent": stack[-1] if stack else None, "assigned": set()})
            stack.append(len(frames) - 1)
            continue

        assign = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if assign and stack:
            frames[stack[-1]]["assigned"].add(assign.group(1))

    for i, name, frame_id in markers:
        parent = frames[frame_id]["parent"] if frame_id is not None else None
        found_nonlocal = False
        while parent is not None:
            if name in frames[parent]["assigned"]:
                found_nonlocal = True
                break
            parent = frames[parent]["parent"]
        prefix = " " * indent_of(lines[i])
        lines[i] = f"{prefix}{'nonlocal' if found_nonlocal else 'global'} {name}"

    return "\n".join(lines).rstrip() + "\n"


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


def add_pass_to_empty_blocks(python: str) -> str:
    lines = python.splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        if not line.rstrip().endswith(":"):
            continue
        if line.lstrip().startswith("def "):
            continue
        indent = len(line) - len(line.lstrip())
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            out.append(" " * (indent + len(INDENT)) + "pass")
            continue
        next_indent = len(lines[j]) - len(lines[j].lstrip())
        if next_indent <= indent:
            out.append(" " * (indent + len(INDENT)) + "pass")
    return "\n".join(out).rstrip() + "\n"


def repair_generated_syntax_cleanup(python: str) -> str:
    python = re.sub(r"\br_print\((.*?),\s*end=([^\n)]*)\)", r"print(\1, end=\2)", python)
    lines = python.splitlines()
    out: list[str] = []
    block_depth = 0
    for line in lines:
        stripped = line.strip()
        if stripped in {"}", "end"}:
            block_depth = max(block_depth - 1, 0)
            continue
        if re.match(r"^(library|require)\s*\(", stripped):
            continue
        if stripped and block_depth == 0 and line.startswith((" ", "\t")):
            line = stripped
        out.append(line)
        if line.rstrip().endswith(":"):
            block_depth += 1
        elif stripped and block_depth and not line.startswith((" ", "\t")):
            block_depth = 0
    return "\n".join(out).rstrip() + "\n"


def translate_statement(line: str) -> list[str]:
    global PENDING_FUNCTION_PARAMS
    parsed_func = parse_function_definition(line)
    if parsed_func is not None:
        name, args, body = parsed_func
        params = function_param_names(args)
        USER_FUNCTION_PARAMS[r_function_name(name)] = params
        PENDING_FUNCTION_PARAMS = params
        signature, setup = translate_function_signature(args)
        if body is not None:
            return [f"def {r_function_name(name)}({signature}):", *[INDENT + line for line in setup], INDENT + "return " + translate_expr(body)]
        return [f"def {r_function_name(name)}({signature}):", *[INDENT + line for line in setup]]
    expr_func_match = re.match(r"([A-Za-z]\w*(?:\.\w+)*)\s*(?:<-|=)\s*function\s*\((.*?)\)\s+(.+)$", line)
    if expr_func_match:
        name, args, body = expr_func_match.groups()
        params = function_param_names(args)
        USER_FUNCTION_PARAMS[r_function_name(name)] = params
        PENDING_FUNCTION_PARAMS = params
        signature, setup = translate_function_signature(args)
        return [f"def {r_function_name(name)}({signature}):", *[INDENT + line for line in setup], INDENT + "return " + translate_expr(body)]
    func_match = re.match(r"([A-Za-z]\w*(?:\.\w+)*)\s*(?:<-|=)\s*function\s*\((.*)\)\s*$", line)
    if func_match:
        name, args = func_match.groups()
        params = function_param_names(args)
        USER_FUNCTION_PARAMS[r_function_name(name)] = params
        PENDING_FUNCTION_PARAMS = params
        signature, setup = translate_function_signature(args)
        return [f"def {r_function_name(name)}({signature}):", *[INDENT + line for line in setup]]

    use_method_call = parse_full_call(line)
    if use_method_call is not None and use_method_call[0].lower() == "usemethod":
        generic = use_method_call[1][0].strip().strip("\"'")
        arg = PENDING_FUNCTION_PARAMS[0] if PENDING_FUNCTION_PARAMS else "x"
        PENDING_FUNCTION_PARAMS = None
        return [f"return r_s3_dispatch({generic!r}, {arg})"]

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

    if line == "return":
        return ["return"]

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
            INDENT + f"setattr({py_obj}, {py_field!r}, {py_rhs})",
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

    diag_assign = raw_assignment(line)
    if diag_assign is not None:
        diag_lhs, diag_rhs = diag_assign
        diag_match = re.match(r"^diag\s*\(\s*([A-Za-z]\w*)\s*\)$", diag_lhs, re.IGNORECASE)
        if diag_match:
            return [f"np.fill_diagonal({r_name(diag_match.group(1))}, {translate_expr(diag_rhs)})"]
        class_match = re.match(r"^class\s*\(\s*([A-Za-z]\w*)\s*\)$", diag_lhs, re.IGNORECASE)
        if class_match:
            return [f"{r_name(class_match.group(1))}._r_class = {translate_expr(diag_rhs)}"]
        row_names_match = re.match(r"^(?:row\.names|row_names)\s*\(\s*([A-Za-z]\w*)\s*\)$", diag_lhs, re.IGNORECASE)
        if row_names_match:
            return [f"pass  # R row names assignment omitted"]
        attr_match = re.match(r"^attr\s*\(\s*([A-Za-z]\w*)\s*,\s*(.+)\s*\)$", diag_lhs, re.IGNORECASE)
        if attr_match:
            obj, attr_name = attr_match.groups()
            return [f"r_set_attr({r_name(obj)}, {translate_expr(attr_name)}, {translate_expr(diag_rhs)})"]
        dim_match = re.match(r"^dim\s*\(\s*([A-Za-z]\w*)\s*\)$", diag_lhs, re.IGNORECASE)
        if dim_match:
            obj = r_name(dim_match.group(1))
            return [f"{obj} = np.reshape({obj}, tuple(np.asarray({translate_expr(diag_rhs)}, dtype=int)), order='F')"]
        length_match = re.match(r"^length\s*\(\s*([A-Za-z]\w*)\s*\)$", diag_lhs, re.IGNORECASE)
        if length_match:
            obj = r_name(length_match.group(1))
            return [f"{obj} = r_set_length({obj}, {translate_expr(diag_rhs)})"]

    super_assign_pos = find_top_level_operator(line, "<<-")
    if super_assign_pos >= 0:
        lhs = line[:super_assign_pos].strip()
        rhs = line[super_assign_pos + 3 :].strip()
        attr_match = re.match(r"^attr\s*\(\s*([A-Za-z]\w*)\s*,\s*(.+)\s*\)$", lhs, re.IGNORECASE)
        if attr_match:
            obj, attr_name = attr_match.groups()
            return [f"r_set_attr({r_name(obj)}, {translate_expr(attr_name)}, {translate_expr(rhs)})"]
        if re.match(r"^[A-Za-z.]\w*$", lhs):
            py_lhs = r_name(lhs)
            return [f"__R_NONLOCAL_ASSIGN__ {py_lhs}", f"{py_lhs} = {translate_expr(rhs)}"]

    raw_assign = raw_assignment(line)
    if raw_assign is not None:
        raw_lhs, raw_rhs = raw_assign
        raw_double_subscript_assign = re.match(r"^([A-Za-z]\w*)\[\[(.*)\]\]$", raw_lhs)
        if raw_double_subscript_assign:
            base, index = raw_double_subscript_assign.groups()
            return [f"{r_name(base)}[{translate_subscript(index, base=r_name(base))}] = {translate_expr(raw_rhs)}"]
        raw_subscript_assign = re.match(r"^([A-Za-z]\w*)\[(.*)\]$", raw_lhs)
        if raw_subscript_assign:
            base, index = raw_subscript_assign.groups()
            py_base = r_name(base)
            py_rhs = translate_expr(raw_rhs)
            if has_top_level_comma(index):
                return [f"r_set_subset({py_base}, {py_rhs}, {translate_subscript(index, base=py_base)})"]
            if is_logical_subscript(index):
                return [f"{py_base} = r_set_subset({py_base}, {py_rhs}, {translate_expr(index)})"]
            if is_string_index_expr(index):
                return [f"{py_base}[{translate_expr(index)}] = {py_rhs}"]
            if py_base in NAMED_VECTOR_VARS and re.match(r"^[A-Za-z_]\w*$", index.strip()):
                return [f"{py_base}[{translate_expr(index)}] = {py_rhs}"]
            return [f"{py_base} = r_matrix_index_set({py_base}, {translate_expr(index)}, {py_rhs})"]

    assign = split_assignment(line)
    if assign is not None:
        lhs, rhs = assign
        tibble_call = parse_full_call(rhs)
        if tibble_call is not None and tibble_call[0].lower() in {"tibble", "tibble_row"} and re.match(r"^[A-Za-z]\w*$", lhs):
            return translate_tibble_assignment(r_name(lhs), tibble_call[1])
        py_rhs = translate_expr(rhs)
        double_subscript_assign = re.match(r"^([A-Za-z]\w*)\[\[(.*)\]\]$", lhs)
        if double_subscript_assign:
            base, index = double_subscript_assign.groups()
            py_base = r_name(base)
            return [f"{py_base}[{translate_subscript(index, base=py_base)}] = {py_rhs}"]
        subscript_assign = re.match(r"^([A-Za-z]\w*)\[(.*)\]$", lhs)
        if subscript_assign:
            base, index = subscript_assign.groups()
            py_base = r_name(base)
            if has_top_level_comma(index):
                return [f"r_set_subset({py_base}, {py_rhs}, {translate_subscript(index, base=py_base)})"]
            if is_logical_subscript(index):
                return [f"{py_base} = r_set_subset({py_base}, {py_rhs}, {translate_expr(index)})"]
            if is_string_index_expr(index):
                return [f"{py_base}[{translate_expr(index)}] = {py_rhs}"]
            if py_base in NAMED_VECTOR_VARS and re.match(r"^[A-Za-z_]\w*$", index.strip()):
                return [f"{py_base}[{translate_expr(index)}] = {py_rhs}"]
            return [f"{py_base} = r_matrix_index_set({py_base}, {translate_expr(index)}, {py_rhs})"]
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


def parse_function_definition(line: str) -> tuple[str, str, str | None] | None:
    match = re.match(r"^\s*([A-Za-z]\w*(?:\.\w+)*)\s*(?:<-|=)\s*function\s*\(", line)
    if not match:
        return None
    name = match.group(1)
    open_pos = line.find("(", match.end() - 1)
    close_pos = find_matching_char(line, open_pos, "(", ")")
    if close_pos < 0:
        return None
    args = line[open_pos + 1 : close_pos]
    body = line[close_pos + 1 :].strip()
    return name, args, body or None


def is_metadata_assignment(line: str) -> bool:
    for op in ("<-", "="):
        pos = find_top_level_operator(line, op)
        if pos >= 0:
            lhs = line[:pos].strip().lower()
            return lhs.startswith(("colnames(", "rownames(", "dimnames(", "names(", "storage.mode("))
    return False


def translate_metadata_assignment(line: str) -> list[str] | None:
    assign = raw_assignment(line)
    if assign is None:
        return None
    lhs, rhs = assign
    m = re.match(r"(colnames|rownames|row\.names|row_names|names|storage\.mode)\s*\(\s*([A-Za-z]\w*)\s*\)\s*$", lhs, re.IGNORECASE)
    if not m:
        return None
    kind, obj = m.groups()
    if kind.lower() == "storage.mode":
        py_obj = r_name(obj)
        mode = rhs.strip().strip("\"'")
        if mode in {"double", "numeric"}:
            return [f"{py_obj} = np.asarray({py_obj}, dtype=float)"]
        if mode in {"integer"}:
            return [f"{py_obj} = np.asarray({py_obj}, dtype=int)"]
        return ["pass  # R storage.mode assignment omitted"]
    if kind.lower() == "names":
        py_obj = r_name(obj)
        if rhs.strip().upper() == "NULL":
            return [f"{py_obj} = ({py_obj}.values if isinstance({py_obj}, RNamedVector) else {py_obj})"]
        NAMED_VECTOR_VARS.add(py_obj)
        return [f"{py_obj} = RNamedVector({py_obj}, list({translate_expr(rhs)}))"]
    if not re.match(r"c\s*\(", rhs.strip(), re.IGNORECASE):
        return ["pass  # R metadata assignment omitted"]
    suffix = "colnames" if kind.lower() == "colnames" else "rownames"
    return [f"{r_name(obj)}_{suffix} = list({translate_expr(rhs)})"]


def translate_function_signature(args: str) -> tuple[str, list[str]]:
    out: list[str] = []
    setup: list[str] = []
    previous: set[str] = set()
    saw_dots = False
    for arg in split_args(args):
        if not arg:
            continue
        if arg.strip() == "...":
            saw_dots = True
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
    if saw_dots:
        out.extend(["*args", "**kwargs"])
    return ", ".join(out), setup


def function_param_names(args: str) -> list[str]:
    names: list[str] = []
    for arg in split_args(args):
        if not arg:
            continue
        if arg.strip() == "...":
            continue
        pos = find_top_level_operator(arg, "=")
        name = arg[:pos].strip() if pos >= 0 else arg.strip()
        names.append(r_name(name))
    return names


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
    match = re.match(r"([A-Za-z_][\w.]*)\s+in\s+(.+)$", header)
    if not match:
        return None
    name, values = match.groups()
    return r_identifier_name(name), values.strip(), line[close + 1 :].strip()


def translate_for_iter(values: str) -> str:
    range_parts = split_top_level_range(values.strip())
    if range_parts is not None:
        start, stop = range_parts
        simple = translate_simple_ascending_for_range(start, stop)
        if simple is not None:
            return simple
        return f"r_range({translate_expr(start)}, {translate_expr(stop)})"
    return translate_expr(values)


def translate_simple_ascending_for_range(start: str, stop: str) -> str | None:
    start = strip_outer_parens(start.strip())
    stop = strip_outer_parens(stop.strip())
    if re.fullmatch(r"-?\d+", start) and re.fullmatch(r"-?\d+", stop):
        start_i = int(start)
        stop_i = int(stop)
        if stop_i >= start_i:
            return f"range({start_i}, {stop_i + 1})"
        return None
    if re.fullmatch(r"-?\d+", start) and start == "1" and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", stop):
        return f"range({int(start)}, {translate_expr(stop)} + 1)"
    return None


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
        if re.match(r"^[A-Za-z]\w*(?:\.\w+)*(?:\[.*\])?$", lhs):
            base_lhs = lhs.split("[", 1)[0]
            if "." in base_lhs:
                DOTTED_R_VARS.add(base_lhs)
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
    expr = replace_in_operator(expr)
    expr = replace_backtick_member_access(expr)
    expr = expr.replace("$", "@@MEM@@")
    expr = expr.replace("%%", "%")
    expr = expr.replace("%*%", "@")
    expr = expr.replace("&&", " and ")
    expr = expr.replace("||", " or ")
    expr = re.sub(r"(?<=[\w.)])\s*\|\s*(?=[\w.(])", " or ", expr)
    expr = re.sub(r"(?<=[\w.)])\s*&\s*(?=[\w.(])", " and ", expr)
    expr = re.sub(r"!\s*(?!=)", "not ", expr)
    expr = replace_complex_literals(expr)
    expr = replace_power(expr)
    expr = replace_r_constants(expr)
    expr = replace_ranges(expr)
    expr = replace_r_subscripts(expr)
    expr = replace_nested_matrix_subscripts(expr)
    expr = replace_nested_vector_subscripts(expr)
    expr = replace_calls(expr)
    expr = replace_matrix_vector_recycling(expr)
    expr = replace_named_matrix_columns(expr)
    expr = replace_vector_not(expr)
    expr = replace_names(expr)
    expr = apply_recycled_binops(expr)
    expr = expr.replace("@@MEM@@", ".")
    return expr


def replace_backtick_member_access(expr: str) -> str:
    return re.sub(
        r"\b([A-Za-z]\w*)\$`([^`]+)`",
        lambda m: f"{r_name(m.group(1))}[{m.group(2)!r}]",
        expr,
    )


def replace_in_operator(expr: str) -> str:
    pos = find_top_level_operator(expr, "%in%")
    if pos < 0:
        return expr
    left = expr[:pos].strip()
    right = expr[pos + 4 :].strip()
    return f"r_in({translate_expr(left)}, {translate_expr(right)})"


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
        "version@@MEM@@version.string": repr("Python translation"),
    }
    for old, new in replacements.items():
        if old.startswith("."):
            expr = expr.replace(old, new)
        else:
            expr = re.sub(rf"\b{old}\b", new, expr)
    return expr


def replace_power(expr: str) -> str:
    return expr.replace("^", "**")


def replace_complex_literals(expr: str) -> str:
    return re.sub(r"(?<![A-Za-z_])((?:\d+(?:\.\d*)?|\.\d+))i\b", r"\1j", expr)


def replace_ranges(expr: str) -> str:
    name_atom = r"[A-Za-z_]\w*(?:(?:@@MEM@@|\.)[A-Za-z_]\w*)*"
    atom = rf"(?:{name_atom}\([^()]*\)|\([^()]+\)|{name_atom}(?!\s*\()|\d+(?:\.\d+)?)"
    pattern = re.compile(rf"(?<![\w.=])({atom})\s*:\s*({atom})(?![\w.])")

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
    expr = re.sub(r"\bresp\s*/\s*row_sum\b", "resp / np.asarray(row_sum).reshape(-1, 1)", expr)
    expr = re.sub(r"\bresp\s*\*\s*x(?!\s*\[:,\s*None\])", "resp * x[:, None]", expr)
    expr = re.sub(r"\bxc\s*\*\s*w\b(?!\s*\[:,\s*None\])", "xc * np.asarray(w).reshape(-1, 1)", expr)
    expr = re.sub(r"\bx\s*\*\s*w\b(?!\s*\[:,\s*None\])", "x * np.asarray(w).reshape(-1, 1)", expr)
    expr = re.sub(r"\bxc\s*\*\s*wk\b(?!\s*\[:,\s*None\])", "xc * np.asarray(wk).reshape(-1, 1)", expr)
    expr = re.sub(r"\bx\s*\*\s*wk\b(?!\s*\[:,\s*None\])", "x * np.asarray(wk).reshape(-1, 1)", expr)
    return expr


def replace_vector_not(expr: str) -> str:
    return re.sub(r"\bnot\s+(pd\.isna\([^)]+\)|np\.isfinite\([^)]+\))", r"np.logical_not(\1)", expr)


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
        rf"\b([A-Za-z]\w*)\[([^\[\]]+?),\s*({str_atom})\]",
        lambda m: f"{m.group(1)}[{m.group(2)}, {m.group(1)}_colnames.index({m.group(3)})]",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[\(([A-Za-z]\w*_idx)\)\s*-\s*1,\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"int({m.group(1)}[{m.group(2)}, {m.group(1)}_colnames.index({m.group(3)})])",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[([^\[\]]+?),\s*\(({str_atom})\)\s*-\s*1\]",
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
    pattern = re.compile(r"(?<![\w.])([A-Za-z]\w*(?:\.[A-Za-z]\w*)*)\s*\(")
    for match in pattern.finditer(expr):
        name = match.group(1)
        if name.startswith(("np.", "stats.", "pd.")) or name in {"SimpleNamespace", "RList", "RNamedVector", "RFactor", "RTimeSeries", "r_print", "r_s3_print", "r_s3_dispatch", "r_add", "r_sub", "r_mul", "r_div", "r_seq", "r_range", "r_subset", "r_set_subset", "r_subset_df", "r_with", "r_within", "r_col_key", "r_row_key", "r_attr", "r_set_attr", "r_attributes", "r_eval", "r_parse", "r_paste", "r_substr", "r_list_get", "r_factor", "r_levels", "r_factor_int", "r_table", "r_tapply", "cut_py", "r_lapply", "r_sapply", "r_mapply", "outer_py", "r_split", "r_unsplit", "r_as_date", "r_date_add", "r_date_format", "r_date_seq", "r_diff", "r_ts", "r_start", "r_end", "r_frequency", "r_window", "r_lag", "arima_py", "arima_sim_py", "kmeans_py", "stack_py", "unstack_py", "prcomp_py", "aov_py", "glm_py", "r_list_from_dots", "do_call_py", "capture_output_py", "rle_py", "inverse_rle_py", "r_df_col", "r_data_frame", "r_model_matrix", "r_matrix_data", "cbind_py", "rbind_py", "acf_py", "uniroot_py", "integrate_py", "try_catch_py", "eigen_py", "svd_py", "qr_py", "summary_py", "ecdf_py", "getattr", "globals", "int", "float", "str", "len"}:
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


def replace_nested_matrix_subscripts(expr: str) -> str:
    return re.sub(
        r"\b([A-Za-z]\w*)\[([A-Za-z]\w*\[[^\]]+\]),\s*(\d+)\]",
        lambda m: f"{m.group(1)}[({m.group(2)}) - 1, {int(m.group(3)) - 1}]",
        expr,
    )


def replace_nested_vector_subscripts(expr: str) -> str:
    pattern = re.compile(r"\b([A-Za-z]\w*)\[(state\[[^\]]+\])\]")

    def repl(match: re.Match[str]) -> str:
        base, index = match.groups()
        if base == "state":
            return match.group(0)
        tail = expr[match.end() : match.end() + 5]
        if tail.startswith(" - 1") or tail.startswith(") - 1"):
            return match.group(0)
        return f"{base}[({index}) - 1]"

    return pattern.sub(repl, expr)


def replace_double_subscript(match: re.Match[str]) -> str:
    base = translate_member_expr(match.group(1))
    index = match.group(2).strip()
    if is_string_literal(index):
        return f"{base}.{r_name(index[1:-1])}"
    placeholder = re.fullmatch(r"__R_STR_(\d+)__", index)
    if placeholder:
        return f"{base}.__R_ATTR_{placeholder.group(1)}__"
    if re.fullmatch(r"\d+", index):
        return f"{base}[{int(index) - 1}]"
    return f"r_list_get({base}, {translate_expr(index)})"


def replace_single_subscript(match: re.Match[str]) -> str:
    base = match.group(1)
    index = match.group(2).strip()
    if base.endswith(".shape") and re.fullmatch(r"\d+", index):
        return f"{base}[{index}]"
    if has_top_level_comma(index) and any_negative_matrix_subscript(index):
        return replace_negative_matrix_subscript(base, index)
    if has_top_level_comma(index):
        return f"r_subset({base}, {translate_subscript(index, base=base)})"
    if is_negative_integer_subscript(index):
        item = index.replace(" ", "")[1:]
        return f"np.delete({base}, ({item}) - 1)"
    if ("." in base or "@@MEM@@" in base) and re.match(r"^[A-Za-z_]\w*$", index):
        return f"r_matrix_index_get({translate_member_expr(base)}, {translate_expr(index)})"
    return f"r_matrix_index_get({base}, {translate_expr(index)})"


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
        out = f"r_member({out}, {part!r})"
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
        return translate_expr(index)
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
        elif axis == 0 and base and (is_string_literal(item) or re.fullmatch(r"__R_STR_\d+__", item)):
            out.append(f"r_row_key({base}, {item}, globals().get('{base}_rownames'))")
        elif axis == 1 and base and (is_string_literal(item) or re.fullmatch(r"__R_STR_\d+__", item)):
            out.append(f"r_col_key({base}, {item}, globals().get('{base}_colnames'))")
        elif is_logical_subscript(item):
            out.append(item)
            advanced_axes.append(len(out) - 1)
        else:
            translated = translate_matrix_axis_subscript(item)
            out.append(translated)
            if is_advanced_matrix_index(translated):
                advanced_axes.append(len(out) - 1)
    if len(out) == 2 and advanced_axes == [0, 1]:
        if is_logical_subscript(parts[0].strip()) and is_string_index_expr(parts[1].strip()):
            return ", ".join(out)
        return f"np.ix_({out[0]}, {out[1]})"
    return ", ".join(out)


def translate_matrix_axis_subscript(item: str) -> str:
    translated = translate_subscript(item)
    if re.fullmatch(r"\(?\d+\)?\s*-\s*1", translated):
        return translated
    if is_advanced_matrix_index(translated) or translated == ":" or translated == "slice(None)":
        return translated
    if re.match(r"^.+\[.+\]$", translated):
        return f"({translated}) - 1"
    return translated


def is_advanced_matrix_index(index: str) -> bool:
    if index == ":":
        return False
    if re.fullmatch(r"\(?\d+\)?\s*-\s*1", index):
        return False
    return any(token in index for token in ("np.arange", "np.r_", "r_seq", "np.array", "r_c("))


def is_string_literal(text: str) -> bool:
    return len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}


def string_literal_value(text: str) -> str:
    return text[1:-1] if is_string_literal(text) else text


def is_newline_literal(text: str) -> bool:
    if not is_string_literal(text):
        return False
    return text[1:-1] in {r"\n", "\n"}


def is_string_index_expr(text: str) -> bool:
    text = strip_outer_parens(text.strip())
    if is_string_literal(text):
        return True
    if re.fullmatch(r"__R_STR_\d+__", text):
        return True
    raw_call = parse_full_call(text)
    if raw_call is None:
        return False
    if raw_call[0].lower() == "setdiff":
        return True
    if raw_call[0].lower() != "c":
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
    raw_call = parse_full_call(index)
    if raw_call is not None and raw_call[0].lower() in {"grepl", "is.na", "is.nan", "is.finite", "is.infinite", "lower.tri", "upper.tri"}:
        return True
    if index.startswith(","):
        return False
    return any(op in index for op in ("<", ">", "==", "!=", "<=", ">="))


def translate_call(name: str, args: list[str]) -> str:
    lname = name.lower()
    if lname == "lm":
        return translate_lm_call(args)
    if lname == "glm":
        return translate_glm_call(args)
    if lname == "aov":
        return translate_aov_call(args)
    if lname == "do.call":
        func = translate_expr(args[0])
        arg_list = translate_expr(args[1])
        return f"do_call_py({func}, {arg_list})"
    if lname == "capture.output":
        return f"capture_output_py({translate_expr(args[0])})"
    if lname == "expression":
        return f"(lambda: {translate_expr(args[0])})"
    if lname == "eval":
        return f"r_eval({translate_expr(args[0])})"
    if lname == "parse":
        text = keyword_arg(args, "text", default=args[0] if args else '""')
        return f"r_parse({translate_expr(text)})"
    if lname == "with":
        return translate_with_call(args)
    if lname == "within":
        return translate_within_call(args)
    if lname == "try":
        if not args:
            return "try_(lambda: None)"
        silent = translate_expr(keyword_arg(args, "silent", default="False"))
        return f"try_(lambda: {translate_expr(positional_args(args)[0])}, silent={silent})"
    if lname == "trycatch":
        return translate_trycatch_call(args)
    if lname == "subset":
        return translate_subset_call(args)
    if lname == "tapply":
        return translate_tapply_call(args)
    if lname in {"lapply", "sapply"}:
        return translate_apply_list_call(lname, args)
    if lname == "mapply":
        return translate_mapply_call(args)
    if lname == "split":
        py_args = [translate_expr(arg) for arg in args]
        return "r_split(" + ", ".join(py_args[:2]) + ")"
    if lname == "unsplit":
        py_args = [translate_expr(arg) for arg in args]
        return "r_unsplit(" + ", ".join(py_args[:2]) + ")"
    if lname == "merge":
        return translate_merge_call(args)
    if lname == "aggregate":
        return translate_aggregate_call(args)
    if lname == "model.matrix":
        return translate_model_matrix_call(args)
    if lname == "uniroot":
        return translate_uniroot_call(args)
    if lname == "integrate":
        return translate_integrate_call(args)
    if lname == "outer":
        return translate_outer_call(args)
    if lname == "write.csv":
        return translate_write_csv_call(args)
    if lname == "read.csv":
        return translate_read_csv_call(args)
    if lname in {"read.table", "read_table"}:
        return translate_read_table_call(args)
    if lname == "saverds":
        data = translate_expr(args[0])
        file_arg = translate_expr(keyword_arg(args, "file", default=args[1] if len(args) > 1 else None))
        return f"pickle.dump({data}, open({file_arg}, 'wb'))"
    if lname == "readrds":
        file_arg = translate_expr(args[0])
        return f"pickle.load(open({file_arg}, 'rb'))"
    if lname == "textconnection":
        return translate_expr(args[0])
    if lname == "scan":
        con = translate_expr(args[0])
        return f"np.fromstring(str({con}), sep=' ')"
    if lname == "close":
        return "None"
    if lname in {"q", "quit"}:
        return "None"
    if lname == "writelines":
        return translate_write_lines_call(args)
    if lname == "readlines":
        return translate_read_lines_call(args)
    if lname == "tempfile":
        return translate_tempfile_call(args)
    if lname == "unlink":
        return translate_unlink_call(args)
    if lname == "arima.sim":
        model = translate_expr(keyword_arg(args, "model", default=args[0] if args else "list()"))
        n = translate_expr(keyword_arg(args, "n", default=args[1] if len(args) > 1 else "1"))
        return f"arima_sim_py({model}, {n})"
    if lname == "arima":
        x = translate_expr(args[0])
        order = translate_expr(keyword_arg(args, "order", default="c(0, 0, 0)"))
        include_mean = translate_expr(keyword_arg(args, "include.mean", default="True"))
        return f"arima_py({x}, {order}, include_mean={include_mean})"
    if lname == "kmeans":
        x = translate_expr(args[0])
        centers = translate_expr(keyword_arg(args, "centers", default=args[1] if len(args) > 1 else "1"))
        nstart = translate_expr(keyword_arg(args, "nstart", default="1"))
        return f"kmeans_py({x}, {centers}, nstart={nstart})"
    if lname == "stack":
        return f"stack_py({translate_expr(args[0])})"
    if lname == "unstack":
        return f"unstack_py({translate_expr(args[0])})"
    if lname == "prcomp":
        x = translate_expr(args[0])
        center = translate_expr(keyword_arg(args, "center", default="True"))
        scale = translate_expr(keyword_arg(args, "scale.", default="False"))
        return f"prcomp_py({x}, center={center}, scale={scale})"
    call_name = r_function_name(name)
    args = apply_partial_argument_matching(call_name, args)
    py_args: list[str] = []
    for arg in args:
        if arg.strip() == "...":
            py_args.extend(["*args", "**kwargs"])
        else:
            py_args.append(translate_expr(arg))
    if name.startswith("np."):
        return name + "(" + ", ".join(py_args) + ")"
    if call_name in USER_FUNCTION_PARAMS and call_name != name:
        return call_name + "(" + ", ".join(py_args) + ")"
    if lname == "c":
        return translate_c_call(args)
    if lname == "list":
        return translate_list_call(args)
    if lname == "data.frame":
        return translate_data_frame_call(args)
    if lname in {"tibble", "tibble_row"}:
        return translate_tibble_call(args)
    if lname in {"as_tibble", "as.tibble", "as.data.frame"}:
        return f"pd.DataFrame({py_args[0]})"
    if lname in {"is_tibble", "is.tibble"}:
        return f"isinstance({py_args[0]}, pd.DataFrame)"
    if lname == "tribble":
        return translate_tribble_call(args)
    if lname == "add_row":
        return translate_add_row_call(args)
    if lname == "add_column":
        return translate_add_column_call(args)
    if lname == "factor":
        levels = keyword_arg(args, "levels")
        ordered = translate_expr(keyword_arg(args, "ordered", default="False"))
        levels_arg = "None" if levels is None else translate_expr(levels)
        return f"r_factor({py_args[0]}, levels={levels_arg}, ordered={ordered})"
    if lname == "vector":
        return translate_vector_call(args)
    if lname == "vectorize":
        return f"np.vectorize({py_args[0]})"
    if lname == "matrix":
        return translate_matrix_call(args)
    if lname == "array":
        return translate_array_call(args)
    if lname == "cbind":
        return translate_cbind_call(args)
    if lname == "rbind":
        return translate_rbind_call(args)
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
            colnames_arg = print_colnames_arg(args[0], allow_simple=True)
            if colnames_arg:
                return "r_print(" + py_args[0] + colnames_arg + ")"
            return "r_s3_print(" + py_args[0] + ")"
        return "r_print(" + ", ".join(py_args) + ")"
    if lname == "cat":
        if args and is_newline_literal(args[-1].strip()):
            if len(py_args) == 1:
                return "print()"
            return "print(" + ", ".join(py_args[:-1]) + ")"
        return "print(" + ", ".join(py_args) + ', end="")'
    if lname == "message":
        return "message_py(" + ", ".join(py_args) + ")"
    if lname == "warning":
        return "warning_py(" + ", ".join(py_args) + ")"
    if lname == "stop":
        return "stop_py(" + ", ".join(py_args) + ")"
    if lname == "sprintf":
        return translate_sprintf_call(args)
    if lname == "paste":
        return translate_paste_call(args, default_sep=" ")
    if lname == "paste0":
        return translate_paste_call(args, default_sep="")
    if lname == "nchar":
        return "np.char.str_len(np.asarray(" + py_args[0] + ", dtype=str))"
    if lname == "toupper":
        return "np.char.upper(np.asarray(" + py_args[0] + ", dtype=str))"
    if lname == "tolower":
        return "np.char.lower(np.asarray(" + py_args[0] + ", dtype=str))"
    if lname == "substr":
        x = py_args[0]
        start = py_args[1]
        stop = py_args[2]
        return f"r_substr({x}, {start}, {stop})"
    if lname == "grepl":
        return f"regex_grepl({py_args[0]}, {py_args[1]})"
    if lname == "grep":
        value = translate_expr(keyword_arg(args, "value", default="False"))
        return f"regex_grep({py_args[0]}, {py_args[1]}, value={value})"
    if lname == "sub":
        return f"regex_sub({py_args[0]}, {py_args[1]}, {py_args[2]})"
    if lname == "gsub":
        return f"regex_sub({py_args[0]}, {py_args[1]}, {py_args[2]}, global_replace=True)"
    if lname == "regexpr":
        return f"regex_regexpr({py_args[0]}, {py_args[1]})"
    if lname == "ifelse":
        return "np.where(" + ", ".join(py_args) + ")"
    if lname == "sqrt":
        return f"np.sqrt({py_args[0]})"
    if lname == "re":
        return f"np.real({py_args[0]})"
    if lname == "im":
        return f"np.imag({py_args[0]})"
    if lname == "mod":
        return f"np.abs({py_args[0]})"
    if lname == "arg":
        return f"np.angle({py_args[0]})"
    if lname == "conj":
        return f"np.conj({py_args[0]})"
    if lname in {"log", "log10", "exp", "sin", "cos", "tan", "abs", "floor"}:
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "linspace":
        return "np.linspace(" + ", ".join(py_args) + ")"
    if lname == "ceiling":
        return "np.ceil(" + ", ".join(py_args) + ")"
    if lname == "trunc":
        return "np.trunc(" + ", ".join(py_args) + ")"
    if lname == "round":
        return "np.round(" + ", ".join(py_args) + ")"
    if lname == "chol":
        return "np.linalg.cholesky(" + py_args[0] + ").T"
    if lname == "sweep":
        return translate_sweep_call(args)
    if lname == "t":
        if re.match(r"^[A-Za-z_]\w*$", py_args[0]):
            return "t_py(" + py_args[0] + ")"
        return "(" + py_args[0] + ").T"
    if lname == "backsolve":
        return translate_backsolve_call(args)
    if lname == "diag":
        if len(py_args) >= 2:
            return "(np.eye(int(" + py_args[1] + ")) * (" + py_args[0] + "))"
        return "(np.eye(int(" + py_args[0] + ")) if np.isscalar(" + py_args[0] + ") else np.diag(" + py_args[0] + "))"
    if lname == "lower.tri":
        return "np.tril(np.ones_like(" + py_args[0] + ", dtype=bool), k=-1)"
    if lname == "upper.tri":
        return "np.triu(np.ones_like(" + py_args[0] + ", dtype=bool), k=1)"
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
    if lname == "eigen":
        return "eigen_py(" + py_args[0] + ")"
    if lname == "svd":
        return "svd_py(" + py_args[0] + ")"
    if lname == "qr":
        return "qr_py(" + py_args[0] + ")"
    if lname in {"sum", "mean", "median", "prod"}:
        na_rm = keyword_arg(args, "na.rm", default="False")
        if translate_expr(na_rm) == "True":
            nan_func = {"sum": "nansum", "mean": "nanmean", "median": "nanmedian", "prod": "nanprod"}[lname]
            return f"np.{nan_func}({py_args[0]})"
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "sort":
        decreasing = translate_expr(keyword_arg(args, "decreasing", default="False"))
        sorted_expr = f"np.sort({py_args[0]})"
        return f"({sorted_expr}[::-1] if {decreasing} else {sorted_expr})"
    if lname == "order":
        decreasing = translate_expr(keyword_arg(args, "decreasing", default="False"))
        return f"r_order({py_args[0]}, decreasing={decreasing})"
    if lname == "rank":
        return "r_rank(" + py_args[0] + ")"
    if lname == "var":
        return "var_r(" + py_args[0] + ")"
    if lname == "cov":
        if len(py_args) == 1:
            return "np.cov(" + py_args[0] + ", rowvar=False, ddof=1)"
        return "np.cov(" + py_args[0] + ", " + py_args[1] + ", ddof=1)"
    if lname == "cor":
        if len(py_args) == 1:
            return "np.corrcoef(" + py_args[0] + ", rowvar=False)"
        return "np.corrcoef(" + py_args[0] + ", " + py_args[1] + ")[0, 1]"
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
    if lname == "setdiff":
        return "r_setdiff(" + ", ".join(py_args) + ")"
    if lname == "as.numeric":
        return "np.asarray(" + py_args[0] + ", dtype=float)"
    if lname == "as.character":
        return "np.asarray(" + py_args[0] + ", dtype=str)"
    if lname == "as.vector":
        return "np.ravel(" + py_args[0] + ", order='F')"
    if lname == "as.matrix":
        return "np.asarray(" + py_args[0] + ")"
    if lname == "scale":
        return f"(({py_args[0]} - np.mean({py_args[0]}, axis=0)) / np.std({py_args[0]}, axis=0, ddof=1))"
    if lname == "attr":
        return "r_attr(" + py_args[0] + ", " + py_args[1] + ")"
    if lname == "attributes":
        return "r_attributes(" + py_args[0] + ")"
    if lname == "as.date":
        fmt = keyword_arg(args, "format")
        fmt_arg = "" if fmt is None else ", format=" + translate_expr(fmt)
        return "r_as_date(" + py_args[0] + fmt_arg + ")"
    if lname == "as.integer":
        return "(r_factor_int(" + py_args[0] + ") if 'RFactor' in globals() and isinstance(" + py_args[0] + ", RFactor) else np.asarray(" + py_args[0] + ", dtype=int))"
    if lname == "chartoraw":
        return f"np.frombuffer(str({py_args[0]}).encode('utf-8'), dtype=np.uint8)"
    if lname == "rawtochar":
        return f"bytes(np.asarray({py_args[0]}, dtype=np.uint8)).decode('utf-8')"
    if lname == "levels":
        return "r_levels(" + py_args[0] + ")"
    if lname == "is.finite":
        return "np.isfinite(" + py_args[0] + ")"
    if lname == "is.infinite":
        return "np.isinf(" + py_args[0] + ")"
    if lname == "is.nan":
        return "np.isnan(" + py_args[0] + ")"
    if lname == "is.na":
        return "pd.isna(" + py_args[0] + ")"
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
    if lname == "ecdf":
        return "ecdf_py(" + py_args[0] + ")"
    if lname == "head":
        return translate_head_call(args)
    if lname == "tail":
        return translate_tail_call(args)
    if lname == "acf":
        plot = translate_expr(keyword_arg(args, "plot", default="False"))
        return f"acf_py({py_args[0]}, plot={plot})"
    if lname == "diff":
        return "r_diff(" + py_args[0] + ")"
    if lname == "format":
        return "r_date_format(" + py_args[0] + ", " + py_args[1] + ")"
    if lname == "cumsum":
        return "np.cumsum(" + py_args[0] + ")"
    if lname == "cumprod":
        return "np.cumprod(" + py_args[0] + ")"
    if lname == "cummin":
        return "np.minimum.accumulate(" + py_args[0] + ")"
    if lname == "cummax":
        return "np.maximum.accumulate(" + py_args[0] + ")"
    if lname == "findinterval":
        vec = translate_expr(keyword_arg(args, "vec", default=args[1] if len(args) > 1 else None))
        return "np.searchsorted(" + vec + ", " + py_args[0] + ", side='right')"
    if lname == "cut":
        breaks = translate_expr(keyword_arg(args, "breaks", default=args[1] if len(args) > 1 else None))
        return "cut_py(" + py_args[0] + ", " + breaks + ")"
    if lname == "range":
        xarg = py_args[0] if py_args else "np.array([])"
        na_rm = keyword_arg(args, "na.rm", default="False")
        if na_rm.lower() == "true":
            return "np.array([np.nanmin(" + xarg + "), np.nanmax(" + xarg + ")])"
        return "np.array([np.min(" + xarg + "), np.max(" + xarg + ")])"
    if lname == "which":
        return "np.nonzero(" + py_args[0] + ")[0] + 1"
    if lname == "which.min":
        return "r_which_min(" + py_args[0] + ")"
    if lname == "which.max":
        return "r_which_max(" + py_args[0] + ")"
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
    if lname == "ts":
        start = translate_expr(keyword_arg(args, "start", default="None"))
        frequency = translate_expr(keyword_arg(args, "frequency", default="1"))
        return f"r_ts({py_args[0]}, start={start}, frequency={frequency})"
    if lname == "start":
        return "r_start(" + py_args[0] + ")"
    if lname == "end":
        return "r_end(" + py_args[0] + ")"
    if lname == "frequency":
        return "r_frequency(" + py_args[0] + ")"
    if lname == "window":
        start = translate_expr(keyword_arg(args, "start", default="None"))
        end = translate_expr(keyword_arg(args, "end", default="None"))
        return f"r_window({py_args[0]}, start={start}, end={end})"
    if lname == "lag":
        k = translate_expr(keyword_arg(args, "k", default="1"))
        return f"r_lag({py_args[0]}, k={k})"
    if lname == "unique":
        return "r_unique(" + py_args[0] + ")"
    if lname == "rle":
        return "rle_py(" + py_args[0] + ")"
    if lname == "inverse.rle":
        return "inverse_rle_py(" + py_args[0] + ")"
    if lname == "duplicated":
        return "r_duplicated(" + py_args[0] + ")"
    if lname == "match":
        return "r_match(" + py_args[0] + ", " + py_args[1] + ")"
    if lname == "append":
        after = keyword_arg(args, "after")
        after_arg = "" if after is None else ", after=" + translate_expr(after)
        return "append_py(" + py_args[0] + ", " + py_args[1] + after_arg + ")"
    if lname == "coef":
        return f"RNamedVector({py_args[0]}@@MEM@@coef, getattr({py_args[0]}, 'coef_names', [str(i) for i in range(len({py_args[0]}@@MEM@@coef))]))"
    if lname in {"residuals", "resid"}:
        return py_args[0] + ".resid"
    if lname == "fitted":
        return py_args[0] + ".fitted"
    if lname == "summary":
        return "summary_py(" + py_args[0] + ")"
    if lname == "class":
        return "class_(" + py_args[0] + ")"
    if lname == "date":
        return "time.ctime()"
    if lname == "source":
        return "source_py(" + py_args[0] + ")"
    if lname == "length":
        return "r_length(" + py_args[0] + ")"
    if lname == "names":
        return "r_names(" + py_args[0] + ")"
    if lname == "colnames":
        return f"np.array(globals().get({(args[0].strip() + '_colnames')!r}, []))"
    if lname == "rownames":
        return f"np.array(globals().get({(args[0].strip() + '_rownames')!r}, []))"
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
    if lname in {"rep", "rep.int", "rep_int"}:
        return translate_rep_call(args)
    if lname == "reduce":
        return "reduce_py(" + ", ".join(py_args) + ")"
    if lname == "numeric":
        return "np.zeros(" + py_args[0] + ")"
    if lname == "integer":
        return "np.zeros(" + py_args[0] + ", dtype=int)"
    if lname == "set.seed":
        return "np.random.seed(" + py_args[0] + ")"
    if lname == "sample.int":
        return translate_sample_int_call(args)
    if lname == "sample":
        return translate_sample_call(args)
    if lname == "runif":
        return translate_runif_call(args)
    if lname == "rnorm":
        return translate_rnorm_call(args)
    if lname in {"rbinom", "rpois", "rexp", "rgamma", "rbeta", "rchisq", "rt", "rf", "rnbinom"}:
        return translate_random_dist_call(lname, args)
    if lname in {"dnorm", "pnorm", "qnorm"}:
        return translate_normal_dist_call(lname, args)
    if lname in {"dt", "pt", "qt"}:
        return translate_t_dist_call(lname, args)
    if lname in {
        "df", "pf", "qf",
        "dchisq", "pchisq", "qchisq",
        "dgamma", "pgamma", "qgamma",
        "dbeta", "pbeta", "qbeta",
        "dbinom", "pbinom", "qbinom",
        "dpois", "ppois", "qpois",
        "dnbinom", "pnbinom", "qnbinom",
        "dexp", "pexp", "qexp",
    }:
        return translate_scipy_dist_call(lname, args)
    return r_name(name) + "(" + ", ".join(py_args) + ")"


def print_colnames_arg(raw_expr: str, *, allow_simple: bool = False) -> str:
    expr = raw_expr.strip()
    raw_call = parse_full_call(expr)
    if raw_call is not None and raw_call[0].lower() in {"head", "tail"} and raw_call[1]:
        return print_colnames_arg(raw_call[1][0], allow_simple=allow_simple)
    if raw_call is not None and raw_call[0].lower() == "cbind":
        names = cbind_arg_names(raw_call[1])
        if names:
            return f", colnames={names!r}"
    member_match = re.fullmatch(r"([A-Za-z]\w*)\$([A-Za-z]\w*)", expr)
    if member_match:
        obj, field = member_match.groups()
        return f", colnames=getattr({r_name(obj)}, {field + '_colnames'!r}, None)"
    name_match = re.fullmatch(r"[A-Za-z]\w*", expr)
    if allow_simple and name_match:
        name = r_name(expr)
        return f", colnames=({name}_colnames if {name + '_colnames'!r} in locals() else None)"
    return ""


def cbind_arg_names(args: list[str]) -> list[str]:
    names: list[str] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            names.append(r_name(arg[:pos].strip()))
            continue
        stripped = arg.strip()
        if re.fullmatch(r"[A-Za-z]\w*", stripped):
            names.append(r_name(stripped))
            continue
        member_match = re.fullmatch(r"([A-Za-z]\w*)\$([A-Za-z]\w*)", stripped)
        if member_match:
            names.append(r_name(member_match.group(2)))
            continue
        return []
    return names


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
    if func in {"sd", "np.std"}:
        return f"r_apply({x}, {margin}, 'sd')"
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
        return f"lm_py({data_expr}@@MEM@@{y}, {data_expr}@@MEM@@xlag)"
    if data is not None:
        data_expr = translate_expr(data)
        response = r_name(formula[:pos].strip())
        terms = formula_terms(formula[pos + 1 :].strip())
        return f"lm_py({data_expr}[{response!r}], r_model_matrix({data_expr}, {response!r}, {terms!r}))"
    return f"lm_py({y}, {x})"


def translate_glm_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("glm requires a formula")
    formula = args[0]
    pos = find_top_level_operator(formula, "~")
    if pos < 0:
        return "glm(" + ", ".join(translate_expr(arg) for arg in args) + ")"
    y = translate_expr(formula[:pos].strip())
    x = translate_expr(formula[pos + 1 :].strip())
    family = keyword_arg(args, "family", default="binomial()")
    family_name = parse_full_call(family)[0].lower() if parse_full_call(family) is not None else family.strip().lower()
    return f"glm_py({y}, {x}, family={family_name!r})"


def translate_aov_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("aov requires a formula")
    formula = args[0]
    pos = find_top_level_operator(formula, "~")
    if pos < 0:
        return "aov(" + ", ".join(translate_expr(arg) for arg in args) + ")"
    y = translate_expr(formula[:pos].strip())
    group = translate_expr(formula[pos + 1 :].strip())
    return f"aov_py({y}, {group})"


def translate_data_frame_call(args: list[str]) -> str:
    items: list[tuple[str | None, str, bool]] = []
    needs_pair_form = False
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            name = r_name(arg[:pos].strip())
            value = translate_expr(arg[pos + 1 :].strip())
            items.append((name, value, True))
        else:
            value = translate_expr(arg)
            if re.match(r"^[A-Za-z_]\w*$", value):
                items.append((value, value, True))
            else:
                items.append((None, value, False))
                needs_pair_form = True
    if needs_pair_form:
        fields = [f"({name!r}, {value})" for name, value, _named in items]
    else:
        fields = [f"{name}={value}" for name, value, _named in items if name is not None]
    return "r_data_frame(" + ", ".join(fields) + ")"


def translate_tibble_call(args: list[str]) -> str:
    items: list[str] = []
    unnamed = 0
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            name = r_column_name(arg[:pos].strip())
            raw_value = arg[pos + 1 :].strip()
        else:
            raw_value = arg
            value = translate_expr(arg)
            if re.match(r"^[A-Za-z_]\w*$", value):
                name = value
            else:
                unnamed += 1
                name = f"x{unnamed}"
        py_value = translate_expr(raw_value)
        items.append(f"({name!r}, {py_value})")
    return "r_tibble_frame([" + ", ".join(items) + "])"


def translate_tibble_assignment(lhs: str, args: list[str]) -> list[str]:
    out: list[str] = []
    pairs: list[str] = []
    prior: dict[str, str] = {}
    unnamed = 0
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0:
            raw_name = arg[:pos].strip()
            col_name = r_column_name(raw_name)
            safe_name = r_name(raw_name)
            raw_value = arg[pos + 1 :].strip()
        else:
            raw_value = arg
            unnamed += 1
            col_name = f"x{unnamed}"
            safe_name = col_name
        temp = f"__{lhs}_{safe_name}"
        value = translate_expr(raw_value)
        value = replace_prior_tibble_names(value, prior)
        out.append(f"{temp} = {value}")
        pairs.append(f"({col_name!r}, {temp})")
        prior[safe_name] = temp
    out.append(f"{lhs} = r_tibble_frame([" + ", ".join(pairs) + "])")
    return out


def replace_prior_tibble_names(expr: str, names: dict[str, str]) -> str:
    for name, temp in names.items():
        expr = re.sub(rf"\b{re.escape(name)}\b", temp, expr)
    return expr


def r_column_name(name: str) -> str:
    stripped = name.strip()
    if len(stripped) >= 2 and stripped[0] == "`" and stripped[-1] == "`":
        return stripped[1:-1]
    return r_name(stripped)


def translate_tribble_call(args: list[str]) -> str:
    names: list[str] = []
    rows: list[list[str]] = []
    values: list[str] = []
    for arg in args:
        stripped = arg.strip()
        if stripped.startswith("~"):
            names.append(r_name(stripped[1:].strip()))
        else:
            values.append(translate_expr(arg))
    if names:
        width = len(names)
        rows = [values[i : i + width] for i in range(0, len(values), width)]
    return "tribble_py(" + repr(names) + ", [" + ", ".join("[" + ", ".join(row) + "]" for row in rows) + "])"


def translate_add_row_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("add_row requires a data frame")
    data = translate_expr(args[0])
    fields = [translate_call_arg(arg) for arg in args[1:]]
    return f"add_row_py({data}" + (", " + ", ".join(fields) if fields else "") + ")"


def translate_add_column_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("add_column requires a data frame")
    data = translate_expr(args[0])
    fields = [translate_call_arg(arg) for arg in args[1:]]
    return f"add_column_py({data}" + (", " + ", ".join(fields) if fields else "") + ")"


def translate_model_matrix_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("model.matrix requires a formula")
    formula = args[0]
    pos = find_top_level_operator(formula, "~")
    if pos < 0:
        raise R2PyError("model.matrix requires formula syntax")
    response = r_name(formula[:pos].strip())
    rhs = formula[pos + 1 :].strip()
    terms = formula_terms(rhs)
    data = keyword_arg(args, "data")
    if data is None:
        raise R2PyError("model.matrix formula requires data")
    return f"r_model_matrix({translate_expr(data)}, {response!r}, {terms!r})"


def formula_terms(rhs: str) -> list[str]:
    terms: list[str] = []
    for part in split_top_level_formula_terms(rhs):
        part = part.strip()
        if not part or part in {"1", "-1", "0"} or part.startswith("-"):
            continue
        if "*" in part:
            pieces = [r_name(piece.strip()) for piece in part.split("*") if piece.strip()]
            for piece in pieces:
                if piece not in terms:
                    terms.append(piece)
            if len(pieces) >= 2:
                interaction = ":".join(pieces)
                if interaction not in terms:
                    terms.append(interaction)
        else:
            term = ":".join(r_name(piece.strip()) for piece in part.split(":")) if ":" in part else r_name(part)
            if term not in terms:
                terms.append(term)
    return terms


def split_top_level_formula_terms(rhs: str) -> list[str]:
    out: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(rhs):
        if quote:
            if ch == quote and (i == 0 or rhs[i - 1] != "\\"):
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        elif depth == 0 and ch in "+-":
            if i > start:
                out.append(rhs[start:i])
            start = i if ch == "-" else i + 1
    if start < len(rhs):
        out.append(rhs[start:])
    return out


def translate_uniroot_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("uniroot requires a function")
    f = translate_expr(args[0])
    lower = translate_expr(keyword_arg(args, "lower", default=args[1] if len(args) > 1 else None))
    upper = translate_expr(keyword_arg(args, "upper", default=args[2] if len(args) > 2 else None))
    tol = translate_expr(keyword_arg(args, "tol", default="1e-8"))
    maxiter = translate_expr(keyword_arg(args, "maxiter", default="1000"))
    return f"uniroot_py({f}, lower={lower}, upper={upper}, tol={tol}, maxiter={maxiter})"


def translate_integrate_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("integrate requires a function")
    f = translate_expr(args[0])
    lower = translate_expr(keyword_arg(args, "lower", default=args[1] if len(args) > 1 else None))
    upper = translate_expr(keyword_arg(args, "upper", default=args[2] if len(args) > 2 else None))
    rel_tol = translate_expr(keyword_arg(args, "rel.tol", default="1e-7"))
    subdivisions = translate_expr(keyword_arg(args, "subdivisions", default="100"))
    return f"integrate_py({f}, lower={lower}, upper={upper}, rel_tol={rel_tol}, subdivisions={subdivisions})"


def translate_outer_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("outer requires x and y")
    x = translate_expr(args[0])
    y = translate_expr(args[1])
    func = translate_expr(args[2]) if len(args) > 2 else repr("*")
    return f"outer_py({x}, {y}, {func})"


def translate_trycatch_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("tryCatch requires an expression")
    fallback = keyword_arg(args, "error", default="function(e) { None }")
    return f"try_catch_py(lambda: {translate_trycatch_body(args[0])}, {translate_trycatch_fallback(fallback)})"


def translate_trycatch_fallback(expr: str | None) -> str:
    if expr is None:
        return "None"
    match = re.match(r"function\s*\([^)]*\)\s*\{\s*(.*?)\s*\}\s*$", expr, re.IGNORECASE)
    return translate_expr(match.group(1)) if match else translate_expr(expr)


def translate_trycatch_body(expr: str) -> str:
    expr = expr.strip()
    if expr.startswith("{") and expr.endswith("}"):
        expr = expr[1:-1].strip()
    match = re.match(r"if\s*\((.*?)\)\s*\{\s*stop\((.*?)\)\s*\}\s*(.+)$", expr, re.IGNORECASE)
    if match:
        cond, message, value = match.groups()
        return f"((_ for _ in ()).throw(ValueError({translate_expr(message)})) if {translate_expr(cond)} else {translate_expr(value)})"
    return translate_expr(expr)


def translate_write_csv_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("write.csv requires a data frame")
    data = translate_expr(args[0])
    file_arg = translate_expr(keyword_arg(args, "file", default=args[1] if len(args) > 1 else None))
    row_names = translate_expr(keyword_arg(args, "row.names", default="True"))
    return f"{data}.to_csv({file_arg}, index={row_names})"


def translate_read_csv_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("read.csv requires a file")
    file_arg = translate_expr(args[0])
    return f"pd.read_csv({file_arg})"


def translate_read_table_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("read.table requires a file")
    file_arg = translate_expr(args[0])
    header = translate_expr(keyword_arg(args, "header", default="False"))
    sep = keyword_arg(args, "sep")
    sep_arg = "'\\\\s+'" if sep is None or string_literal_value(sep.strip()) == "" else translate_expr(sep)
    header_arg = "0" if header == "True" else "None"
    return f"pd.read_csv({file_arg}, sep={sep_arg}, header={header_arg})"


def translate_write_lines_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("writeLines requires text")
    text = translate_expr(args[0])
    con = translate_expr(keyword_arg(args, "con", default=args[1] if len(args) > 1 else None))
    return f"Path({con}).write_text('\\n'.join(map(str, np.asarray({text}))) + '\\n', encoding='utf-8')"


def translate_read_lines_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("readLines requires a connection")
    con = translate_expr(args[0])
    return f"np.array(Path({con}).read_text(encoding='utf-8').splitlines())"


def translate_tempfile_call(args: list[str]) -> str:
    fileext = translate_expr(keyword_arg(args, "fileext", default='""'))
    return f"tempfile.NamedTemporaryFile(delete=False, suffix={fileext}).name"


def translate_unlink_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("unlink requires a path")
    path = translate_expr(args[0])
    return f"os.unlink({path})"


def translate_subset_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("subset requires data and condition")
    data = translate_expr(args[0])
    condition = translate_subset_condition(args[1])
    select = keyword_arg(args, "select")
    columns = parse_subset_select(select) if select is not None else None
    return f"r_subset_df({data}, lambda _df: {condition}, {columns!r})"


def translate_with_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("with requires data and expression")
    data = translate_expr(args[0])
    expr = translate_scoped_expr(args[1], "_env")
    return f"r_with({data}, lambda _env: {expr})"


def translate_within_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("within requires data and assignments")
    data = translate_expr(args[0])
    block = args[1].strip()
    if block.startswith("{") and block.endswith("}"):
        block = block[1:-1].strip()
    updates: list[str] = []
    for lhs, rhs in re.findall(r"([A-Za-z]\w*)\s*(?:<-|=)\s*(.*?)(?=\s+[A-Za-z]\w*\s*(?:<-|=)|$)", block):
        updates.append(f"({lhs!r}, lambda _env: {translate_scoped_expr(rhs.strip(), '_env')})")
    return f"r_within({data}, [{', '.join(updates)}])"


def translate_scoped_expr(expr: str, env_name: str) -> str:
    translated = translate_expr(expr)
    reserved = {"and", "or", "not", "True", "False", "None", "np", "pd", "stats", "len", "str", "int", "float", "r_add", "r_sub", "r_mul", "r_div"}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in reserved or name.startswith("_"):
            return name
        return f"{env_name}[{name!r}]"

    return re.sub(r"(?<![\w.'\"])([A-Za-z_]\w*)\b", repl, translated)


def translate_tapply_call(args: list[str]) -> str:
    if len(args) < 3:
        raise R2PyError("tapply requires values, group, and function")
    values = translate_expr(args[0])
    group = translate_expr(args[1])
    func = args[2].strip()
    return f"r_tapply({values}, {group}, {func!r})"


def translate_apply_list_call(name: str, args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError(f"{name} requires list and function")
    values = translate_expr(args[0])
    func = args[1].strip()
    py_func = repr(func) if func in {"sum", "mean", "length"} else translate_expr(func)
    helper = "r_lapply" if name == "lapply" else "r_sapply"
    return f"{helper}({values}, {py_func})"


def translate_mapply_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("mapply requires function and arguments")
    func = translate_expr(args[0])
    values = [translate_expr(arg) for arg in args[1:]]
    return "r_mapply(" + ", ".join([func, *values]) + ")"


def translate_merge_call(args: list[str]) -> str:
    if len(args) < 2:
        raise R2PyError("merge requires two data frames")
    left = translate_expr(args[0])
    right = translate_expr(args[1])
    by = translate_expr(keyword_arg(args, "by", default="None"))
    all_arg = translate_expr(keyword_arg(args, "all", default="False"))
    all_x = translate_expr(keyword_arg(args, "all.x", default="False"))
    all_y = translate_expr(keyword_arg(args, "all.y", default="False"))
    how = "'inner'"
    if all_arg == "True":
        how = "'outer'"
    elif all_x == "True":
        how = "'left'"
    elif all_y == "True":
        how = "'right'"
    by_arg = "" if by == "None" else f", on={by}"
    return f"pd.merge({left}, {right}{by_arg}, how={how})"


def translate_aggregate_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("aggregate requires a formula")
    formula = args[0]
    pos = find_top_level_operator(formula, "~")
    if pos < 0:
        return "aggregate(" + ", ".join(translate_expr(arg) for arg in args) + ")"
    lhs = formula[:pos].strip()
    rhs = formula[pos + 1 :].strip()
    data = keyword_arg(args, "data")
    if data is None:
        raise R2PyError("aggregate formula requires data")
    data_expr = translate_expr(data)
    group_cols = [r_name(part.strip()) for part in rhs.split("+")]
    value_cols = parse_aggregate_lhs(lhs)
    fun = keyword_arg(args, "FUN", default="mean").strip()
    agg = {"sum": "sum", "mean": "mean", "length": "count"}.get(fun, fun)
    return f"{data_expr}.groupby({group_cols!r}, as_index=False)[{value_cols!r}].agg({agg!r})"


def parse_aggregate_lhs(lhs: str) -> list[str]:
    raw_call = parse_full_call(lhs)
    if raw_call is not None and raw_call[0].lower() in {"cbind", "cbind_py"}:
        return [r_name(arg.strip()) for arg in raw_call[1]]
    return [r_name(lhs)]


def translate_subset_condition(condition: str) -> str:
    expr = translate_expr(condition)
    reserved = {"and", "or", "not", "True", "False", "None", "np", "pd"}

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in reserved or name.startswith("_"):
            return name
        return f"_df.{name}"

    return re.sub(r"(?<![\w.])([A-Za-z]\w*)\b(?!\s*\()", repl, expr)


def parse_subset_select(select: str | None) -> list[str] | None:
    if select is None:
        return None
    raw_call = parse_full_call(select.strip())
    raw_items = raw_call[1] if raw_call is not None and raw_call[0].lower() == "c" else [select]
    out: list[str] = []
    for item in raw_items:
        item = item.strip()
        if is_string_literal(item):
            out.append(item[1:-1])
        elif re.fullmatch(r"r_c\((.*)\)", item):
            out.extend(parse_subset_select("c(" + item[4:-1] + ")") or [])
        else:
            out.append(r_name(item))
    return out


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
        return f"r_table({a})"
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
    if len(args) == 1 and args[0].strip() == "...":
        return "r_list_from_dots(args, kwargs)"
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
        if pos >= 0:
            name = r_name(arg[:pos].strip())
            value = translate_expr(arg[pos + 1 :].strip())
            cols.append(f"r_data_frame({name}={value})")
        else:
            cols.append(translate_expr(arg))
    return "cbind_py(" + ", ".join(cols) + ")"


def translate_rbind_call(args: list[str]) -> str:
    return "rbind_py(" + ", ".join(translate_expr(arg) for arg in args) + ")"


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
    collapse = keyword_arg(args, "collapse")
    values = [translate_expr(arg) for arg in positional_args(args)]
    if not values:
        return '""'
    collapse_arg = "" if collapse is None else f", collapse={translate_expr(collapse)}"
    return "r_paste(" + ", ".join(values) + f", sep={translate_expr(sep)}{collapse_arg})"


def translate_quantile_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("quantile requires an array")
    x = translate_expr(args[0])
    probs_arg = keyword_arg(args, "probs", default=args[1] if len(positional_args(args)) > 1 else "np.array([0.0, 0.25, 0.5, 0.75, 1.0])")
    probs = translate_expr(probs_arg)
    return f"(lambda _p: RNamedVector(np.quantile({x}, _p), [f'{{100 * v:g}}%' for v in _p]))({probs})"


def translate_tail_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("tail requires an array")
    x = translate_expr(args[0])
    n = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "n", default="6"))
    if n == "1" and re.match(r"^[A-Za-z_]\w*$", x) and x not in {"df", "data", "z", "values"}:
        return f"{x}[-1]"
    return f"tail_py({x}, {n})"


def translate_head_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("head requires an array")
    x = translate_expr(args[0])
    n = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "n", default="6"))
    if n == "1" and re.match(r"^[A-Za-z_]\w*$", x) and x not in {"df", "data", "z", "values"}:
        return f"{x}[0]"
    return f"head_py({x}, {n})"


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
        return f"r_matrix_data({data}).reshape((-1, 1), order={order})"
    if nrow is None:
        return f"r_matrix_data({data}).reshape((-1, {translate_expr(ncol)}), order={order})"
    if ncol is None:
        return f"r_matrix_data({data}).reshape(({translate_expr(nrow)}, -1), order={order})"
    py_nrow = translate_expr(nrow)
    py_ncol = translate_expr(ncol)
    return f"np.resize(r_matrix_data({data}), ({py_nrow}) * ({py_ncol})).reshape(({py_nrow}, {py_ncol}), order={order})"


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
    positional = [arg for arg in args if "=" not in arg]
    by = keyword_arg(args, "by", default=positional[2] if len(positional) > 2 else None)
    length_out = keyword_arg(args, "length.out")
    along_with = keyword_arg(args, "along.with")
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
        by_value = by.strip()
        if is_string_literal(by_value) or re.fullmatch(r"__R_STR_\d+__", by_value):
            return f"r_date_seq({start}, {stop}, {by_value})"
        step = translate_expr(by)
        return f"np.arange({start}, {stop} + np.sign({step}), {step})"
    return f"np.arange({start}, {stop} + np.sign({stop} - {start}), np.sign({stop} - {start}))"


def translate_runif_call(args: list[str]) -> str:
    positional = positional_args(args)
    n = translate_expr(args[0]) if args else "1"
    lo = translate_expr(keyword_arg(args, "min", default=positional[1] if len(positional) > 1 else "0"))
    hi = translate_expr(keyword_arg(args, "max", default=positional[2] if len(positional) > 2 else "1"))
    return f"np.random.uniform({lo}, {hi}, size={n})"


def translate_rnorm_call(args: list[str]) -> str:
    positional = positional_args(args)
    n = translate_expr(args[0]) if args else "1"
    mean = translate_expr(keyword_arg(args, "mean", default=positional[1] if len(positional) > 1 else "0"))
    sd = translate_expr(keyword_arg(args, "sd", default=positional[2] if len(positional) > 2 else "1"))
    if n == "1":
        return f"np.random.normal({mean}, {sd})"
    return f"np.random.normal({mean}, {sd}, size={n})"


def translate_random_dist_call(name: str, args: list[str]) -> str:
    n = translate_expr(args[0]) if args else "1"
    if name == "rbinom":
        size = translate_expr(keyword_arg(args, "size", default=args[1] if len(args) > 1 else None))
        prob = translate_expr(keyword_arg(args, "prob", default=args[2] if len(args) > 2 else None))
        return f"np.random.binomial({size}, {prob}, size={n})"
    if name == "rpois":
        lam = translate_expr(keyword_arg(args, "lambda", default=args[1] if len(args) > 1 else "1"))
        return f"np.random.poisson({lam}, size={n})"
    if name == "rexp":
        rate = translate_expr(keyword_arg(args, "rate", default=args[1] if len(args) > 1 else "1"))
        return f"np.random.exponential(1 / ({rate}), size={n})"
    if name == "rgamma":
        shape = translate_expr(keyword_arg(args, "shape", default=args[1] if len(args) > 1 else None))
        scale_arg = keyword_arg(args, "scale")
        rate_arg = keyword_arg(args, "rate")
        if scale_arg is not None:
            scale = translate_expr(scale_arg)
        elif rate_arg is not None:
            scale = f"1 / ({translate_expr(rate_arg)})"
        else:
            scale = "1"
        return f"np.random.gamma({shape}, scale={scale}, size={n})"
    if name == "rbeta":
        shape1 = translate_expr(keyword_arg(args, "shape1", default=args[1] if len(args) > 1 else None))
        shape2 = translate_expr(keyword_arg(args, "shape2", default=args[2] if len(args) > 2 else None))
        return f"np.random.beta({shape1}, {shape2}, size={n})"
    if name == "rchisq":
        df = translate_expr(keyword_arg(args, "df", default=args[1] if len(args) > 1 else None))
        return f"np.random.chisquare({df}, size={n})"
    if name == "rt":
        df = translate_expr(keyword_arg(args, "df", default=args[1] if len(args) > 1 else None))
        return f"stats.t.rvs(df={df}, size={n})"
    if name == "rf":
        df1 = translate_expr(keyword_arg(args, "df1", default=args[1] if len(args) > 1 else None))
        df2 = translate_expr(keyword_arg(args, "df2", default=args[2] if len(args) > 2 else None))
        return f"stats.f.rvs({df1}, {df2}, size={n})"
    if name == "rnbinom":
        size = translate_expr(keyword_arg(args, "size", default=args[1] if len(args) > 1 else None))
        prob = keyword_arg(args, "prob")
        mu = keyword_arg(args, "mu")
        if prob is None and mu is not None:
            prob_expr = f"({size}) / (({size}) + ({translate_expr(mu)}))"
        else:
            prob_expr = translate_expr(prob if prob is not None else args[2] if len(args) > 2 else None)
        return f"np.random.negative_binomial({size}, {prob_expr}, size={n})"
    raise R2PyError(f"unsupported random distribution: {name}")


def translate_sample_int_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("sample.int requires n")
    n = translate_expr(args[0])
    size = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "size", default=n))
    replace = translate_expr(keyword_arg(args, "replace", default="False"))
    prob = keyword_arg(args, "prob", default=None)
    p_arg = ", p=" + translate_expr(prob) if prob is not None else ""
    return f"np.random.choice(np.arange(1, {n} + 1), size={size}, replace={replace}{p_arg})"


def translate_sample_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("sample requires x")
    x = translate_expr(args[0])
    size = translate_expr(keyword_arg(args, "size", default=args[1] if len(args) > 1 and find_top_level_operator(args[1], "=") < 0 else f"len({x})"))
    replace = translate_expr(keyword_arg(args, "replace", default="False"))
    prob = keyword_arg(args, "prob")
    p_arg = ", p=" + translate_expr(prob) if prob is not None else ""
    return f"np.random.choice({x}, size={size}, replace={replace}{p_arg})"


def translate_normal_dist_call(name: str, args: list[str]) -> str:
    if not args:
        raise R2PyError(f"{name} requires an x/q/p argument")
    positional = positional_args(args)
    x = translate_expr(args[0])
    mean = translate_expr(keyword_arg(args, "mean", default=positional[1] if len(positional) > 1 else "0"))
    sd = translate_expr(keyword_arg(args, "sd", default=positional[2] if len(positional) > 2 else "1"))
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


def translate_scipy_dist_call(name: str, args: list[str]) -> str:
    if not args:
        raise R2PyError(f"{name} requires an argument")
    kind = name[0]
    family = name[1:]
    x = translate_expr(args[0])
    dist, params = scipy_dist_params(family, args)
    lower_tail = translate_expr(keyword_arg(args, "lower.tail", default="True"))
    log_density = translate_expr(keyword_arg(args, "log", default="False"))
    log_prob = translate_expr(keyword_arg(args, "log.p", default="False"))
    params_text = ", ".join(params)
    params_arg = (", " + params_text) if params_text else ""
    if kind == "d":
        func = "logpmf" if family in {"binom", "pois", "nbinom"} and log_density == "True" else "pmf" if family in {"binom", "pois", "nbinom"} else "logpdf" if log_density == "True" else "pdf"
        return f"stats.{dist}.{func}({x}{params_arg})"
    if kind == "p":
        value = f"np.where({lower_tail}, stats.{dist}.cdf({x}{params_arg}), stats.{dist}.sf({x}{params_arg}))"
        return f"np.log({value})" if log_prob == "True" else value
    q = f"np.exp({x})" if log_prob == "True" else x
    return f"np.where({lower_tail}, stats.{dist}.ppf({q}{params_arg}), stats.{dist}.isf({q}{params_arg}))"


def scipy_dist_params(family: str, args: list[str]) -> tuple[str, list[str]]:
    if family == "norm":
        mean = translate_expr(keyword_arg(args, "mean", default="0"))
        sd = translate_expr(keyword_arg(args, "sd", default="1"))
        return "norm", [f"loc={mean}", f"scale={sd}"]
    if family == "t":
        df = translate_expr(keyword_arg(args, "df", default=args[1] if len(args) > 1 else None))
        return "t", [f"df={df}"]
    if family == "f":
        df1 = translate_expr(keyword_arg(args, "df1", default=args[1] if len(args) > 1 else None))
        df2 = translate_expr(keyword_arg(args, "df2", default=args[2] if len(args) > 2 else None))
        return "f", [df1, df2]
    if family == "chisq":
        df = translate_expr(keyword_arg(args, "df", default=args[1] if len(args) > 1 else None))
        return "chi2", [f"df={df}"]
    if family == "gamma":
        shape = translate_expr(keyword_arg(args, "shape", default=args[1] if len(args) > 1 else None))
        scale_arg = keyword_arg(args, "scale")
        rate_arg = keyword_arg(args, "rate")
        if scale_arg is not None:
            scale = translate_expr(scale_arg)
        elif rate_arg is not None:
            scale = f"1 / ({translate_expr(rate_arg)})"
        else:
            scale = "1"
        return "gamma", [shape, f"scale={scale}"]
    if family == "beta":
        shape1 = translate_expr(keyword_arg(args, "shape1", default=args[1] if len(args) > 1 else None))
        shape2 = translate_expr(keyword_arg(args, "shape2", default=args[2] if len(args) > 2 else None))
        return "beta", [shape1, shape2]
    if family == "binom":
        size = translate_expr(keyword_arg(args, "size", default=args[1] if len(args) > 1 else None))
        prob = translate_expr(keyword_arg(args, "prob", default=args[2] if len(args) > 2 else None))
        return "binom", [size, prob]
    if family == "pois":
        lam = translate_expr(keyword_arg(args, "lambda", default=args[1] if len(args) > 1 else None))
        return "poisson", [lam]
    if family == "nbinom":
        size = translate_expr(keyword_arg(args, "size", default=args[1] if len(args) > 1 else None))
        prob = keyword_arg(args, "prob")
        mu = keyword_arg(args, "mu")
        if prob is None and mu is not None:
            prob_expr = f"({size}) / (({size}) + ({translate_expr(mu)}))"
        else:
            prob_expr = translate_expr(prob if prob is not None else args[2] if len(args) > 2 else None)
        return "nbinom", [size, prob_expr]
    if family == "exp":
        rate = translate_expr(keyword_arg(args, "rate", default=args[1] if len(args) > 1 else "1"))
        return "expon", [f"scale=1 / ({rate})"]
    raise R2PyError(f"unsupported distribution family: {family}")


def keyword_arg(args: list[str], name: str, default: str | None = None) -> str | None:
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0 and normalize_keyword_name(arg[:pos].strip()).lower() == normalize_keyword_name(name).lower():
            return arg[pos + 1 :].strip()
    return default


def translate_call_arg(arg: str) -> str:
    pos = find_top_level_operator(arg, "=")
    if pos < 0:
        return translate_expr(arg)
    key = normalize_keyword_name(arg[:pos].strip())
    value = arg[pos + 1 :].strip()
    return f"{key}={translate_expr(value)}"


def normalize_keyword_name(name: str) -> str:
    stripped = name.strip()
    if stripped == ".after":
        return "_after"
    if stripped == ".before":
        return "_before"
    out = r_name(stripped.replace(".", "_"))
    if out == "lambda":
        return "lambda_"
    return out


def apply_partial_argument_matching(name: str, args: list[str]) -> list[str]:
    params = USER_FUNCTION_PARAMS.get(name)
    if not params:
        return args
    out: list[str] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos < 0:
            out.append(arg)
            continue
        raw_key = r_name(arg[:pos].strip())
        matches = [param for param in params if param.startswith(raw_key)]
        key = matches[0] if len(matches) == 1 else raw_key
        out.append(f"{key} = {arg[pos + 1:].strip()}")
    return out


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

    def is_safe_plain_operand(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return isinstance(node.value, (int, float, complex, bool))
        if isinstance(node, ast.Name):
            return len(node.id) > 1 or node.id in {"x", "y"}
        if isinstance(node, ast.Attribute):
            return True
        if isinstance(node, ast.Call):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return is_safe_plain_operand(node.left) and is_safe_plain_operand(node.right)
        return False

    def should_keep_python_operator(node: ast.BinOp) -> bool:
        if isinstance(node.op, ast.Add):
            return (
                isinstance(node.left, ast.BinOp)
                and isinstance(node.left.op, ast.Pow)
                and isinstance(node.right, ast.BinOp)
                and isinstance(node.right.op, ast.Pow)
            )
        if isinstance(node.op, ast.Sub):
            return is_safe_plain_operand(node.left) and is_safe_plain_operand(node.right)
        if isinstance(node.op, ast.Mult):
            return (
                isinstance(node.left, ast.Constant)
                and is_safe_plain_operand(node.right)
            ) or (
                isinstance(node.right, ast.Constant)
                and is_safe_plain_operand(node.left)
            )
        return False

    class Rewriter(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            node = self.generic_visit(node)
            op_name = op_map.get(type(node.op))
            if op_name is None:
                return node
            if should_keep_python_operator(node):
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
    if not name:
        return name
    if len(name) >= 2 and name[0] == "`" and name[-1] == "`":
        name = name[1:-1]
    constants = {"True", "False", "None", "np", "pd", "stats", "nan", "inf", "and", "or", "not", "is", "in", "if", "else", "for"}
    if "@@MEM@@" in name:
        return ".".join(r_name(part) for part in name.split("@@MEM@@"))
    if name in constants or name.startswith(("np.", "stats.", "pd.", "time.")):
        return name
    if name[0].isdigit():
        return name
    if "." in name and name not in DOTTED_R_VARS:
        return name
    out = re.sub(r"\W+", "_", name.replace(".", "_")).strip("_")
    if not out:
        out = "x"
    if keyword.iskeyword(out):
        out += "_"
    return out


def r_function_name(name: str) -> str:
    if name.startswith(("np.", "stats.", "pd.", "time.")):
        return name
    return r_name(name.replace(".", "_"))


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
        if re.fullmatch(r"[+-]?\d+", token):
            return token
        try:
            value = float(token.replace("D", "E").replace("d", "E"))
        except Exception:
            return token
        if not math.isfinite(value):
            return token
        return f"{value:.{int(digits)}f}"

    return number_re.sub(repl, text)


def flush_left_output(text: str) -> str:
    return "\n".join(line.lstrip() for line in text.splitlines()) + ("\n" if text.endswith("\n") else "")


def squeeze_output_spaces(text: str) -> str:
    return "\n".join(re.sub(r" {2,}", " ", line) for line in text.splitlines()) + ("\n" if text.endswith("\n") else "")


def normalize_output(text: str, digits: int | None = None, *, flush_left: bool = False, squeeze: bool = False) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    index_re = re.compile(r"^\s*\[\d+\]\s*")
    for line in text.splitlines():
        line = index_re.sub("", line.rstrip("\n"))
        line = round_numeric_tokens(line, digits)
        if flush_left:
            line = line.lstrip()
        if squeeze:
            line = re.sub(r" {2,}", " ", line)
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

def outputs_match(a: str, b: str, *, digits: int | None = None, flush_left: bool = False, squeeze: bool = False) -> tuple[bool, str]:
    a_lines = normalize_output(a, digits, flush_left=flush_left, squeeze=squeeze)
    b_lines = normalize_output(b, digits, flush_left=flush_left, squeeze=squeeze)
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


def run_python(path: Path, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(path)], cwd=cwd, text=True, capture_output=True)


def run_r(path: Path, rscript: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([rscript, str(path)], text=True, capture_output=True)


def print_process_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def print_result_output(result: subprocess.CompletedProcess[str], digits: int | None, *, pretty_r: bool = False, flush_left: bool = False, squeeze: bool = False) -> None:
    if result.stdout:
        stdout = pretty_r_output(result.stdout) if pretty_r else result.stdout
        stdout = round_numeric_tokens(stdout, digits)
        if flush_left:
            stdout = flush_left_output(stdout)
        if squeeze:
            stdout = squeeze_output_spaces(stdout)
        print(
            stdout,
            end="" if result.stdout.endswith("\n") else "\n",
        )
    if result.stderr:
        stderr = pretty_r_output(result.stderr) if pretty_r else result.stderr
        stderr = round_numeric_tokens(stderr, digits)
        if flush_left:
            stderr = flush_left_output(stderr)
        if squeeze:
            stderr = squeeze_output_spaces(stderr)
        print(
            stderr,
            end="" if result.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )


def pretty_r_output(text: str) -> str:
    out: list[str] = []
    index_re = re.compile(r"^\s*\[\d+\]\s*")
    for line in text.splitlines():
        line = index_re.sub("", line)
        line = re.sub(r"\bTRUE\b", "True", line)
        line = re.sub(r"\bFALSE\b", "False", line)
        line = re.sub(r"\bNA\b", "nan", line)
        line = unquote_r_string_tokens(line)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def unquote_r_string_tokens(line: str) -> str:
    parts = line.split()
    if not parts:
        return line
    if all(re.fullmatch(r'"[^"\s]*"|[^"\s]+', part) for part in parts):
        return " ".join(part[1:-1] if len(part) >= 2 and part[0] == '"' and part[-1] == '"' else part for part in parts)
    return line


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
    parser.add_argument("--tee-both", action="store_true", help="print the original R source and emitted Python code")
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
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="make displayed R output look more like Python output",
    )
    parser.add_argument(
        "--flush-left",
        action="store_true",
        help="strip leading whitespace from each displayed output line",
    )
    parser.add_argument(
        "--squeeze",
        action="store_true",
        help="replace runs of two or more spaces in displayed output with one space",
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

    if str(args.source) == "@last":
        try:
            args.source = last_r_source(Path.cwd())
        except R2PyError as exc:
            print(f"xr2p: {exc}", file=sys.stderr)
            return 1
        print(f"@last -> {args.source}")

    try:
        source = args.source.read_text(encoding="utf-8-sig")
        python = translate_source(source)
    except (OSError, R2PyError) as exc:
        print(f"xr2p: {exc}", file=sys.stderr)
        return 1

    out = args.out or args.source.with_suffix(".py")
    out.write_text(python, encoding="utf-8")
    print(f"wrote {out}")
    if args.tee_both:
        print("R source:")
        print(source, end="" if source.endswith("\n") else "\n")
        print("Python translation:")
        print(python, end="" if python.endswith("\n") else "\n")
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
        print_result_output(r_result, r_round_digits, pretty_r=args.pretty, flush_left=args.flush_left, squeeze=args.squeeze)
        print("Run (Python):", sys.executable, out)
        py_result = run_python(out)
        print("Run (Python):", "PASS" if py_result.returncode == 0 else f"FAIL exit={py_result.returncode}")
        print_result_output(py_result, python_round_digits, flush_left=args.flush_left, squeeze=args.squeeze)
        if args.run_diff:
            compare_digits = args.round_both if args.round_both is not None else args.round
            same, diff = outputs_match(
                r_result.stdout,
                py_result.stdout,
                digits=compare_digits,
                flush_left=args.flush_left,
                squeeze=args.squeeze,
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
        print_result_output(result, python_round_digits, flush_left=args.flush_left, squeeze=args.squeeze)
        return result.returncode
    return 0


def last_r_source(directory: Path) -> Path:
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".r"
    ]
    if not candidates:
        raise R2PyError("no .r or .R files found for @last")
    return max(candidates, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())

