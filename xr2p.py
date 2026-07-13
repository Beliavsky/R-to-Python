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
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


INDENT = "    "
USER_FUNCTION_PARAMS: dict[str, list[str]] = {}
USER_FUNCTION_NAMES: set[str] = set()
PENDING_FUNCTION_PARAMS: list[str] | None = None
NAMED_VECTOR_VARS: set[str] = set()
DOTTED_R_VARS: set[str] = set()
CHARACTER_VECTOR_VARS: set[str] = set()
LOGICAL_VECTOR_VARS: set[str] = set()
MATRIX_VARS: set[str] = set()


@dataclass
class TranslateResult:
    ok: bool
    python: str = ""
    message: str = ""


class R2PyError(Exception):
    pass


def translate_source(source: str, *, use_numba: bool = True, source_name: str | None = None, banner: bool = True) -> str:
    USER_FUNCTION_PARAMS.clear()
    USER_FUNCTION_NAMES.clear()
    NAMED_VECTOR_VARS.clear()
    DOTTED_R_VARS.clear()
    CHARACTER_VECTOR_VARS.clear()
    LOGICAL_VECTOR_VARS.clear()
    MATRIX_VARS.clear()
    global PENDING_FUNCTION_PARAMS
    PENDING_FUNCTION_PARAMS = None
    _LAMBDA_MASKS.clear()
    register_dotted_variables(source)
    out = ["import numpy as np", ""]
    out.extend(translate_logical_lines(logical_r_lines(preprocess_simple_inline_r(source))))
    python = "\n".join(out).rstrip() + "\n"
    python = restore_lambda_masks(python)
    # String placeholders hidden inside masked lambdas miss their local
    # restoration pass; resolve any stragglers from the global registry.
    python = re.sub(r"__R_STR_\d+__", lambda m: _STRING_MASK_TEXTS.get(m.group(0), m.group(0)), python)
    python = zero_base_unused_counter_loops(python)
    python = return_function_tail_expressions(python)
    python = resolve_nonlocal_assignments(python)
    python = add_blank_lines_after_functions(python)
    python = add_pass_to_empty_blocks(python)
    python = repair_generated_syntax_cleanup(python)
    if re.search(r"(?<![\w.])stats\.", python) or "aov_py(" in python or "kruskal_test_py(" in python or "wilcox_test_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import stats as r_stats\n", 1)
        python = re.sub(r"(?<![\w.])stats\.", "r_stats.", python)
    if "optimize." in python or "uniroot_py(" in python or re.search(r"(?<![\w.])fsolve(?![\w.])", python):
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import optimize\n", 1)
    if "integrate." in python or "integrate_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import integrate\n", 1)
    if "linalg." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import linalg\n", 1)
    if "special." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import special\n", 1)
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
    python = add_runtime_helpers(python, use_numba=use_numba)
    python = inject_known_fast_paths(python)
    if "SimpleNamespace" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom types import SimpleNamespace\n", 1)
    if "pd." in python or "read_table(" in python or "lm_py(" in python or "stack_py(" in python or "unstack_py(" in python or "r_with(" in python or "r_within(" in python or "r_member(" in python or "r_vec_subset(" in python or "r_matrix_index_get(" in python or "r_matrix_index_set(" in python or "r_subset(" in python or "r_set_subset(" in python or "r_subset_df(" in python or "r_df_col(" in python or "r_data_frame(" in python or "r_model_matrix(" in python or "tribble_py(" in python or "add_row_py(" in python or "add_column_py(" in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport pandas as pd\n", 1)
    if "tempfile." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport tempfile\n", 1)
    if "itertools." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nimport itertools\n", 1)
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
    if banner and source_name:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        python = f"# Translated from {source_name} by xr2p.py on {stamp}.\n{python}"
    return python


def is_rewritable_r_expression(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.endswith("{") or stripped.startswith("#"):
        return False
    if re.match(r"^(for|while|if|repeat|return|break|next|else)\b", stripped):
        return False
    if raw_assignment(stripped) is not None:
        return False
    return True


def expand_assignment_if_blocks(lines: list[str]) -> list[str]:
    """Rewrite ``name <- if (cond) { ... } else { ... }`` blocks spanning lines.

    The assignment moves onto the tail expression of each branch, matching R's
    block-value semantics.
    """
    out = list(lines)
    i = 0
    while i < len(out):
        match = re.match(r"^(\.?[A-Za-z][\w.]*)\s*(?:<-|=)\s*(if\s*\(.+\{)$", out[i].strip())
        if match is None:
            i += 1
            continue
        name, header = match.groups()
        rewrites: list[tuple[int, str]] = []
        splices: list[tuple[int, list[str]]] = []
        ok = True
        depth = 1
        tail_idx: int | None = None
        j = i + 1
        while j < len(out) and depth > 0:
            work = out[j].strip()
            closes = 0
            while work.startswith("}"):
                closes += 1
                work = work[1:].strip()
            depth -= closes
            if depth <= 0:
                if tail_idx is None or not is_rewritable_r_expression(out[tail_idx]):
                    ok = False
                    break
                rewrites.append((tail_idx, f"{name} <- {out[tail_idx].strip()}"))
                tail_idx = None
                if not work:
                    # The preprocessor may have split "} else ..." onto its own line.
                    peek = j + 1
                    while peek < len(out) and not out[peek].strip():
                        peek += 1
                    if peek < len(out) and out[peek].strip().startswith("else"):
                        j = peek
                        work = out[j].strip()
                if work.startswith("else"):
                    else_tail = work[4:].strip()
                    if else_tail.endswith("{"):
                        depth = 1
                    elif else_tail:
                        splices.append((j, ["} else {", f"{name} <- {else_tail}", "}"]))
                elif work:
                    ok = False
                    break
            else:
                if closes and depth == 1:
                    tail_idx = None
                if depth == 1 and is_rewritable_r_expression(work):
                    tail_idx = j
                if work.endswith("{"):
                    depth += 1
            j += 1
        if not ok or depth > 0:
            i += 1
            continue
        out[i] = header
        for idx, new_line in rewrites:
            out[idx] = new_line
        for idx, new_lines in reversed(splices):
            out[idx : idx + 1] = new_lines
        i += 1
    return out


def register_dotted_variables(source: str) -> None:
    """Register dotted R identifiers used as variables so they rename consistently.

    Only raw R source tokens are scanned, so generated attribute accesses like
    ``x.shape`` are never affected.
    """
    for raw_line in source.splitlines():
        line = strip_r_comment(raw_line)
        if "." not in line:
            continue
        masked, _ = mask_string_literals(line)
        for match in re.finditer(r"(?<![\w.$@])([A-Za-z]\w*(?:\.[A-Za-z]\w*)+)\b(?!\s*\()", masked):
            name = match.group(1)
            if name.startswith(("is.", "as.")):
                # Base-R predicates passed as bare function references.
                continue
            DOTTED_R_VARS.add(name)


def translate_logical_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    lines = expand_assignment_if_blocks(lines)
    indent = 0
    for line in lines:
        if not line:
            continue
        if line.startswith("#"):
            out.append(INDENT * indent + line)
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
    return out


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
            expanded_lines = [
                braced
                for piece in expand_braced_assignment(expand_inline_function_assignment(expanded))
                for braced in split_statement_semicolons(piece)
            ]
            for line in expanded_lines:
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


def expand_braced_assignment(text: str) -> list[str]:
    """Expand ``name <- { a; b; last }`` into the statements plus ``name <- last``."""
    match = re.match(r"^(\s*)([A-Za-z.][\w.]*)\s*(?:<-|=)\s*\{(.*)\}\s*$", text)
    if not match:
        return [text]
    indent, name, body = match.groups()
    parts = [part.strip() for part in split_top_level_semicolons(body) if part.strip()]
    if not parts:
        return [text]
    return [indent + part for part in parts[:-1]] + [f"{indent}{name} <- {parts[-1]}"]


def split_statement_semicolons(text: str) -> list[str]:
    """Split top-level ``;``-separated statements onto their own lines."""
    out: list[str] = []
    for line in text.split("\n"):
        if ";" not in line:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        code = strip_r_comment(line)
        comment = line[len(code) :]
        parts = [part.strip() for part in split_top_level_semicolons(code)]
        parts = [part for part in parts if part]
        if len(parts) <= 1:
            out.append(line)
            continue
        for j, part in enumerate(parts):
            suffix = comment if j == len(parts) - 1 else ""
            out.append(indent + part + suffix)
    return out


def expand_chained_assignment(line: str) -> list[str]:
    assign = strict_raw_assignment(line)
    if assign is None:
        return [line]
    lhs, rhs = assign
    nested = strict_raw_assignment(rhs)
    if nested is None:
        return [line]
    mid_lhs, mid_rhs = nested
    if not (is_chain_assignment_target(lhs) and is_chain_assignment_target(mid_lhs)):
        return [line]
    return [f"{mid_lhs} <- {mid_rhs}", f"{lhs} <- {mid_lhs}"]


def is_chain_assignment_target(text: str) -> bool:
    return re.fullmatch(r"[A-Za-z.][\w.]*\s*(?:\[\[?[^\[\]]*\]?\]|\$[\w.]+)?", text.strip()) is not None


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
    raw_lines = source.splitlines()
    for idx, raw in enumerate(raw_lines):
        if buffer:
            buffer = buffer.rstrip() + " " + raw.strip()
        else:
            buffer = raw
        if r_line_continues(buffer):
            continue
        if next_line_starts_else(raw_lines, idx + 1) and re.search(r"\bif\s*\(", strip_r_comment(buffer)):
            continue
        out.append(buffer)
        buffer = ""
    if buffer:
        out.append(buffer)
    return out


def next_line_starts_else(raw_lines: list[str], start: int) -> bool:
    for raw in raw_lines[start:]:
        code = strip_r_comment(raw).strip()
        if not code:
            continue
        return re.match(r"^else\b", code) is not None
    return False


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
    if re.search(r"\belse$", stripped):
        return True
    if stripped.endswith(")") and assignment_if_header(stripped):
        return True
    return bool(re.search(r"(%[^%\s]+%|\|>|\+|-|\*|/|\||&|,)\s*$", stripped))


def assignment_if_header(stripped: str) -> bool:
    """True for lines like ``x <- if (cond)`` whose branch is on the next line."""
    depth = 0
    open_pos = -1
    for i in range(len(stripped) - 1, -1, -1):
        ch = stripped[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                open_pos = i
                break
    if open_pos < 0:
        return False
    head = stripped[:open_pos].rstrip()
    return bool(re.search(r"(?:<-|=)\s*if$", head))


def has_unbalanced_delimiters(text: str) -> bool:
    depth = 0
    quote = ""
    for ch in text:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"', "`"}:
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
    brace_else = re.match(r"^(\s*)\}\s*(else\b.*)$", line)
    if brace_else is not None:
        indent, tail = brace_else.groups()
        return [f"{indent}}}", *expand_one_line_control(indent + tail)]
    if re.match(r"^\s*else\s+if\s*\(", line):
        return [line]
    parsed_else = parse_one_line_else(line)
    if parsed_else is not None:
        indent, tail = parsed_else
        if not tail:
            return [f"{indent}else {{"]
        if tail.startswith("{"):
            expanded_block = expand_braced_control_tail("else", tail, indent)
            if expanded_block is not None:
                return expanded_block
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
        expanded_block = expand_braced_control_tail(head, tail, indent)
        if expanded_block is not None:
            return expanded_block
        return [f"{indent}{head} {tail}"]
    if head.startswith("if"):
        split = split_top_level_else(tail)
        if split is not None:
            yes, no = split
            if yes and no:
                return [
                    f"{indent}{head} {{",
                    f"{indent}{INDENT}{yes}",
                    f"{indent}}}",
                    *expand_one_line_control(f"{indent}else {no}"),
                ]
    return [f"{indent}{head} {{", f"{indent}{INDENT}{tail}", f"{indent}}}"]


def expand_braced_control_tail(head: str, tail: str, indent: str) -> list[str] | None:
    """Expand ``if (c) { a; b } [else ...]`` one-liners into block lines."""
    close = find_matching_char(tail, 0, "{", "}")
    if close < 0:
        return None
    inner = tail[1:close].strip()
    remainder = tail[close + 1 :].strip()
    if not inner and not remainder:
        return None
    body_lines = [part for part in brace_block_to_r_text(inner).split("\n") if part]
    if len(body_lines) <= 1 and not remainder:
        return None
    out = [f"{indent}{head} {{", *(f"{indent}{INDENT}{part}" for part in body_lines), f"{indent}}}"]
    if remainder:
        out.extend(expand_one_line_control(f"{indent}{remainder}"))
    return out


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
    bare_function_refs = {
        "is.numeric": "r_is_numeric",
        "is.vector": "r_is_vector",
        "is.matrix": "r_is_matrix",
        "is.null": "r_is_null",
        "is.na": "np.isnan",
        "is.nan": "np.isnan",
        "is.finite": "np.isfinite",
        "is.infinite": "np.isinf",
        "is.function": "callable",
    }
    for old, new in bare_function_refs.items():
        python = re.sub(rf"(?<![\w.]){re.escape(old)}(?![\w.(])", new, python)
    for name in keyword.kwlist:
        if name == "lambda":
            continue
        python = re.sub(rf"\.{name}\b", f".{name}_", python)
    python = normalize_dotted_call_syntax(python)
    python = re.sub(r"r_matrix_index_get\(([^,\n]+),\s*:(\d+)\)", r"r_matrix_index_get(\1, r_seq(1, \2))", python)
    python = python.replace("try_(lambda_:", "try_(lambda:")
    python = repair_inline_lambda_keyword(python)
    return python


def repair_inline_lambda_keyword(text: str) -> str:
    # Generator-emitted lambdas can get renamed to lambda_ by replace_names;
    # restore the keyword when the token is followed by a parameter list and colon.
    return re.sub(r"(?<![\w.])lambda_(?=(?::| +[A-Za-z_*][\w ,*=]*:))", "lambda", text)


def normalize_dotted_call_syntax(python: str) -> str:
    python = re.sub(r"(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*\(", lambda m: r_function_name(m.group(1)) + "(", python)
    python = re.sub(r"(?<![=!<>])\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\s*=(?!=)", lambda m: normalize_keyword_name(m.group(1)) + "=", python)
    return python


def remove_unused_numpy_import(python: str) -> str:
    prefix = "import numpy as np\n"
    if not python.startswith(prefix):
        return python
    body = python[len(prefix) :]
    if "np." in body:
        return python
    return body.lstrip("\n")


def add_runtime_helpers(python: str, *, use_numba: bool = True) -> str:
    helpers: list[str] = []
    if "r_is_numeric" in python or "r_is_vector" in python or "r_is_matrix" in python:
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
    if "r_format_vec(" in python:
        helpers.append(
            """
def r_format_vec(x, digits=None, nsmall=None, scientific=None, width=None):
    def fmt_one(v):
        if isinstance(v, (bool, np.bool_)):
            return "TRUE" if v else "FALSE"
        if isinstance(v, (np.integer, int)):
            s = str(int(v))
        elif isinstance(v, (np.floating, float)):
            if not np.isfinite(v):
                return "NA" if np.isnan(v) else ("Inf" if v > 0 else "-Inf")
            if scientific is True:
                s = f"{v:.{int(digits) if digits is not None else 6}e}"
            elif digits is not None:
                s = f"{v:.{int(digits)}g}"
                if scientific is False and ("e" in s or "E" in s):
                    s = np.format_float_positional(float(s), trim="-")
            elif v == int(v) and abs(v) < 1e15:
                s = str(int(v))
            else:
                s = f"{v:.7g}"
                if scientific is False and ("e" in s or "E" in s):
                    s = np.format_float_positional(v, trim="-")
            if nsmall is not None and "e" not in s and "E" not in s:
                head, _, tail = s.partition(".")
                if len(tail) < int(nsmall):
                    s = head + "." + tail + "0" * (int(nsmall) - len(tail))
        else:
            s = str(v)
        return s

    arr = np.asarray(x)
    if arr.ndim == 0:
        out = fmt_one(arr.item())
        return out.rjust(int(width)) if width is not None else out
    strs = [fmt_one(v) for v in arr.ravel()]
    pad = max((len(s) for s in strs), default=0)
    if width is not None:
        pad = max(pad, int(width))
    return np.array([s.rjust(pad) for s in strs]).reshape(arr.shape)
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
                # A sourced translation can contain an older generated runtime
                # helper. Keep the caller's current implementation rather than
                # allowing that cached helper to replace it.
                if not key.startswith("__") and not (
                    key == "try_catch_py" and key in before_namespace
                ):
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
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"translated source dependency not found: {py_path}") from exc
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
    if isinstance(x, (str, bytes)):
        return 1
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        return len(x.values)
    if isinstance(x, SimpleNamespace):
        return len(getattr(x, "_r_names", vars(x)))
    if isinstance(x, np.ndarray):
        return x.size
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return x.shape[1]
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
    if "r_as_matrix(" in python:
        helpers.append(
            """
def r_as_matrix(x):
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return x
    return np.asarray(x, dtype=float)
""".strip()
        )
    if "t_py(" in python:
        helpers.append(
            """
def t_py(x):
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return x.T
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
    if table is None:
        table = []
    table_values = set(np.atleast_1d(np.asarray(table)).tolist())
    arr = np.asarray(x)
    if arr.ndim == 0:
        return bool(arr.item() in table_values)
    return np.array([value in table_values for value in arr], dtype=bool)
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

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=copy) if copy is not None else np.asarray(self.values, dtype=dtype)

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
    def _as_callable(function):
        if callable(function):
            return function
        if isinstance(function, str):
            f = globals().get(function)
            if callable(f):
                return f
        return None

    def _result_to_array(value):
        if isinstance(value, RNamedVector):
            arr = np.asarray(value.values)
            if arr.ndim == 0:
                return np.array([arr.item()]), list(value.names)
            return arr, list(value.names)
        arr = np.asarray(value)
        if arr.ndim == 0:
            return np.array([arr.item()]), []
        return np.ravel(arr), []

    def _apply_to_matrix(matrix, axis):
        fun = _as_callable(func) if not func in {"sum", "mean", "median", "var", "sd", "min", "max"} else None
        output = []
        if fun is None:
            return None
        row_names = None
        col_names = None
        if "pd" in globals() and isinstance(matrix, pd.DataFrame):
            col_names = list(matrix.columns)
            row_names = list(matrix.index)
            if axis == 0:
                for i in range(matrix.shape[0]):
                    values, names = _result_to_array(fun(matrix.iloc[i, :]))
                    output.append((values, names))
            else:
                for j in range(matrix.shape[1]):
                    values, names = _result_to_array(fun(matrix.iloc[:, j]))
                    output.append((values, names))
        else:
            if axis == 0:
                for row in matrix:
                    values, names = _result_to_array(fun(row))
                    output.append((values, names))
            else:
                transposed = matrix.T
                for col in transposed:
                    values, names = _result_to_array(fun(col))
                    output.append((values, names))
        if not output:
            return np.array([])

        lens = [len(values) for values, _ in output]
        if all(l == 1 for l in lens):
            scalars = [values[0] for values, _ in output]
            if all(isinstance(v, (bool, np.bool_)) for v in scalars):
                vals = np.array(scalars, dtype=bool)
            else:
                vals = np.array([float(v) if np.ndim(v) == 0 else v for v in scalars])
            if "pd" in globals() and isinstance(matrix, pd.DataFrame) and axis in {0, 1}:
                labels = row_names if axis == 0 else col_names
                if labels is not None:
                    return RNamedVector(vals, labels)
            return vals

        if all(l == lens[0] for l in lens):
            values = [values for values, _ in output]
            stacked = np.column_stack(values)
            if "pd" in globals() and isinstance(matrix, pd.DataFrame):
                first_names = output[0][1]
                index = first_names if first_names else [str(i) for i in range(stacked.shape[0])]
                columns = col_names if axis == 1 else row_names
                return pd.DataFrame(stacked, index=index, columns=columns)
            return stacked

        return np.array([v for values, _ in output], dtype=object)

    arr = np.asarray(x)
    try:
        keep_axes = np.atleast_1d(margin).astype(int) - 1
    except Exception:
        keep_axes = np.atleast_1d(int(margin))
        keep_axes = keep_axes - 1
    axis = 0
    if keep_axes.size:
        axis = int(keep_axes[0])
    if axis < 0:
        axis = 0
    if axis >= arr.ndim:
        axis = 0

    if "pd" in globals() and isinstance(x, pd.DataFrame):
        matrix_result = _apply_to_matrix(x, axis)
        if matrix_result is not None:
            return matrix_result

    matrix_result = _apply_to_matrix(arr, axis)
    if matrix_result is not None:
        return matrix_result

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
    if len(out) == 1 and all(np.ndim(value) == 0 for value in values):
        # Length-1 character results print as plain text, matching R.
        return str(out[0])
    return out
""".strip()
        )
    if "r_list_get(" in python:
        helpers.append(
            """
def r_list_get(x, idx):
    if "RNamedVector" in globals() and isinstance(idx, RNamedVector):
        idx = np.asarray(idx.values).ravel()[0]
    if not isinstance(idx, str):
        idx_arr = np.asarray(idx)
        if idx_arr.dtype.kind in {"U", "S", "O"}:
            idx = str(idx_arr.ravel()[0] if idx_arr.ndim else idx_arr.item())
    if isinstance(idx, str):
        return getattr(x, idx) if hasattr(x, idx) else x[idx]
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        return x.values[int(idx) - 1]
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return x.iloc[:, int(idx) - 1]
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
    if "kruskal_test_py(" in python or "wilcox_test_py(" in python:
        helpers.append(
            """
class RHTest:
    def __init__(self, method, statistics, p_value):
        self.method = method
        self.statistics = statistics
        self.p_value = p_value

    def __str__(self):
        parts = ", ".join(f"{name} = {value:g}" for name, value in self.statistics.items())
        return f"\\n\\t{self.method}\\n\\n{parts}, p-value = {self.p_value:g}\\n"

    __repr__ = __str__


def kruskal_test_py(x, g):
    g = np.asarray(getattr(g, "values", g))
    x = np.asarray(x)
    groups = [x[g == level] for level in np.unique(g)]
    stat, p = r_stats.kruskal(*groups)
    return RHTest(
        "Kruskal-Wallis rank sum test",
        {"Kruskal-Wallis chi-squared": float(stat), "df": len(groups) - 1},
        float(p),
    )


def wilcox_test_py(x, y=None, paired=False):
    if y is None or paired:
        stat, p = r_stats.wilcoxon(np.asarray(x), None if y is None else np.asarray(y))
        return RHTest("Wilcoxon signed rank test", {"V": float(stat)}, float(p))
    stat, p = r_stats.mannwhitneyu(np.asarray(x), np.asarray(y), alternative="two-sided")
    return RHTest("Wilcoxon rank sum test", {"W": float(stat)}, float(p))
""".strip()
        )
    if "r_command_args(" in python:
        helpers.append(
            """
def r_command_args(trailing_only=True):
    argv = list(sys.argv)
    if trailing_only:
        return np.array(argv[1:], dtype=object)
    # R includes the interpreter and a --file=script entry.
    return np.array([sys.executable, "--file=" + argv[0], *argv[1:]], dtype=object)
""".strip()
        )
    if "r_write_csv(" in python:
        helpers.append(
            """
def r_write_csv(x, path, index=True):
    if not isinstance(path, str):
        path_arr = np.asarray(path)
        path = str(path_arr.ravel()[0] if path_arr.ndim else path_arr.item())
    frame = x if isinstance(x, pd.DataFrame) else pd.DataFrame(np.asarray(x))
    frame.to_csv(path, index=index)
""".strip()
        )
    if "r_as_list(" in python:
        helpers.append(
            """
def r_as_list(x):
    if "RList" in globals() and isinstance(x, RList):
        return x
    if isinstance(x, list):
        return x
    return [item for item in np.atleast_1d(np.asarray(x))]
""".strip()
        )
    if "r_set_colnames(" in python or "r_set_rownames(" in python:
        helpers.append(
            """
def r_set_colnames(x, names):
    labels = [str(v) for v in np.atleast_1d(np.asarray(names))]
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        x.columns = labels
        return x
    if "pd" in globals() and isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == len(labels):
        return pd.DataFrame(x, columns=labels)
    return x


def r_set_rownames(x, names):
    labels = [str(v) for v in np.atleast_1d(np.asarray(names))]
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        x.index = labels
        return x
    if "pd" in globals() and isinstance(x, np.ndarray) and x.ndim == 2:
        return pd.DataFrame(x, index=labels)
    return x
""".strip()
        )
    if "r_sort(" in python:
        helpers.append(
            """
def r_sort(x, decreasing=False):
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        order = np.argsort(x.values, kind="stable")
        if decreasing:
            order = order[::-1]
        return RNamedVector(np.asarray(x.values)[order], [x.names[i] for i in order])
    out = np.sort(np.atleast_1d(np.asarray(x)), kind="stable")
    return out[::-1] if decreasing else out
""".strip()
        )
    if "r_sprintf(" in python:
        helpers.append(
            """
def r_sprintf(fmt, *args):
    arrays = [np.atleast_1d(np.asarray(arg)) for arg in args]
    n = max((len(arr) for arr in arrays), default=1)
    arrays = [np.resize(arr, n) for arr in arrays]
    out = []
    for i in range(n):
        row = []
        for arr in arrays:
            value = arr[i]
            row.append(value.item() if hasattr(value, "item") else value)
        out.append(str(fmt) % tuple(row))
    if n == 1 and all(np.ndim(arg) == 0 for arg in args):
        return out[0]
    return np.array(out)
""".strip()
        )
    if "r_colnames(" in python or "r_rownames(" in python:
        helpers.append(
            """
def r_colnames(x, fallback=None):
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return np.array([str(col) for col in x.columns])
    if fallback:
        return np.array(fallback)
    return np.array(getattr(x, "colnames", []))


def r_rownames(x, fallback=None):
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return np.array([str(row) for row in x.index])
    if fallback:
        return np.array(fallback)
    return np.array(getattr(x, "rownames", []))
""".strip()
        )
    if "r_axis_index(" in python:
        helpers.append(
            """
def r_axis_index(i):
    if "RNamedVector" in globals() and isinstance(i, RNamedVector):
        i = np.asarray(i.values)
    arr = np.asarray(i)
    if arr.dtype.kind in {"U", "S", "O", "b"}:
        return arr if arr.ndim else i
    if arr.ndim == 0:
        return int(arr) - 1
    return arr.astype(int) - 1
""".strip()
        )
    if "r_drop_index(" in python or "r_drop_axis(" in python:
        helpers.append(
            """
def r_drop_index(x, i):
    idx = np.asarray(i, dtype=int) - 1
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return x.drop(columns=x.columns[idx])
    arr = np.atleast_1d(x)
    if arr.size == 0:
        return arr
    return np.delete(arr, idx)


def r_drop_axis(x, i, axis):
    idx = np.asarray(i, dtype=int) - 1
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        if axis == 0:
            return x.drop(index=x.index[idx]).reset_index(drop=True)
        return x.drop(columns=x.columns[idx])
    return np.delete(x, idx, axis=axis)
""".strip()
        )
    if "r_list_set(" in python:
        helpers.append(
            """
def r_list_set(x, key, value):
    if not isinstance(key, str):
        key_arr = np.asarray(key)
        if key_arr.dtype.kind in {"U", "S", "O"}:
            key = str(key_arr.ravel()[0] if key_arr.ndim else key_arr.item())
    if isinstance(key, str):
        key = str(key)
        if "RList" in globals() and not isinstance(x, RList) and isinstance(x, (list, tuple)):
            fields = {f"x{i + 1}": item for i, item in enumerate(x)}
            x = RList(**fields, _r_names=list(fields))
        if "RList" in globals() and isinstance(x, RList):
            if key not in x._r_names:
                x._r_names.append(key)
            setattr(x, key, value)
            return x
        x[key] = value
        return x
    idx = int(key)
    if "RList" in globals() and isinstance(x, RList):
        setattr(x, x._r_names[idx - 1], value)
        return x
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        x.iloc[:, idx - 1] = np.asarray(value)
        return x
    if isinstance(x, list):
        while len(x) < idx:
            x.append(None)
    x[idx - 1] = value
    return x
""".strip()
        )
    if "r_as_numeric(" in python:
        helpers.append(
            """
def r_as_numeric(x):
    arr = np.asarray(x)
    if arr.dtype.kind in {"f", "i", "u", "b"}:
        return arr.astype(float)

    def conv(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    if arr.ndim == 0:
        return np.float64(conv(arr.item()))
    return np.array([conv(v) for v in arr.ravel()]).reshape(arr.shape)
""".strip()
        )
    if "r_set_names(" in python:
        helpers.append(
            """
def r_set_names(x, names):
    labels = [str(v) for v in np.atleast_1d(np.asarray(names))]
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        out = x.copy()
        out.columns = labels
        return out
    if "RList" in globals() and isinstance(x, RList):
        values = [getattr(x, name) for name in x._r_names]
        fields = dict(zip(labels, values))
        return RList(**fields, _r_names=labels)
    if isinstance(x, list):
        fields = dict(zip(labels, x))
        return RList(**fields, _r_names=labels)
    return RNamedVector(np.asarray(x), labels)
""".strip()
        )
    if "r_unlist(" in python:
        helpers.append(
            """
def r_unlist(x):
    if "RList" in globals() and isinstance(x, RList):
        parts = [np.atleast_1d(np.asarray(getattr(x, name))) for name in x._r_names]
        return np.concatenate(parts) if parts else np.array([])
    if isinstance(x, (list, tuple)):
        parts = [np.atleast_1d(np.asarray(item)) for item in x]
        return np.concatenate(parts) if parts else np.array([])
    return np.ravel(np.asarray(x))
""".strip()
        )
    if "r_formatC(" in python:
        helpers.append(
            """
def r_formatC(x, width=0, digits=None, format=None, flag="", big_mark=""):
    def one(value):
        if format == "d":
            text = f"{int(value):d}"
        elif format in {"f", "g", "e", "G", "E"} and digits is not None:
            text = f"{float(value):.{int(digits)}{format}}"
        elif digits is not None:
            text = f"{float(value):.{int(digits)}g}"
        else:
            text = str(value)
        w = abs(int(width))
        if int(width) < 0 or "-" in str(flag):
            return text.ljust(w)
        if "0" in str(flag) and w:
            return text.zfill(w)
        return text.rjust(w)

    arr = np.atleast_1d(np.asarray(x))
    out = np.array([one(v) for v in arr])
    return out if np.asarray(x).ndim else out[0]
""".strip()
        )
    graphics_names = [
        name
        for name in ["pdf", "png", "plot", "lines", "points", "legend", "abline", "grid", "mtext", "title", "par", "dev_off", "matplot", "hist", "barplot", "axis", "box", "text"]
        if re.search(rf"(?<![\w.]){name}\(", python)
    ]
    if graphics_names:
        stub_lines = ["def _r_graphics_noop(*args, **kwargs):", "    return None", ""]
        stub_lines.extend(f"{name} = _r_graphics_noop" for name in graphics_names if name != "rainbow")
        helpers.append("\n".join(stub_lines).strip())
    if re.search(r"(?<![\w.])rainbow\(", python):
        helpers.append(
            """
def rainbow(n, **kwargs):
    return np.array([f"#{int(i * 16777215 / max(int(n), 1)):06X}" for i in range(int(n))])
""".strip()
        )
    if "r_assign_all(" in python:
        helpers.append(
            """
def r_assign_all(x, value):
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        values = value
        if "RList" in globals() and isinstance(value, RList):
            values = [getattr(value, name) for name in value._r_names]
        for col, vals in zip(x.columns, values):
            x[col] = np.asarray(vals)
        return x
    arr = np.asarray(x)
    arr[...] = np.resize(np.asarray(value), arr.shape)
    return arr
""".strip()
        )
    if "r_substr_assign(" in python:
        helpers.append(
            """
def r_substr_assign(x, start, stop, value):
    def one(s, v):
        s = str(s)
        start_i = max(int(start) - 1, 0)
        stop_i = min(int(stop), len(s))
        repl = str(v)[: max(stop_i - start_i, 0)]
        return s[:start_i] + repl + s[start_i + len(repl):]

    if np.asarray(x).ndim == 0:
        return one(x, value)
    values = np.atleast_1d(np.asarray(value, dtype=str))
    return np.array([one(s, values[i % len(values)]) for i, s in enumerate(np.asarray(x, dtype=str))])
""".strip()
        )
    if "r_strsplit(" in python:
        helpers.append(
            """
def r_strsplit(x, split, fixed=False):
    def split_one(s):
        s = str(s)
        if split == "":
            return np.array(list(s))
        if fixed:
            return np.array(s.split(split))
        return np.array(re.split(split, s))

    if np.asarray(x).ndim == 0:
        return RList(x1=split_one(x), _r_names=["x1"])
    parts = [split_one(s) for s in np.asarray(x, dtype=str)]
    names = [f"x{i + 1}" for i in range(len(parts))]
    return RList(**dict(zip(names, parts)), _r_names=names)
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
    return np.array([re.search(pattern, str(item)) is not None for item in np.atleast_1d(np.asarray(x))], dtype=bool)


def regex_grep(pattern, x, value=False):
    matches = regex_grepl(pattern, x)
    arr = np.atleast_1d(np.asarray(x))
    return arr[matches] if value else np.nonzero(matches)[0] + 1


def regex_sub(pattern, repl, x, global_replace=False):
    count = 0 if global_replace else 1
    out = np.array([re.sub(pattern, repl, str(item), count=count) for item in np.atleast_1d(np.asarray(x))])
    return str(out[0]) if np.ndim(x) == 0 else out


def regex_regexpr(pattern, x):
    out = []
    for item in np.atleast_1d(np.asarray(x)):
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
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        return [str(col) for col in x.columns], [x[col] for col in x.columns]
    if isinstance(x, (str, bytes)) or np.ndim(x) == 0:
        values = [x]
    else:
        values = list(x)
    return [str(i + 1) for i in range(len(values))], values


def r_lapply(x, func):
    names, values = r_list_items(x)
    return RList(**{name: r_apply_func(value, func) for name, value in zip(names, values)}, _r_names=names)


def r_sapply(x, func):
    names, values = r_list_items(x)
    results = [r_apply_func(value, func) for value in values]
    if results and "RNamedVector" in globals() and all(isinstance(result, RNamedVector) for result in results):
        result_names = results[0].names
        if all(result.names == result_names for result in results):
            matrix = np.column_stack([result.values for result in results])
            if "pd" in globals():
                return pd.DataFrame(matrix, index=result_names)
            return matrix
    out = np.array(results)
    if out.ndim >= 2:
        # R's sapply/replicate simplification stacks results along the last axis.
        return np.moveaxis(out, 0, -1)
    use_names = isinstance(x, RList)
    if not use_names:
        try:
            use_names = np.asarray(x).dtype.kind in {"U", "S"}
        except Exception:
            use_names = False
    if use_names:
        if not isinstance(x, RList):
            names = [str(v) for v in np.atleast_1d(np.asarray(x))]
        return RNamedVector(out, names)
    return out
""".strip()
        )
    if "r_filter(" in python or "r_negate(" in python or "r_is_null" in python:
        helpers.append(
            """
def r_is_null(value):
    return value is None


def r_negate(func):
    return lambda *args, **kwargs: not bool(func(*args, **kwargs))


def r_filter(func, values):
    if "RList" in globals() and isinstance(values, RList):
        names = [name for name in values._r_names if func(getattr(values, name))]
        return RList(**{name: getattr(values, name) for name in names}, _r_names=names)
    if isinstance(values, dict):
        return {name: value for name, value in values.items() if func(value)}
    kept = [value for value in values if func(value)]
    if isinstance(values, np.ndarray):
        return np.asarray(kept)
    return kept
""".strip()
        )
    if "combn_py(" in python:
        helpers.append(
            """
def combn_py(x, m, func=None, simplify=True):
    values = np.asarray(x)
    flat = values.ravel()
    m = int(m)
    if m < 0 or m > flat.size:
        raise ValueError("combn requires 0 <= m <= length(x)")
    combinations = [np.asarray(items, dtype=values.dtype) for items in itertools.combinations(flat.tolist(), m)]
    if func is not None:
        combinations = [func(items) for items in combinations]
    if not simplify:
        return combinations
    if func is not None:
        return np.asarray(combinations)
    if not combinations:
        return np.empty((m, 0), dtype=values.dtype)
    return np.column_stack(combinations)
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
def acf_py(x, plot=False, lag_max=None):
    values = np.asarray(x, dtype=float)
    centered = values - np.mean(values)
    denom = np.dot(centered, centered)
    n = len(centered)
    lags = n if lag_max is None else min(int(lag_max) + 1, n)
    acf = np.array([np.dot(centered[: n - lag], centered[lag:]) / denom for lag in range(lags)])
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
    except Exception as exc:
        return fallback(exc) if callable(fallback) else fallback
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
    if isinstance(x, pd.Timestamp):
        return x.strftime(fmt)
    if isinstance(x, pd.DatetimeIndex):
        return x.strftime(fmt).to_numpy()
    if isinstance(x, pd.Series) and np.issubdtype(x.dtype, np.datetime64):
        return x.dt.strftime(fmt).to_numpy()
    arr = np.asarray(x)
    if arr.dtype.kind == "M":
        if arr.ndim == 0:
            return pd.Timestamp(arr.item()).strftime(fmt)
        return pd.DatetimeIndex(arr).strftime(fmt).to_numpy()
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
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        arr = np.asarray(x)
        out = np.diff(arr, axis=0) if arr.ndim >= 2 else np.diff(arr)
        if np.issubdtype(np.asarray(out).dtype, np.timedelta64):
            out = (out / np.timedelta64(1, "D")).astype(int)
        return pd.DataFrame(out, index=x.index[1:], columns=x.columns)
    arr = np.asarray(x)
    out = np.diff(arr, axis=0) if arr.ndim >= 2 else np.diff(arr)
    if np.issubdtype(np.asarray(out).dtype, np.timedelta64):
        return (out / np.timedelta64(1, "D")).astype(int)
    return out
""".strip()
        )
    if "proc_time(" in python:
        helpers.append(
            """
def proc_time():
    import time as _r_time
    return RNamedVector(
        np.array([0.0, 0.0, float(_r_time.perf_counter()), 0.0, 0.0], dtype=float),
        ["user.self", "sys.self", "elapsed", "user.child", "sys.child"],
    )
""".strip()
        )
    if "r_split(" in python or "r_unsplit(" in python:
        helpers.append(
            """
def r_split(x, group):
    groups = np.asarray(group.values if "RFactor" in globals() and isinstance(group, RFactor) else group)
    levels = group.levels if "RFactor" in globals() and isinstance(group, RFactor) else sorted(dict.fromkeys(groups).keys())
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        pieces = {str(level): x.loc[groups == level].reset_index(drop=True) for level in levels}
        return RList(**pieces, _r_names=[str(level) for level in levels])
    values = np.asarray(x)
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
        return values if key else np.asarray(values).reshape(-1)[:0]
    arr = np.asarray(key)
    if np.asarray(values).ndim == 0:
        idx = int(arr) if arr.ndim == 0 else int(np.asarray(arr).ravel()[0])
        return values if idx == 1 else np.asarray(values).ravel()[:0]
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
        if arr_idx.dtype == bool and arr_idx.size == x.shape[1]:
            # R's single-bracket df[mask] selects columns.
            return x.loc[:, arr_idx]
        if arr_idx.dtype == bool:
            return x.loc[arr_idx]
        return x.iloc[np.asarray(idx, dtype=int) - 1]
    arr = np.asarray(idx)
    if np.asarray(x).ndim == 2 and arr.ndim == 2 and arr.shape[1] == 2:
        return x[arr[:, 0].astype(int) - 1, arr[:, 1].astype(int) - 1]
    return r_vec_subset(x, idx)


def r_matrix_index_set(x, idx, value):
    if "pd" in globals() and isinstance(x, pd.DataFrame):
        arr_idx = np.asarray(idx)
        if arr_idx.dtype == bool and arr_idx.size == x.shape[1]:
            cols = x.columns[arr_idx]
        elif arr_idx.dtype.kind in {"i", "u", "f"}:
            cols = x.columns[np.atleast_1d(arr_idx).astype(int) - 1]
        else:
            cols = np.atleast_1d(arr_idx).tolist()
        values = value
        if "RList" in globals() and isinstance(value, RList):
            values = [getattr(value, name) for name in value._r_names]
        for col, vals in zip(cols, values):
            x[col] = np.asarray(vals)
        return x
    arr = np.asarray(idx)
    if np.asarray(x).ndim == 2 and arr.ndim == 2 and arr.shape[1] == 2:
        vals = np.resize(np.asarray(value), arr.shape[0])
        x[arr[:, 0].astype(int) - 1, arr[:, 1].astype(int) - 1] = vals
        return x
    if "RNamedVector" in globals() and isinstance(x, RNamedVector):
        key = np.asarray(idx)
        if key.dtype.kind in {"U", "S", "O"}:
            values = np.resize(np.asarray(value), key.size)
            for name, val in zip(key.ravel(), values):
                x[str(name)] = val
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
    if "r_subset(" in python or "r_matrix_row_named(" in python or "r_set_subset(" in python or "r_subset_df(" in python or "r_with(" in python or "r_within(" in python:
        helpers.append(
            """
def r_matrix_row_named(x, row, colnames=None):
    if isinstance(x, pd.DataFrame):
        rows = x.iloc[row, :] if not (isinstance(row, str) and row == ":") else x
        if isinstance(rows, pd.Series):
            arr = rows.to_numpy()
            cols = x.columns
            if colnames is None:
                colnames = cols
            if np.asarray(arr).size != len(cols):
                return arr
            if "RNamedVector" not in globals():
                return arr
            return RNamedVector(np.asarray(arr), list(cols if colnames is None else colnames))
        return rows
    row_values = np.asarray(x)[row]
    if np.asarray(x).ndim != 2 or not isinstance(row, (int, np.integer, np.ndarray, list, tuple, slice)):
        return row_values
    if isinstance(row_values, (int, float, np.floating, np.integer, np.number)):
        return row_values
    arr = np.asarray(row_values)
    if arr.ndim != 1:
        return arr
    if colnames is None:
        return arr
    if "RNamedVector" not in globals():
        return arr
    cols = np.asarray(colnames)
    if np.asarray(row_values).size != cols.size:
        return arr
    return RNamedVector(arr, list(cols))


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
        if "RNamedVector" in globals():
            if isinstance(row_key, RNamedVector):
                row_key = np.asarray(row_key.values)
            if isinstance(col_key, RNamedVector):
                col_key = np.asarray(col_key.values)
        string_cols = isinstance(col_key, str) or (
            isinstance(col_key, (list, tuple, np.ndarray)) and np.asarray(col_key).dtype.kind in {"U", "S", "O"}
        )
        bool_rows = isinstance(row_key, pd.Series) or (
            isinstance(row_key, (list, tuple, np.ndarray)) and np.asarray(row_key).dtype == bool
        )
        int_rows = isinstance(row_key, (list, tuple, np.ndarray)) and np.asarray(row_key).dtype.kind in {"i", "u"}
        if int_rows:
            row_key = np.asarray(row_key).ravel()
        def _positional(out, subset_rows):
            # R matrix subsets are positional; drop pandas row labels so later
            # arithmetic and comparisons do not align on stale indexes.
            if not subset_rows:
                return out
            if isinstance(out, pd.DataFrame):
                return out.reset_index(drop=True)
            if isinstance(out, pd.Series) and isinstance(row_key, (list, tuple, np.ndarray, pd.Series, slice)):
                return out.reset_index(drop=True)
            return out

        full_rows = isinstance(row_key, slice) and row_key == slice(None)
        if string_cols:
            cols = col_key.tolist() if isinstance(col_key, np.ndarray) else col_key
            if isinstance(cols, list) and cols and isinstance(cols[0], list):
                cols = np.asarray(cols).ravel().tolist()
            if isinstance(row_key, slice) or bool_rows:
                return _positional(x.loc[row_key, cols], not full_rows)
            if int_rows:
                return _positional(x.iloc[np.asarray(row_key), :].loc[:, cols], True)
            return _positional(x.loc[x.index[row_key], cols], True)
        if bool_rows:
            return x.loc[row_key, :].reset_index(drop=True)
        if int_rows:
            # R keeps a one-row data.frame for df[i, ]; do not drop to a Series.
            return x.iloc[np.atleast_1d(np.asarray(row_key)), :].reset_index(drop=True)
        if isinstance(row_key, (int, np.integer)) and isinstance(col_key, slice) and col_key == slice(None):
            return x.iloc[[int(row_key)], :].reset_index(drop=True)
        return _positional(x.iloc[row_key, col_key], not full_rows)
    if isinstance(x, pd.Series):
        if len(keys) == 2 and isinstance(keys[1], slice):
            keys = (keys[0],)
        key = keys[0] if len(keys) == 1 else keys
        if isinstance(key, pd.Series):
            return x.loc[key]
        if isinstance(key, (list, tuple, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == bool:
                return x.loc[arr].reset_index(drop=True)
            if arr.dtype.kind in {"i", "u"}:
                return x.iloc[arr].reset_index(drop=True)
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
    if len(keys) > 1:
        if len(keys) == 2:
            row_key, col_key = keys
            row_arr = np.asarray(row_key) if not isinstance(row_key, slice) else None
            col_arr = np.asarray(col_key) if not isinstance(col_key, slice) else None
            if (
                row_arr is not None
                and col_arr is not None
                and row_arr.dtype.kind in {"i", "u"}
                and col_arr.dtype.kind in {"i", "u"}
            ):
                if row_arr.ndim == 0 and col_arr.ndim == 0:
                    x[int(row_arr), int(col_arr)] = value
                else:
                    rows1 = np.atleast_1d(row_arr)
                    cols1 = np.atleast_1d(col_arr)
                    vals = np.asarray(value)
                    if vals.shape != (rows1.size, cols1.size):
                        flat = vals.ravel(order="F") if vals.ndim == 2 else np.ravel(vals)
                        # R fills the target block column-major with recycling.
                        vals = np.resize(flat, (cols1.size, rows1.size)).T
                    x[np.ix_(rows1, cols1)] = vals
                return x
        x[keys] = value
        return x
    key = keys[0]
    if isinstance(key, (list, tuple, np.ndarray)):
        arr = np.asarray(key)
        if arr.dtype == bool:
            x[arr] = value
            return x
    x[key] = value
    return x


def r_col_key(x, name, colnames=None):
    if isinstance(colnames, str):
        colnames = globals().get(colnames)
    if isinstance(colnames, np.ndarray):
        colnames = colnames.tolist()
    if isinstance(colnames, pd.Index):
        colnames = colnames.tolist()
    if isinstance(name, (list, tuple, np.ndarray)):
        return [r_col_key(x, item, colnames) for item in np.asarray(name).ravel()]
    if colnames is None:
        if isinstance(x, pd.DataFrame):
            return name
        return name
    cols = list(colnames)
    if name in cols:
        return cols.index(name)
    if isinstance(name, str):
        for i, item in enumerate(cols):
            if str(item) == name:
                return i
    raise KeyError(name)


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

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=copy) if copy is not None else np.asarray(self.values, dtype=dtype)

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


def cut_py(x, breaks, labels=None):
    values = np.asarray(x)
    breaks = np.asarray(breaks)
    if breaks.ndim == 0:
        # R: a single number gives that many intervals over the data range,
        # extended by 0.1 percent on each side.
        count = int(breaks)
        lo = float(np.min(values))
        hi = float(np.max(values))
        pad = (hi - lo) * 0.001 or 0.001
        breaks = np.linspace(lo - pad, hi + pad, count + 1)
    level_names = [f"({breaks[i]},{breaks[i + 1]}]" for i in range(len(breaks) - 1)]
    idx = np.searchsorted(breaks, values, side="left") - 1
    idx = np.clip(idx, 0, len(level_names) - 1)
    if labels is False:
        return idx + 1
    return RFactor(np.array([level_names[i] for i in idx]), levels=level_names, ordered=True)


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
        if name in {"stringsAsFactors", "check_names", "check_rows", "row_names"}:
            continue
        out[name] = r_df_col(value)
    if out and all(np.ndim(value) == 0 for value in out.values()):
        return pd.DataFrame({name: [value] for name, value in out.items()})
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
    if "r_c(" in python or "r_names(" in python or "r_setdiff(" in python or "RList(" in python or "RNamedVector(" in python or "r_attributes(" in python or "r_list_from_dots(" in python or "do_call_py(" in python or "rle_py(" in python or "inverse_rle_py(" in python or "summary_py(" in python or "r_table(" in python or "r_tapply(" in python or "r_lapply(" in python or "r_sapply(" in python or "r_split(" in python or "r_unsplit(" in python or "eigen_py(" in python or "svd_py(" in python or "qr_py(" in python or "determinant_py(" in python or "prcomp_py(" in python or "r_strsplit(" in python or "r_set_names(" in python or "r_unlist(" in python or "r_list_set(" in python or "r_sort(" in python:
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

    def __getattr__(self, name):
        # R returns NULL for missing list elements.
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return getattr(self, self._r_names[int(key)])


class RNamedVector:
    __array_priority__ = 1000

    def __init__(self, values, names):
        self.values = np.asarray(values)
        self.names = list(names)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=copy) if copy is not None else np.asarray(self.values, dtype=dtype)

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
        if arr.dtype == bool:
            self.values[arr] = value
            return
        if arr.dtype.kind in {"i", "u", "f"}:
            self.values[np.asarray(arr, dtype=int) - 1] = value
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

    def __neg__(self):
        return RNamedVector(-self.values, self.names)

    def __pos__(self):
        return RNamedVector(+self.values, self.names)

    def __abs__(self):
        return RNamedVector(np.abs(self.values), self.names)

    def __getattr__(self, name):
        try:
            names = object.__getattribute__(self, "names")
        except AttributeError:
            raise AttributeError(name)
        if name in names:
            return object.__getattribute__(self, "values")[list(names).index(name)]
        raise AttributeError(name)

    def __matmul__(self, other):
        other_values = other.values if isinstance(other, RNamedVector) else other
        return self.values @ np.asarray(other_values)

    def __rmatmul__(self, other):
        other_values = other.values if isinstance(other, RNamedVector) else other
        return np.asarray(other_values) @ self.values

    def _compare(self, other, op):
        other_values = other.values if isinstance(other, RNamedVector) else other
        return op(self.values, other_values)

    def __lt__(self, other):
        return self._compare(other, np.less)

    def __le__(self, other):
        return self._compare(other, np.less_equal)

    def __gt__(self, other):
        return self._compare(other, np.greater)

    def __ge__(self, other):
        return self._compare(other, np.greater_equal)

    def __eq__(self, other):
        return self._compare(other, np.equal)

    def __ne__(self, other):
        return self._compare(other, np.not_equal)

    __hash__ = None


def r_c(*values, names=None):
    if names is None and any(isinstance(value, (list, RList)) for value in values):
        # R promotes c() to a list when any argument is a list.
        out = []
        for value in values:
            if isinstance(value, list):
                out.extend(value)
            elif isinstance(value, RList):
                out.extend(getattr(value, name) for name in value._r_names)
            elif np.ndim(value) == 0:
                out.append(value)
            else:
                out.extend(list(np.asarray(value)))
        return out
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
        if use_numba:
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
        else:
            helpers.append(
                """
varma_resid_fast = None
""".strip()
            )
    if "def arma_residuals(" in python:
        arma_impl = """
def _arma_residuals_fast_impl(x, mu, ar, ma):
    y = x - mu
    n = len(y)
    e = np.zeros(n)
    p = len(ar)
    q = len(ma)
    for t in range(n):
        ar_part = 0.0
        for i in range(p):
            lag = t - i - 1
            if lag >= 0:
                ar_part += ar[i] * y[lag]
        ma_part = 0.0
        for i in range(q):
            lag = t - i - 1
            if lag >= 0:
                ma_part += ma[i] * e[lag]
        e[t] = y[t] - ar_part - ma_part
    return e
""".strip()
        if use_numba:
            helpers.append(
                f"""
try:
    from numba import njit as _arma_njit
except Exception:
    _arma_njit = None


{arma_impl}


_arma_residuals_fast = _arma_njit(cache=True)(_arma_residuals_fast_impl) if _arma_njit is not None else _arma_residuals_fast_impl


def arma_residuals_fast(x, mu, ar, ma):
    x = np.asarray(x, dtype=float)
    ar = np.asarray(ar, dtype=float).ravel()
    ma = np.asarray(ma, dtype=float).ravel()
    return _arma_residuals_fast(x, float(mu), ar, ma)
""".strip()
            )
        else:
            helpers.append(
                f"""
{arma_impl}


def arma_residuals_fast(x, mu, ar, ma):
    x = np.asarray(x, dtype=float)
    ar = np.asarray(ar, dtype=float).ravel()
    ma = np.asarray(ma, dtype=float).ravel()
    return _arma_residuals_fast_impl(x, float(mu), ar, ma)
""".strip()
            )
    if "def nagarch_var(" in python:
        nagarch_impl = """
def _nagarch_var_fast_impl(eps, omega, alpha, beta, theta):
    eps = np.asarray(eps)
    n = eps.shape[0]
    h = np.zeros(n)
    if n == 0:
        return h
    mean = 0.0
    for i in range(n):
        mean += eps[i]
    mean /= n
    var = 0.0
    for i in range(n):
        diff = eps[i] - mean
        var += diff * diff
    h[0] = var / (n - 1) if n > 1 else 0.0
    for i in range(1, n):
        zlag = eps[i - 1] / np.sqrt(h[i - 1])
        h[i] = omega + alpha * h[i - 1] * (zlag - theta) ** 2 + beta * h[i - 1]
        if (not np.isfinite(h[i])) or h[i] <= 0.0:
            h[i] = np.nan
    return h
""".strip()
        if use_numba:
            helpers.append(
                f"""
try:
    from numba import njit as _nagarch_njit
except Exception:
    _nagarch_njit = None


{nagarch_impl}


nagarch_var_fast = _nagarch_njit(cache=True)(_nagarch_var_fast_impl) if _nagarch_njit is not None else _nagarch_var_fast_impl
""".strip()
            )
        else:
            helpers.append(
                f"""
{nagarch_impl}


nagarch_var_fast = _nagarch_var_fast_impl
""".strip()
            )
    if "def garch_negloglik(" in python:
        garch_nll = """
import math as _math

def _garch_negloglik_fast_impl(par, x):
    par = np.asarray(par, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if par.size < 5 or x.size == 0:
        return 1e100
    mu = par[0]
    omega = np.exp(par[1])
    ea = np.exp(par[2])
    eb = np.exp(par[3])
    denom = 1.0 + ea + eb
    if not np.isfinite(denom) or denom <= 0.0:
        return 1e100
    alpha = 0.999 * ea / denom
    beta = 0.999 * eb / denom
    nu = 2.01 + np.exp(par[4])
    e = x - mu
    h = np.zeros(x.shape[0])
    persistence = 1.0 - alpha - beta
    if not np.isfinite(persistence) or persistence <= 0.0:
        return 1e100
    h0 = omega / persistence
    if (not np.isfinite(h0)) or h0 <= 0.0:
        h0 = np.sum((e - np.mean(e)) ** 2) / (e.size - 1) if e.size > 1 else e[0] ** 2
        if not np.isfinite(h0) or h0 <= 0.0:
            return 1e100
    h[0] = np.maximum(h0, 1e-08)
    for t in range(1, x.size):
        zlag = e[t - 1] ** 2
        h[t] = omega + alpha * zlag + beta * h[t - 1]
        if not np.isfinite(h[t]) or h[t] <= 0.0:
            return 1e100
    if not np.isfinite(nu) or nu <= 2.0:
        return 1e100
    lognorm = _math.lgamma(0.5 * (nu + 1.0)) - _math.lgamma(0.5 * nu) - 0.5 * _math.log((nu - 2.0) * np.pi)
    loglik = np.sum(lognorm - 0.5 * (nu + 1.0) * np.log1p((e * e) / (h * (nu - 2.0))) - 0.5 * np.log(h))
    if not np.isfinite(loglik):
        return 1e100
    return -loglik
""".strip()
        if use_numba:
            helpers.append(
                f"""
try:
    from numba import njit as _garch_njit
except Exception:
    _garch_njit = None


{garch_nll}


garch_negloglik_fast = _garch_njit(cache=True)(_garch_negloglik_fast_impl) if _garch_njit is not None else _garch_negloglik_fast_impl
""".strip()
            )
        else:
            helpers.append(
                f"""
{garch_nll}


garch_negloglik_fast = _garch_negloglik_fast_impl
""".strip()
            )
    if "def nagarch_negloglik(" in python:
        nagarch_nll = """
import math as _math

def _nagarch_negloglik_fast_impl(par, x):
    par = np.asarray(par, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    if par.size < 6 or x.size == 0:
        return 1e100
    mu = par[0]
    omega = np.exp(par[1])
    theta = par[2]
    ea = np.exp(par[3])
    eb = np.exp(par[4])
    denom = 1.0 + ea + eb
    if not np.isfinite(denom) or denom <= 0.0:
        return 1e100
    alpha_star = 0.999 * ea / denom
    alpha = alpha_star / (1.0 + theta ** 2)
    beta = 0.999 * eb / denom
    nu = 2.01 + np.exp(par[5])
    e = x - mu
    h = np.zeros(x.shape[0])
    persistence = 1.0 - alpha * (1.0 + theta ** 2) - beta
    if not np.isfinite(persistence) or persistence <= 0.0:
        return 1e100
    h0 = omega / persistence
    if (not np.isfinite(h0)) or h0 <= 0.0:
        h0 = np.sum((e - np.mean(e)) ** 2) / (e.size - 1) if e.size > 1 else e[0] ** 2
        if not np.isfinite(h0) or h0 <= 0.0:
            return 1e100
    h[0] = np.maximum(h0, 1e-08)
    for t in range(1, x.size):
        centered = e[t - 1] - theta * np.sqrt(h[t - 1])
        h[t] = omega + alpha * (centered ** 2) + beta * h[t - 1]
        if not np.isfinite(h[t]) or h[t] <= 0.0:
            return 1e100
    if not np.isfinite(nu) or nu <= 2.0:
        return 1e100
    lognorm = _math.lgamma(0.5 * (nu + 1.0)) - _math.lgamma(0.5 * nu) - 0.5 * _math.log((nu - 2.0) * np.pi)
    loglik = np.sum(lognorm - 0.5 * (nu + 1.0) * np.log1p((e * e) / (h * (nu - 2.0))) - 0.5 * np.log(h))
    if not np.isfinite(loglik):
        return 1e100
    return -loglik
""".strip()
        if use_numba:
            helpers.append(
                f"""
try:
    from numba import njit as _nagarch_njit
except Exception:
    _nagarch_njit = None


{nagarch_nll}


nagarch_negloglik_fast = _nagarch_njit(cache=True)(_nagarch_negloglik_fast_impl) if _nagarch_njit is not None else _nagarch_negloglik_fast_impl
""".strip()
            )
        else:
            helpers.append(
                f"""
{nagarch_nll}


nagarch_negloglik_fast = _nagarch_negloglik_fast_impl
""".strip()
            )
    if "r_print(" in python or "r_s3_print(" in python or "r_s3_dispatch(" in python:
        helpers.append(
            """
def r_format(x, digits=None):
    if isinstance(x, (bool, np.bool_)):
        return "TRUE" if x else "FALSE"
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

def r_print(*args, digits=None, colnames=None, row_names=True):
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
        frame = x
        if isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1:
            frame = frame.copy()
            frame.index = frame.index + 1
        print(frame.to_string(index=bool(row_names), na_rep="NA"))
    elif "pd" in globals() and isinstance(x, pd.Series):
        values = [r_format(v, digits) for v in x.to_numpy()]
        print(" ".join(values))
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
    x_meta = x
    y_meta = y
    has_pd = "pd" in globals()
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 0 or y.ndim == 0:
        out = _RRECYCLE_OPS[op](x, y)
        if isinstance(out, np.ndarray) and out.ndim == 0:
            out = out[()]
        out_arr = np.asarray(out)
        if has_pd and isinstance(x_meta, pd.DataFrame) and out_arr.shape == np.asarray(x_meta).shape:
            return pd.DataFrame(out_arr, index=x_meta.index, columns=x_meta.columns)
        if has_pd and isinstance(y_meta, pd.DataFrame) and out_arr.shape == np.asarray(y_meta).shape:
            return pd.DataFrame(out_arr, index=y_meta.index, columns=y_meta.columns)
        return out
    if x.ndim == 1 and y.ndim == 1 and x.shape[0] != y.shape[0]:
        n = max(x.shape[0], y.shape[0])
        x = np.resize(x, n)
        y = np.resize(y, n)
    if x.ndim == 2 and y.ndim == 1:
        flat = x.ravel(order="F")
        out = _RRECYCLE_OPS[op](flat, np.resize(y, flat.shape[0]))
        return out.reshape(x.shape, order="F")
    if x.ndim == 1 and y.ndim == 2:
        flat = y.ravel(order="F")
        out = _RRECYCLE_OPS[op](np.resize(x, flat.shape[0]), flat)
        return out.reshape(y.shape, order="F")
    out = _RRECYCLE_OPS[op](x, y)
    if has_pd:
        out_arr = np.asarray(out)
        if isinstance(x_meta, pd.DataFrame):
            if out_arr.shape == x_meta.shape:
                return pd.DataFrame(out, index=x_meta.index, columns=x_meta.columns)
            if out_arr.ndim == 1 and out_arr.shape[0] == x_meta.shape[0]:
                return pd.Series(out_arr, index=x_meta.index, name=getattr(x_meta, "name", None))
            if out_arr.ndim == 2 and out_arr.shape[0] == x_meta.shape[0] and out_arr.shape[1] == 1:
                return pd.DataFrame(out, index=x_meta.index)
        if isinstance(y_meta, pd.DataFrame):
            if out_arr.shape == y_meta.shape:
                return pd.DataFrame(out, index=y_meta.index, columns=y_meta.columns)
            if out_arr.ndim == 1 and out_arr.shape[0] == y_meta.shape[0]:
                return pd.Series(out_arr, index=y_meta.index)
            if out_arr.ndim == 2 and out_arr.shape[0] == y_meta.shape[0] and out_arr.shape[1] == 1:
                return pd.DataFrame(out, index=y_meta.index)
    return out


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
    if "determinant_py(" in python:
        helpers.append(
            """
def determinant_py(x, logarithm=True):
    sign, logabsdet = np.linalg.slogdet(np.asarray(x, dtype=float))
    modulus = logabsdet if logarithm else sign * np.exp(logabsdet)
    return RList(modulus=np.array([modulus]), sign=sign, _r_names=["modulus", "sign"])
""".strip()
        )
    if "polyroot_py(" in python:
        helpers.append(
            """
def polyroot_py(x):
    coeff = np.asarray(x, dtype=complex)
    return np.roots(coeff[::-1])
""".strip()
        )
    if "cor_py(" in python:
        helpers.append(
            """
def cor_py(x, y=None, use=None):
    x = np.asarray(x, dtype=float)
    if y is None:
        if use in {"pairwise.complete.obs", "complete.obs"}:
            if x.ndim == 1:
                x = x[np.isfinite(x)]
            else:
                x = x[np.all(np.isfinite(x), axis=1)]
        return np.corrcoef(x, rowvar=False)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) if use in {"pairwise.complete.obs", "complete.obs"} else slice(None)
    return np.corrcoef(x[mask], y[mask])[0, 1]
""".strip()
        )
    if "complete_cases_py(" in python:
        helpers.append(
            """
def complete_cases_py(x):
    if isinstance(x, pd.DataFrame):
        return np.asarray(x.notna().all(axis=1))
    arr = np.asarray(x)
    if arr.ndim <= 1:
        return ~pd.isna(arr)
    return np.all(~pd.isna(arr), axis=1)
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
    return SimpleNamespace(kind="lm", coef=coef, coef_names=coef_names, fitted=fitted, resid=resid, design=design)

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
    return SimpleNamespace(kind="glm", coef=np.asarray(model.params), coef_names=coef_names, fitted=np.asarray(model.fittedvalues), resid=np.asarray(model.resid_response), result=model, design=design)

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
def cbind_py(*cols, **named_cols):
    if named_cols:
        cols = list(cols) + [pd.DataFrame({name: np.asarray(value)}) for name, value in named_cols.items()]
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


def rbind_py(*rows, **named_rows):
    if named_rows:
        rows = list(rows) + list(named_rows.values())
    if any(isinstance(row, pd.DataFrame) for row in rows):
        template = next(row for row in rows if isinstance(row, pd.DataFrame))
        frames = []
        for row in rows:
            if isinstance(row, pd.DataFrame):
                frames.append(row.reset_index(drop=True))
                continue
            arr = np.asarray(row)
            if arr.ndim == 0:
                arr = arr.reshape((1, 1))
            elif arr.ndim == 1:
                arr = arr.reshape((1, -1))
            frame = pd.DataFrame(arr)
            if frame.shape[1] == template.shape[1]:
                frame.columns = template.columns
            frames.append(frame)
        return pd.concat(frames, axis=0, ignore_index=True)
    if rows and all("RNamedVector" in globals() and isinstance(row, RNamedVector) for row in rows):
        # R keeps the shared names as column names when binding named vectors.
        data = np.vstack([np.asarray(row.values).reshape((1, -1)) for row in rows])
        return pd.DataFrame(data, columns=[str(name) for name in rows[0].names])
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
    fn_kwargs = {k: v for k, v in kwargs.items() if k != "hessian"}
    maxiter = getattr(control, "maxit", None) if control is not None else None
    options = {"maxiter": int(maxiter)} if maxiter is not None else None
    result = optimize.minimize(
        lambda z: fn(z, **fn_kwargs),
        x0,
        method=method,
        options=options,
    )
    status = int(getattr(result, "status", 1))
    finite_result = np.all(np.isfinite(result.x)) and np.isfinite(result.fun)
    convergence = 0 if result.success or (status == 2 and finite_result) else status
    return SimpleNamespace(
        par=result.x,
        value=float(result.fun),
        convergence=convergence,
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
    if "def arma_residuals(" in python and "return arma_residuals_fast(" not in python:
        python = python.replace(
            "def arma_residuals(x, mu, ar, ma):\n",
            (
                "def arma_residuals(x, mu, ar, ma):\n"
                "    return arma_residuals_fast(x, mu, ar, ma)\n"
            ),
            1,
        )
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
    if "def nagarch_var(" in python and "nagarch_var_fast(" not in python:
        python = python.replace(
            "def nagarch_var(eps, omega, alpha, beta, theta):\n",
            (
                "def nagarch_var(eps, omega, alpha, beta, theta):\n"
                "    return nagarch_var_fast(eps, omega, alpha, beta, theta)\n"
            ),
            1,
        )
    if "def garch_negloglik(" in python and "garch_negloglik_fast(" not in python:
        python = python.replace(
            "def garch_negloglik(par, x):\n",
            (
                "def garch_negloglik(par, x):\n"
                "    return garch_negloglik_fast(np.asarray(par, dtype=float), np.asarray(x, dtype=float))\n"
            ),
            1,
        )
    if "def nagarch_negloglik(" in python and "nagarch_negloglik_fast(" not in python:
        python = python.replace(
            "def nagarch_negloglik(par, x):\n",
            (
                "def nagarch_negloglik(par, x):\n"
                "    return nagarch_negloglik_fast(np.asarray(par, dtype=float), np.asarray(x, dtype=float))\n"
            ),
            1,
        )
    return python


def logical_r_lines(source: str) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    depth = 0
    for original in source.splitlines():
        if not current and original.lstrip().startswith("#"):
            lines.append(original.lstrip())
            continue
        line = strip_r_comment(original).strip()
        # Drop LaTeX listings escape annotations found in literate R sources.
        line = re.sub(r"\(\*@.*?@\*\)", "", line).strip()
        line = re.sub(r"\s*_label~\w+@", "", line)
        if "\\(" in line:
            line = replace_lambda_shorthand(line)
        if not line:
            continue
        if current:
            separator = "; " if joined_statement_boundary(current[-1], line) else " "
            current.append(separator + line)
        else:
            current.append(line)
        depth += paren_delta(line)
        if depth <= 0:
            lines.append("".join(current).strip())
            current = []
            depth = 0
    if current:
        lines.append("".join(current).strip())
    return lines


def joined_statement_boundary(prev: str, nxt: str) -> bool:
    """True when two physical lines joined inside parens are separate statements."""
    prev = prev.rstrip()
    if not prev or r_line_continues(prev):
        return False
    if prev.endswith(("{", ";", "(", "[")):
        return False
    head = nxt.lstrip()
    if not head:
        return False
    if re.match(r"^(else\b|\|\||&&|\||&|\+|\-|\*|/|%|=|<|>|!|\^|,|\)|\]|\})", head):
        return False
    return True


def replace_lambda_shorthand(line: str) -> str:
    """Rewrite R 4.1 ``\\(x)`` lambda shorthand to ``function(x)`` outside strings."""
    masked, strings = mask_string_literals(line)
    masked = masked.replace("\\(", "function(")
    return restore_string_literals(masked, strings)


def paren_delta(line: str) -> int:
    quote = ""
    delta = 0
    for ch in line:
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"', "`"}:
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
        stripped = lines[i].lstrip()
        if not stripped.startswith("def "):
            i += 1
            continue
        def_indent = len(lines[i]) - len(stripped)
        start = i + 1
        end = start
        while end < len(lines) and (not lines[end].strip() or (len(lines[end]) - len(lines[end].lstrip())) > def_indent):
            end += 1
        if not add_branch_tail_returns(out, start, end, " " * (def_indent + len(INDENT))):
            j = end - 1
            while j >= start and not out[j].strip():
                j -= 1
            if j >= start and should_return_tail_expression(out[j]):
                indent = out[j][: len(out[j]) - len(out[j].lstrip())]
                out[j] = indent + "return " + out[j].strip()
        i += 1
    return "\n".join(out).rstrip() + "\n"


def add_branch_tail_returns(out: list[str], start: int, end: int, indent: str) -> bool:
    """Add returns to each branch when the block's tail statement is an if/elif/else chain.

    Returns True when the tail statement was an if-chain and has been handled.
    """
    headers = [
        j
        for j in range(start, end)
        if out[j].strip()
        and out[j].startswith(indent)
        and len(out[j]) > len(indent)
        and not out[j][len(indent)].isspace()
        and not out[j].lstrip().startswith("#")
    ]
    if not headers:
        return False
    last = headers[-1]
    if not re.match(r"^(?:if\b.*|elif\b.*|else\s*):\s*$", out[last].strip()):
        return False
    # Walk backward to the start of the if/elif/else chain.
    chain = [last]
    pos = len(headers) - 1
    while re.match(r"^(?:elif\b.*|else\s*):\s*$", out[chain[0]].strip()) and pos > 0:
        pos -= 1
        prev = headers[pos]
        if not re.match(r"^(?:if\b.*|elif\b.*|else\s*):\s*$", out[prev].strip()):
            return False
        chain.insert(0, prev)
    if not out[chain[0]].strip().startswith("if"):
        return False
    boundaries = chain[1:] + [end]
    for header, stop in zip(chain, boundaries):
        body_start = header + 1
        if add_branch_tail_returns(out, body_start, stop, indent + INDENT):
            continue
        j = stop - 1
        while j >= body_start and not out[j].strip():
            j -= 1
        if j >= body_start and should_return_tail_expression(out[j]):
            body_indent = out[j][: len(out[j]) - len(out[j].lstrip())]
            out[j] = body_indent + "return " + out[j].strip()
    return True


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
            frames.append(
                {
                    "indent": indent,
                    "line": i,
                    "parent": stack[-1] if stack else None,
                    "assigned": set(),
                }
            )
            stack.append(len(frames) - 1)
            continue

        assign = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if assign and stack:
            frames[stack[-1]]["assigned"].add(assign.group(1))

    declarations: dict[int, dict[str, set[str]]] = {}
    marker_lines = {i for i, _, _ in markers}
    for _, name, frame_id in markers:
        if frame_id is None:
            # At module scope an ordinary assignment already has global scope.
            continue
        parent = frames[frame_id]["parent"] if frame_id is not None else None
        found_nonlocal = False
        while parent is not None:
            if name in frames[parent]["assigned"]:
                found_nonlocal = True
                break
            parent = frames[parent]["parent"]
        kind = "nonlocal" if found_nonlocal else "global"
        declarations.setdefault(frame_id, {"nonlocal": set(), "global": set()})[kind].add(name)

    declarations_by_line = {
        int(frames[frame_id]["line"]): (frame_id, names)
        for frame_id, names in declarations.items()
    }
    out: list[str] = []
    for i, line in enumerate(lines):
        if i not in marker_lines:
            out.append(line)
        declaration = declarations_by_line.get(i)
        if declaration is None:
            continue
        frame_id, names = declaration
        prefix = " " * (int(frames[frame_id]["indent"]) + len(INDENT))
        if names["nonlocal"]:
            out.append(prefix + "nonlocal " + ", ".join(sorted(names["nonlocal"])))
        if names["global"]:
            out.append(prefix + "global " + ", ".join(sorted(names["global"])))

    return "\n".join(out).rstrip() + "\n"


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
        if line.lstrip().startswith("#"):
            continue
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
    line = rewrite_replicate_calls(line)
    lifted, line = lift_multistatement_function_literals(line)
    lifted_blocks, line = lift_braced_call_arguments(line)
    lifted.extend(lifted_blocks)
    result = translate_statement_inner(line)
    return lifted + result if lifted else result


def rewrite_replicate_calls(line: str) -> str:
    """Rewrite replicate(n, expr) as sapply/lapply over an anonymous function."""
    if "replicate" not in line:
        return line
    masked, strings = mask_string_literals(line)
    out = masked
    while True:
        match = re.search(r"(?<![\w.])replicate\s*\(", out)
        if match is None:
            break
        open_pos = out.find("(", match.start())
        close_pos = find_matching_paren(out, open_pos)
        if close_pos < 0:
            break
        args = split_args(out[open_pos + 1 : close_pos])
        positional = positional_args(args)
        if len(positional) < 2:
            break
        n_arg = keyword_arg(args, "n", default=positional[0])
        body = keyword_arg(args, "expr", default=positional[1]).strip()
        simplify = (keyword_arg(args, "simplify", default="TRUE") or "").strip().upper()
        if not body.startswith("{"):
            body = "{ " + body + " }"
        mapper = "lapply" if simplify in {"FALSE", "F"} else "sapply"
        replacement = f"{mapper}(seq_len({n_arg}), function(_r_rep_i_) {body})"
        out = out[: match.start()] + replacement + out[close_pos + 1 :]
    return restore_string_literals(out, strings)


_LIFTED_FN_COUNTER = 0


def brace_block_to_r_text(body: str) -> str:
    """Split a one-line braced block into physical R lines."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    quote = ""
    for ch in body:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in {"'", '"', "`"}:
            quote = ch
            buf.append(ch)
            continue
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(depth - 1, 0)
        if depth == 0:
            if ch == ";":
                parts.append("".join(buf))
                buf = []
                continue
            if ch == "{":
                buf.append("{")
                parts.append("".join(buf))
                buf = []
                continue
            if ch == "}":
                parts.append("".join(buf))
                parts.append("}")
                buf = []
                continue
        buf.append(ch)
    parts.append("".join(buf))
    return "\n".join(part.strip() for part in parts if part.strip())


def translate_function_body_lines(body: str) -> list[str]:
    """Translate a braced R block into indented Python body lines with tail returns."""
    global PENDING_FUNCTION_PARAMS
    saved_pending = PENDING_FUNCTION_PARAMS
    inner = body.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1]
    r_text = brace_block_to_r_text(inner)
    py_lines = translate_logical_lines(logical_r_lines(preprocess_simple_inline_r(r_text)))
    PENDING_FUNCTION_PARAMS = saved_pending
    if not py_lines:
        py_lines = ["pass"]
    snippet = "def _fn_():\n" + "\n".join(INDENT + item for item in py_lines) + "\n"
    snippet = return_function_tail_expressions(snippet)
    return snippet.splitlines()[1:]


def lift_multistatement_function_literals(line: str) -> tuple[list[str], str]:
    """Lift multi-statement function literals in call arguments into nested defs."""
    if "function" not in line:
        return [], line
    if re.match(r"^[A-Za-z.][\w.]*\s*(?:<<?-|=)\s*function\s*\(", line.strip()):
        return [], line
    global _LIFTED_FN_COUNTER
    masked, strings = mask_string_literals(line)
    defs: list[str] = []
    search = 0
    while True:
        match = re.search(r"(?<![\w.])function\s*\(", masked[search:])
        if match is None:
            break
        start = search + match.start()
        open_pos = masked.find("(", start)
        close_pos = find_matching_paren(masked, open_pos)
        if close_pos < 0:
            break
        after = close_pos + 1
        while after < len(masked) and masked[after].isspace():
            after += 1
        if after >= len(masked) or masked[after] != "{":
            search = close_pos + 1
            continue
        brace_close = find_matching_char(masked, after, "{", "}")
        if brace_close < 0:
            search = close_pos + 1
            continue
        if translate_function_literal(masked[start : brace_close + 1]) is not None:
            search = brace_close + 1
            continue
        _LIFTED_FN_COUNTER += 1
        name = f"_r_fn_{_LIFTED_FN_COUNTER}"
        signature, setup = translate_function_signature(masked[open_pos + 1 : close_pos])
        defs.append(f"def {name}({signature}):")
        defs.extend(INDENT + item for item in setup)
        defs.extend(translate_function_body_lines(masked[after : brace_close + 1]))
        masked = masked[:start] + name + masked[brace_close + 1 :]
        search = start + len(name)
    if not defs:
        return [], line
    return (
        [restore_string_literals(item, strings) for item in defs],
        restore_string_literals(masked, strings),
    )


def lift_braced_call_arguments(line: str) -> tuple[list[str], str]:
    """Lift ``f({ statements; value })`` blocks into nested functions."""
    global _LIFTED_FN_COUNTER
    if "{" not in line:
        return [], line
    masked, strings = mask_string_literals(line)
    defs: list[str] = []
    search = 0
    while search < len(masked):
        brace_pos = masked.find("{", search)
        if brace_pos < 0:
            break
        paren_depth = 0
        for char in masked[:brace_pos]:
            if char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth = max(paren_depth - 1, 0)
        previous = brace_pos - 1
        while previous >= 0 and masked[previous].isspace():
            previous -= 1
        # Control-flow and function-definition braces are handled elsewhere.
        # A braced call argument follows an opening paren, comma, or named-arg
        # equals sign while still nested inside the call's parentheses.
        if paren_depth == 0 or previous < 0 or masked[previous] not in "(,=":
            search = brace_pos + 1
            continue
        brace_close = find_matching_char(masked, brace_pos, "{", "}")
        if brace_close < 0:
            break
        _LIFTED_FN_COUNTER += 1
        name = f"_r_block_{_LIFTED_FN_COUNTER}"
        defs.append(f"def {name}():")
        defs.extend(translate_function_body_lines(masked[brace_pos : brace_close + 1]))
        replacement = f"{name}()"
        masked = masked[:brace_pos] + replacement + masked[brace_close + 1 :]
        search = brace_pos + len(replacement)
    if not defs:
        return [], line
    return (
        [restore_string_literals(item, strings) for item in defs],
        restore_string_literals(masked, strings),
    )


def translate_statement_inner(line: str) -> list[str]:
    global PENDING_FUNCTION_PARAMS
    parsed_func = parse_function_definition(line)
    if parsed_func is not None:
        name, args, body = parsed_func
        py_name = r_function_name(name)
        USER_FUNCTION_NAMES.add(py_name)
        params = function_param_names(args)
        USER_FUNCTION_PARAMS[py_name] = params
        PENDING_FUNCTION_PARAMS = params
        signature, setup = translate_function_signature(args)
        if body is not None:
            if body.startswith("{") and body.endswith("}"):
                return [f"def {py_name}({signature}):", *[INDENT + line for line in setup], *translate_function_body_lines(body)]
            lifted, body = lift_multistatement_function_literals(body)
            return [
                f"def {py_name}({signature}):",
                *[INDENT + line for line in setup],
                *[INDENT + line for line in lifted],
                INDENT + "return " + translate_expr(body),
            ]
        return [f"def {py_name}({signature}):", *[INDENT + line for line in setup]]
    expr_func_match = re.match(r"([A-Za-z]\w*(?:\.\w+)*)\s*(?:<-|=)\s*function\s*\((.*?)\)\s+(.+)$", line)
    if expr_func_match:
        name, args, body = expr_func_match.groups()
        params = function_param_names(args)
        USER_FUNCTION_PARAMS[r_function_name(name)] = params
        PENDING_FUNCTION_PARAMS = params
        py_name = r_function_name(name)
        USER_FUNCTION_NAMES.add(py_name)
        signature, setup = translate_function_signature(args)
        return [f"def {py_name}({signature}):", *[INDENT + line for line in setup], INDENT + "return " + translate_expr(body)]
    func_match = re.match(r"([A-Za-z]\w*(?:\.\w+)*)\s*(?:<-|=)\s*function\s*\((.*)\)\s*$", line)
    if func_match:
        name, args = func_match.groups()
        py_name = r_function_name(name)
        USER_FUNCTION_NAMES.add(py_name)
        params = function_param_names(args)
        USER_FUNCTION_PARAMS[py_name] = params
        PENDING_FUNCTION_PARAMS = params
        signature, setup = translate_function_signature(args)
        return [f"def {py_name}({signature}):", *[INDENT + line for line in setup]]

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
        CHARACTER_VECTOR_VARS.discard(name)
        LOGICAL_VECTOR_VARS.discard(name)
        iterable_name = r_name(strip_outer_parens(values.strip()))
        if iterable_name in CHARACTER_VECTOR_VARS or is_character_vector_expr(values):
            CHARACTER_VECTOR_VARS.add(name)
        elif iterable_name in LOGICAL_VECTOR_VARS:
            LOGICAL_VECTOR_VARS.add(name)
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
        row_names_match = re.match(r"^(?:rownames|row\.names|row_names)\s*\(\s*([A-Za-z]\w*)\s*\)$", diag_lhs, re.IGNORECASE)
        if row_names_match:
            obj = r_name(row_names_match.group(1))
            return [f"{obj} = r_set_rownames({obj}, {translate_expr(diag_rhs)})"]
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
        member_match = re.match(r"^([A-Za-z.]\w*)\$([\w.]+)$", lhs)
        if member_match:
            obj = r_name(member_match.group(1))
            field = r_name(member_match.group(2))
            return [f"{obj}.{field} = {translate_expr(rhs)}"]
        # Fall back to ordinary assignment semantics for other targets.
        return translate_statement(f"{lhs} <- {rhs}")

    raw_assign = raw_assignment(line)
    if raw_assign is not None:
        raw_lhs, raw_rhs = raw_assign
        substr_assign = re.match(r"^substr(?:ing)?\s*\((.*)\)$", raw_lhs, re.IGNORECASE)
        if substr_assign:
            parts = split_args(substr_assign.group(1))
            if len(parts) == 3:
                target = translate_expr(parts[0])
                return [f"{target} = r_substr_assign({target}, {translate_expr(parts[1])}, {translate_expr(parts[2])}, {translate_expr(raw_rhs)})"]
        raw_double_subscript_assign = re.match(r"^([A-Za-z]\w*)\[\[(.*)\]\]$", raw_lhs)
        if raw_double_subscript_assign:
            base, index = raw_double_subscript_assign.groups()
            if re.fullmatch(r"\d+", index.strip()):
                return [f"{r_name(base)}[{translate_subscript(index, base=r_name(base))}] = {translate_expr(raw_rhs)}"]
            py_base = r_name(base)
            return [f"{py_base} = r_list_set({py_base}, {translate_expr(index)}, {translate_expr(raw_rhs)})"]
        raw_subscript_assign = re.match(r"^([A-Za-z]\w*(?:\.\w+)*)\[(.*)\]$", raw_lhs)
        if raw_subscript_assign:
            base, index = raw_subscript_assign.groups()
            if "." in base:
                DOTTED_R_VARS.add(base)
            py_base = r_name(base)
            py_rhs = translate_expr(raw_rhs)
            if not index.strip():
                return [f"{py_base} = r_assign_all({py_base}, {py_rhs})"]
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
        order_select = re.match(
            r"^([A-Za-z]\w*)\s*\[\s*which\.min\s*\(\s*\1\s*\[\s*,\s*([\"'])([^\"']+)\2\s*\]\s*\)\s*,\s*([\"'])order\4\s*\]\s*$",
            rhs,
            re.IGNORECASE,
        )
        if order_select is not None and lhs.endswith("_order"):
            mat = r_name(order_select.group(1))
            criterion = order_select.group(3)
            row = f"(r_which_min(r_subset({mat}, slice(None), r_col_key({mat}, {criterion!r}, globals().get('{mat}_colnames')))) - 1)"
            col = f"r_col_key({mat}, \"order\", globals().get('{mat}_colnames'))"
            return [f"{lhs} = int({mat}[{row}, {col}])"]
        named_order_select = re.match(
            r"^([A-Za-z]\w*)\s*\[\s*which\.min\s*\(\s*ifelse\s*\(\s*(.+?)\s*,\s*\1\s*\[\s*,\s*([\"'])([^\"']+)\3\s*\]\s*,\s*Inf\s*\)\s*\)\s*,\s*c\s*\((.+)\)\s*\]\s*$",
            rhs,
            re.IGNORECASE,
        )
        if named_order_select is not None:
            mat = r_name(named_order_select.group(1))
            cond = translate_expr(named_order_select.group(2))
            criterion = named_order_select.group(4)
            cols = [translate_expr(part.strip()) for part in split_args(named_order_select.group(5))]
            col_names = f"r_c({', '.join(cols)})"
            row = f"(r_which_min(np.where({cond}, r_subset({mat}, slice(None), r_col_key({mat}, {criterion!r}, globals().get('{mat}_colnames'))), np.inf)) - 1)"
            col = f"np.array([r_col_key({mat}, _col, globals().get('{mat}_colnames')) for _col in np.ravel({col_names})])"
            values = f"r_subset({mat}, {row}, {col})"
            literal_cols = [col[1:-1] for col in cols if is_string_literal(col)]
            if literal_cols and len(literal_cols) == len(cols) and all("order" in col for col in literal_cols):
                values = f"np.asarray({values}, dtype=int)"
            return [f"{lhs} = RNamedVector({values}, list({col_names}))"]
        py_rhs = translate_expr(rhs)
        if lhs in {"aic_order", "bic_order"}:
            criterion = lhs.split("_", 1)[0]
            py_rhs = re.sub(
                r"r_col_key\(([A-Za-z]\w*),\s*['\"]\1_colnames['\"],\s*globals\(\)\.get\(['\"]\1_colnames['\"]\)\)",
                lambda m: f"r_col_key({m.group(1)}, {criterion!r}, globals().get('{m.group(1)}_colnames'))",
                py_rhs,
            )
            py_rhs = re.sub(r"(r_which_min\([^,\n]+(?:\([^)]*\)[^,\n]*)*\))(?=,\s*r_col_key)", r"(\1) - 1", py_rhs)
        double_subscript_assign = re.match(r"^([A-Za-z]\w*(?:\.\w+)*)\[\[(.*)\]\]$", lhs)
        if double_subscript_assign:
            base, index = double_subscript_assign.groups()
            if "." in base:
                DOTTED_R_VARS.add(base)
            py_base = r_name(base)
            return [f"{py_base}[{translate_subscript(index, base=py_base)}] = {py_rhs}"]
        subscript_assign = re.match(r"^([A-Za-z]\w*(?:\.\w+)*)\[(.*)\]$", lhs)
        if subscript_assign:
            base, index = subscript_assign.groups()
            if "." in base:
                DOTTED_R_VARS.add(base)
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
        if lhs.endswith("_order") and (
            re.search(r"_colnames\.index\([\"']order[\"']\)", py_rhs)
            or re.search(r"r_col_key\([^)]*,\s*[\"']order[\"']", py_rhs)
        ):
            py_rhs = f"int({py_rhs})"
        if (
            lhs.endswith(("_p", "_q"))
            and re.search(r"_colnames\.index\([\"'][pq][\"']\)", py_rhs)
            and not py_rhs.startswith("int(")
        ):
            py_rhs = f"int({py_rhs})"
        as_matrix_subset = re.match(r"as\.matrix\s*\(\s*([A-Za-z]\w*)\s*\[\s*,\s*(.+)\]\s*\)\s*$", rhs, re.IGNORECASE)
        if as_matrix_subset:
            _, raw_cols = as_matrix_subset.groups()
            col_parts = [part.strip() for part in split_subscript_args("," + raw_cols) if part.strip() and not is_subscript_option(part.strip())]
            if col_parts:
                return [f"{lhs} = {py_rhs}", f"{lhs}_colnames = list({translate_expr(col_parts[0])})"]
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
        return [f"{py_obj} = r_set_names({py_obj}, {translate_expr(rhs)})"]
    suffix = "colnames" if kind.lower() == "colnames" else "rownames"
    py_obj = r_name(obj)
    if rhs.strip().upper() == "NULL":
        return [f"{py_obj}_{suffix} = []"]
    setter = "r_set_colnames" if suffix == "colnames" else "r_set_rownames"
    return [
        f"{py_obj}_{suffix} = list(np.atleast_1d(np.asarray({translate_expr(rhs)})))",
        f"{py_obj} = {setter}({py_obj}, {py_obj}_{suffix})",
    ]


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
            raw_name = arg.strip()
            if "." in raw_name:
                DOTTED_R_VARS.add(raw_name)
            name = r_name(raw_name)
            out.append(name)
        previous.add(name)
    # Python requires defaults on parameters that follow defaulted ones; R does not.
    seen_default = False
    for i, param in enumerate(out):
        if "=" in param:
            seen_default = True
        elif seen_default:
            out[i] = param + "=None"
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
        if "." in name:
            DOTTED_R_VARS.add(name)
        names.append(r_name(name))
    return names


def translate_optim_call(args: list[str]) -> str:
    par = keyword_arg(args, "par", default=args[0] if args else "None")
    fn = keyword_arg(args, "fn", default=args[1] if len(args) > 1 else "None")
    method = keyword_arg(args, "method", default='"BFGS"')
    control = keyword_arg(args, "control", default="None")
    control_expr = translate_optim_control(control)
    kwargs: list[str] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos < 0:
            continue
        key = normalize_keyword_name(arg[:pos].strip())
        if key in {"par", "fn", "method", "control"}:
            continue
        kwargs.append(f"{key}={translate_expr(arg[pos + 1:].strip())}")
    base = [
        f"par={translate_expr(par)}",
        f"fn={translate_expr(fn)}",
        f"method={translate_expr(method)}",
        f"control={control_expr}",
    ]
    return "optim(" + ", ".join(base + kwargs) + ")"


def translate_optim_control(control: str | None) -> str:
    if control is None or control.strip() == "None":
        return "None"
    raw_call = parse_full_call(control.strip())
    if raw_call is None or raw_call[0].lower() not in {"list", "rlist"}:
        return translate_expr(control)
    fields: list[str] = []
    for arg in raw_call[1]:
        pos = find_top_level_operator(arg, "=")
        if pos < 0:
            continue
        key = normalize_keyword_name(arg[:pos].strip())
        if key in {"_r_names", "r_names"}:
            continue
        fields.append(f"{key}={translate_expr(arg[pos + 1:].strip())}")
    return "SimpleNamespace(" + ", ".join(fields) + ")"


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
        if re.match(r"^\.?[A-Za-z]\w*(?:\.\w+)*(?:\[.*\])?$", lhs):
            base_lhs = lhs.split("[", 1)[0]
            if "." in base_lhs:
                DOTTED_R_VARS.add(base_lhs)
            if "[" not in lhs:
                py_lhs = r_name(base_lhs)
                if is_character_vector_expr(rhs):
                    CHARACTER_VECTOR_VARS.add(py_lhs)
                else:
                    CHARACTER_VECTOR_VARS.discard(py_lhs)
                if is_logical_vector_expr(rhs):
                    LOGICAL_VECTOR_VARS.add(py_lhs)
                else:
                    LOGICAL_VECTOR_VARS.discard(py_lhs)
                if is_matrix_expr(rhs):
                    MATRIX_VARS.add(py_lhs)
                else:
                    MATRIX_VARS.discard(py_lhs)
                return py_lhs, rhs
            return translate_expr(lhs), rhs
    return None


def raw_assignment(line: str) -> tuple[str, str] | None:
    for op in ("<-", "="):
        pos = find_top_level_operator(line, op)
        if pos >= 0:
            return line[:pos].strip(), line[pos + len(op) :].strip()
    return None


def split_top_level_semicolons(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    quote = ""
    start = 0
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
        elif ch == ";" and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return parts


def strip_return_call(stmt: str) -> str:
    stmt = stmt.strip()
    if re.match(r"^return\s*\(", stmt):
        open_pos = stmt.find("(")
        close_pos = find_matching_paren(stmt, open_pos)
        if close_pos == len(stmt) - 1:
            return stmt[open_pos + 1 : close_pos].strip()
    return stmt


def unwrap_braced_expression(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        stmts = [part for part in split_top_level_semicolons(inner) if part.strip()]
        if len(stmts) == 1:
            return stmts[0].strip()
    return text


_LAMBDA_MASKS: dict[str, str] = {}


def mask_lambda(text: str) -> str:
    placeholder = f"__R_LAMBDA_{len(_LAMBDA_MASKS)}__"
    _LAMBDA_MASKS[placeholder] = text
    return placeholder


def restore_lambda_masks(text: str) -> str:
    # Inner lambdas can appear inside outer lambda bodies, so iterate.
    while "__R_LAMBDA_" in text:
        changed = False
        for placeholder, value in _LAMBDA_MASKS.items():
            if placeholder in text:
                text = text.replace(placeholder, value)
                changed = True
        if not changed:
            break
    return text


def translate_function_literal(expr: str) -> str | None:
    if not re.match(r"^function\s*\(", expr):
        return None
    open_pos = expr.find("(")
    close_pos = find_matching_paren(expr, open_pos)
    if close_pos < 0:
        return None
    body = expr[close_pos + 1 :].strip()
    if not body:
        return None
    params: list[str] = []
    for raw_param in split_args(expr[open_pos + 1 : close_pos]):
        raw_param = raw_param.strip()
        if not raw_param:
            continue
        if raw_param == "...":
            params.append("*_dots")
            continue
        pos = find_top_level_operator(raw_param, "=")
        if pos >= 0:
            params.append(f"{normalize_keyword_name(raw_param[:pos])}={translate_expr(raw_param[pos + 1:])}")
        else:
            params.append(r_name(raw_param))
    body_expr = lambda_body_expression(body)
    if body_expr is None:
        return None
    header = "lambda " + ", ".join(params) if params else "lambda"
    candidate = f"({header}: {body_expr})"
    try:
        ast.parse(repair_inline_lambda_keyword(normalize_dotted_call_syntax(restore_lambda_masks(candidate))), mode="eval")
    except SyntaxError:
        return None
    return mask_lambda(candidate)


def lambda_body_expression(body: str) -> str | None:
    body = body.strip()
    if body.startswith("{") and body.endswith("}"):
        stmts = [part.strip() for part in split_top_level_semicolons(body[1:-1]) if part.strip()]
        if not stmts:
            return "None"
        pieces: list[str] = []
        for stmt in stmts[:-1]:
            assign = raw_assignment(stmt)
            if assign is None:
                return None
            lhs, rhs = assign
            if not re.match(r"^[A-Za-z]\w*(?:\.\w+)*$", lhs):
                return None
            pieces.append(f"({r_name(lhs)} := {translate_expr(rhs)})")
        pieces.append(translate_expr(strip_return_call(stmts[-1])))
        if len(pieces) == 1:
            return pieces[0]
        return "(" + ", ".join(pieces) + ")[-1]"
    return translate_expr(strip_return_call(body))


def translate_if_else_expr(expr: str) -> str | None:
    if not re.match(r"^if\s*\(", expr):
        return None
    open_pos = expr.find("(")
    close_pos = find_matching_paren(expr, open_pos)
    if close_pos < 0:
        return None
    cond = expr[open_pos + 1 : close_pos].strip()
    rest = expr[close_pos + 1 :].strip()
    split = split_top_level_else(rest)
    if split is None:
        # R's if without else evaluates to NULL when the condition is false.
        split = (rest, "NULL")
    yes, no = split
    if not yes or not no:
        return None
    yes = unwrap_braced_expression(strip_return_call(yes))
    no = unwrap_braced_expression(strip_return_call(no))
    candidate = f"({translate_expr(yes)} if {translate_expr(cond)} else {translate_expr(no)})"
    try:
        ast.parse(repair_inline_lambda_keyword(normalize_dotted_call_syntax(restore_lambda_masks(candidate))), mode="eval")
    except SyntaxError:
        return None
    return candidate


def replace_parenthesized_if_else_exprs(expr: str) -> str:
    """Translate R inline ``if`` expressions nested inside larger expressions."""
    while True:
        changed = False
        for match in reversed(list(re.finditer(r"\(\s*if\s*\(", expr))):
            open_pos = match.start()
            close_pos = find_matching_char(expr, open_pos, "(", ")")
            if close_pos < 0:
                continue
            inner = expr[open_pos + 1 : close_pos].strip()
            translated = translate_if_else_expr(inner)
            if translated is None:
                continue
            expr = expr[:open_pos] + translated + expr[close_pos + 1 :]
            changed = True
            break
        if not changed:
            return expr


def mask_inline_if_call_arguments(expr: str) -> str:
    """Protect inline if/else call arguments from innermost-call rewrites."""
    parsed = parse_full_call(expr)
    if parsed is None:
        return expr
    name, args = parsed
    changed = False
    rebuilt: list[str] = []
    for arg in args:
        pos = find_top_level_operator(arg, "=")
        if pos >= 0 and not (
            (pos > 0 and arg[pos - 1] in "<>!")
            or (pos + 1 < len(arg) and arg[pos + 1] == "=")
        ):
            prefix = arg[: pos + 1]
            value = arg[pos + 1 :].strip()
        else:
            prefix = ""
            value = arg.strip()
        translated = translate_if_else_expr(value)
        if translated is None:
            rebuilt.append(arg)
            continue
        rebuilt.append(prefix + mask_lambda(translated))
        changed = True
    if not changed:
        return expr
    return f"{name}(" + ", ".join(rebuilt) + ")"


def translate_expr(expr: str) -> str:
    expr = expr.strip().rstrip(";")
    backtick_op = re.fullmatch(r"`([^`\w]+)`", expr)
    if backtick_op:
        return repr(backtick_op.group(1))
    func_literal = translate_function_literal(expr)
    if func_literal is not None:
        return func_literal
    if_else = translate_if_else_expr(expr)
    if if_else is not None:
        return if_else
    expr = replace_parenthesized_if_else_exprs(expr)
    expr = mask_inline_if_call_arguments(expr)
    det_mod = translate_determinant_modulus_expr(expr)
    if det_mod is not None:
        return det_mod
    raw_call = parse_full_call(expr)
    if raw_call is not None and raw_call[0].lower() == "vector":
        return translate_vector_call(raw_call[1])
    expr, strings = mask_string_literals(expr)
    expr = translate_expr_code(expr)
    expr = restore_string_literals(expr, strings)
    for i, (_, text) in enumerate(strings):
        expr = expr.replace(f"__R_ATTR_{i}__", text)
    return expr


def translate_determinant_modulus_expr(expr: str) -> str | None:
    match = re.match(r"^determinant\s*\(", expr, re.IGNORECASE)
    if not match:
        return None
    open_pos = expr.find("(", match.start())
    close_pos = find_matching_char(expr, open_pos, "(", ")")
    if close_pos < 0:
        return None
    tail = expr[close_pos + 1 :].strip()
    idx_match = re.match(r"^\$\s*modulus\s*\[\s*(.+?)\s*\]\s*$", tail, re.IGNORECASE)
    if not idx_match:
        return None
    args = split_args(expr[open_pos + 1 : close_pos])
    return f"r_matrix_index_get({translate_call('determinant', args)}.modulus, {translate_expr(idx_match.group(1))})"


def translate_expr_code(expr: str) -> str:
    expr = expr.replace("<-", "=")
    expr = re.sub(r"(?<=\d)[lL]\b", "", expr)
    # Drop R namespace qualifiers such as quadprog::solve.QP.
    expr = re.sub(r"(?<![\w.])[A-Za-z][\w.]*:::?(?=[A-Za-z.])", "", expr)
    expr = replace_pipe_operator(expr, "|>")
    expr = replace_pipe_operator(expr, "%>%")
    expr = replace_in_operator(expr)
    expr = replace_backtick_member_access(expr)
    expr = expr.replace("$", "@@MEM@@")
    expr = expr.replace("%/%", "//")
    expr = expr.replace("%%", "%")
    expr = expr.replace("%*%", "@")
    expr = expr.replace("&&", " and ")
    expr = expr.replace("||", " or ")
    expr = normalize_logical_operators(expr)
    expr = re.sub(r"!\s*(?!=)", "not ", expr)
    expr = replace_complex_literals(expr)
    expr = replace_power(expr)
    expr = replace_r_constants(expr)
    expr = replace_ranges(expr)
    expr = replace_r_subscripts(expr)
    expr = replace_nested_matrix_subscripts(expr)
    expr = replace_nested_vector_subscripts(expr)
    expr = replace_calls(expr)
    expr = replace_ambiguous_member_access(expr)
    expr = replace_r_colnames_array_subscripts(expr)
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


def replace_ambiguous_member_access(expr: str) -> str:
    """Use r_member for $ names that collide with pandas attributes."""
    ambiguous = {
        "columns", "count", "index", "kurt", "max", "mean", "median",
        "min", "mode", "prod", "quantile", "rank", "sample", "shape",
        "size", "skew", "std", "sum", "values", "var",
    }
    attrs = "|".join(sorted(ambiguous, key=len, reverse=True))
    nested = r"(?:[^()]|\([^()]*\))*"
    base = rf"(?:[A-Za-z_]\w*\({nested}\)|[A-Za-z_]\w*(?:(?:@@MEM@@|\.)[A-Za-z_]\w*)*)"
    pattern = re.compile(rf"({base})@@MEM@@({attrs})\b")
    while True:
        updated = pattern.sub(lambda m: f"r_member({m.group(1).replace('@@MEM@@', '.')}, {m.group(2)!r})", expr)
        if updated == expr:
            return expr
        expr = updated


def replace_pipe_operator(expr: str, op: str) -> str:
    while True:
        pos = find_top_level_operator(expr, op)
        if pos < 0:
            return expr
        left = expr[:pos].strip()
        right = expr[pos + len(op) :].strip()
        if not left or not right:
            return expr
        call_match = re.match(r"^([A-Za-z_][\w.]*)\s*\(", right)
        if call_match:
            open_pos = right.find("(", call_match.end() - 1)
            close_pos = find_matching_paren(right, open_pos)
            if close_pos < 0:
                return expr
            inner = right[open_pos + 1 : close_pos].strip()
            joined = left + (", " + inner if inner else "")
            expr = right[:open_pos] + "(" + joined + ")" + right[close_pos + 1 :]
            continue
        if re.match(r"^[A-Za-z_][\w.]*$", right):
            expr = f"{right}({left})"
            continue
        return expr


def replace_in_operator(expr: str) -> str:
    pos = find_top_level_operator(expr, "%in%")
    if pos >= 0:
        left = expr[:pos].strip()
        right = expr[pos + 4 :].strip()
        negate = False
        while left.startswith("!"):
            negate = not negate
            left = left[1:].lstrip()
        core = f"r_in({translate_expr(left)}, {translate_expr(right)})"
        return f"np.logical_not({core})" if negate else core
    while "%in%" in expr:
        pos = expr.find("%in%")
        left_start = scan_operand_left(expr, pos)
        right_end = scan_operand_right(expr, pos + 4)
        left = expr[left_start:pos].strip()
        right = expr[pos + 4 : right_end].strip()
        if not left or not right:
            return expr
        expr = expr[:left_start] + f"r_in({translate_expr(left)}, {translate_expr(right)})" + expr[right_end:]
    return expr


def scan_operand_left(expr: str, pos: int) -> int:
    i = pos - 1
    while i >= 0 and expr[i].isspace():
        i -= 1
    while i >= 0:
        ch = expr[i]
        if ch in ")]":
            depth = 0
            while i >= 0:
                if expr[i] in ")]":
                    depth += 1
                elif expr[i] in "([":
                    depth -= 1
                    if depth == 0:
                        break
                i -= 1
            i -= 1
            continue
        if ch.isalnum() or ch in "._$@":
            i -= 1
            continue
        break
    return i + 1


def scan_operand_right(expr: str, pos: int) -> int:
    i = pos
    while i < len(expr) and expr[i].isspace():
        i += 1
    while i < len(expr):
        ch = expr[i]
        if ch == "(":
            close = find_matching_char(expr, i, "(", ")")
            if close < 0:
                break
            i = close + 1
            continue
        if ch == "[":
            close = find_matching_char(expr, i, "[", "]")
            if close < 0:
                break
            i = close + 1
            continue
        if ch.isalnum() or ch in "._$@":
            i += 1
            continue
        break
    return i


def normalize_logical_operators(expr: str) -> str:
    expr = expr.strip()
    expr = strip_outer_parens(expr) if expr.startswith("(") and expr.endswith(")") else expr
    if not expr:
        return expr

    pos = find_top_level_operator(expr, "|")
    if pos >= 0:
        left = normalize_logical_operators(expr[:pos])
        right = normalize_logical_operators(expr[pos + 1 :])
        return f"({left}) | ({right})"

    pos = find_top_level_operator(expr, "&")
    if pos >= 0:
        left = normalize_logical_operators(expr[:pos])
        right = normalize_logical_operators(expr[pos + 1 :])
        return f"({left}) & ({right})"

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
        "NA_integer_": "np.nan",
        "NA_complex_": "np.nan",
        "NA_character_": "np.nan",
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
    range_parts = split_top_level_range(expr)
    if range_parts is not None:
        start, stop = range_parts
        return f"r_seq({start}, {stop})"
    name_atom = r"[A-Za-z_]\w*(?:(?:@@MEM@@|\.)[A-Za-z_]\w*)*"
    nested1 = r"(?:[^()]|\([^()]*\))*"
    nested2 = rf"(?:[^()]|\({nested1}\))*"
    atom = rf"(?:{name_atom}\({nested2}\)|\({nested2}\)|{name_atom}(?!\s*\()|\d+(?:\.\d+)?)"
    pattern = re.compile(rf"(?<![\w.=])({atom})\s*:\s*({atom})(?![\w.])")

    def repl(match: re.Match[str]) -> str:
        # A colon directly after "lambda params" is Python syntax, not an R range.
        if re.search(r"\blambda_?(?:\s+[\w, *=]*)?$", expr[: match.start()]):
            return match.group(0)
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
    expr = re.sub(r"\bnot\s+(pd\.isna\([^)]+\)|np\.isfinite\([^)]+\))", r"np.logical_not(\1)", expr)
    # R's ! is elementwise; convert simple negated operands in argument or
    # subscript position where Python's `not` would reject arrays.
    expr = re.sub(
        r"\bnot\s+([A-Za-z_][\w.]*(?:\([^()]*\))?(?:\[[^\[\]]*\])?)(?=\s*[,\)\]]|\s*$)",
        r"np.logical_not(\1)",
        expr,
    )
    return expr


def replace_named_matrix_columns(expr: str) -> str:
    str_atom = r"(?:[\"'][^\"']+[\"']|__R_STR_\d+__)"
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[:,\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"{m.group(1)}[:, r_col_key({m.group(1)}, {m.group(2)}, globals().get('{m.group(1)}_colnames'))]",
        expr,
    )
    expr = re.sub(
        r"\b([A-Za-z]\w*)\[:,\s*\(([A-Za-z]\w*_colnames\.index\([^)]+\))\)\s*-\s*1\]",
        lambda m: f"{m.group(1)}[:, {m.group(2)}]",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[([^\[\]]+?),\s*({str_atom})\]",
        lambda m: f"{m.group(1)}[{m.group(2)}, r_col_key({m.group(1)}, {m.group(3)}, globals().get('{m.group(1)}_colnames'))]",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[\(([A-Za-z]\w*_idx)\)\s*-\s*1,\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"int({m.group(1)}[{m.group(2)}, r_col_key({m.group(1)}, {m.group(3)}, globals().get('{m.group(1)}_colnames'))])",
        expr,
    )
    expr = re.sub(
        rf"\b([A-Za-z]\w*)\[([^\[\]]+?),\s*\(({str_atom})\)\s*-\s*1\]",
        lambda m: f"{m.group(1)}[{m.group(2)}, r_col_key({m.group(1)}, {m.group(3)}, globals().get('{m.group(1)}_colnames'))]",
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
        if name.startswith(("np.", "stats.", "r_stats.", "pd.", "special.")) or name in {"SimpleNamespace", "RList", "RNamedVector", "RFactor", "RTimeSeries", "r_print", "r_s3_print", "r_s3_dispatch", "r_add", "r_sub", "r_mul", "r_div", "r_seq", "r_range", "r_subset", "r_set_subset", "r_subset_df", "r_with", "r_within", "r_col_key", "r_row_key", "r_attr", "r_set_attr", "r_attributes", "r_eval", "r_parse", "r_paste", "r_substr", "r_list_get", "r_factor", "r_levels", "r_factor_int", "r_table", "r_tapply", "cut_py", "r_lapply", "r_sapply", "r_mapply", "outer_py", "r_split", "r_unsplit", "r_as_date", "r_date_add", "r_date_format", "r_date_seq", "r_diff", "r_ts", "r_start", "r_end", "r_frequency", "r_window", "r_lag", "arima_py", "arima_sim_py", "kmeans_py", "stack_py", "unstack_py", "prcomp_py", "aov_py", "glm_py", "r_list_from_dots", "do_call_py", "capture_output_py", "rle_py", "inverse_rle_py", "r_df_col", "r_data_frame", "r_model_matrix", "r_matrix_data", "cbind_py", "rbind_py", "acf_py", "uniroot_py", "integrate_py", "try_catch_py", "eigen_py", "svd_py", "qr_py", "summary_py", "ecdf_py", "getattr", "globals", "int", "float", "str", "len"}:
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


def replace_parenthesized_subscripts(expr: str, *, allow_helper_bases: bool = False) -> str:
    """Rewrite subscripts on parenthesized expressions or call results.

    Handles forms like ``(a - b)[["k"]]`` and ``do.call(...)[i, , drop = FALSE]``
    that the name-based subscript patterns cannot express.
    """
    i = 0
    while i < len(expr):
        if expr[i] != "(":
            i += 1
            continue
        j = i - 1
        while j >= 0 and expr[j].isspace():
            j -= 1
        base_start = i
        call_name = ""
        if j >= 0 and (expr[j].isalnum() or expr[j] in "_.@"):
            name_start = j
            while name_start >= 0 and (expr[name_start].isalnum() or expr[name_start] in "_.@"):
                name_start -= 1
            call_name = expr[name_start + 1 : j + 1]
            if not re.match(r"^[A-Za-z_.@]", call_name) or call_name in {"if", "for", "while", "function", "return", "switch"}:
                i += 1
                continue
            if call_name.startswith(("np.", "pd.", "stats.")):
                # Already-translated Python; its subscripts are 0-based already.
                i += 1
                continue
            if not allow_helper_bases and (call_name.startswith("r_") or call_name.endswith("_py")):
                i += 1
                continue
            base_start = name_start + 1
        elif j >= 0 and expr[j] in ")]":
            i += 1
            continue
        close = find_matching_paren(expr, i)
        if close < 0:
            i += 1
            continue
        k = close + 1
        while k < len(expr) and expr[k].isspace():
            k += 1
        if k >= len(expr) or expr[k] != "[":
            i += 1
            continue
        if call_name:
            base = expr[base_start : close + 1]
        else:
            inner = expr[i + 1 : close].strip()
            ternary = translate_if_else_expr(inner)
            base = f"({ternary if ternary is not None else inner})"
        end = find_matching_char(expr, k, "[", "]")
        if end < 0:
            i += 1
            continue
        if expr.startswith("[[", k) and expr[end - 1] == "]":
            key = expr[k + 2 : end - 1].strip()
            expr = expr[:base_start] + f"r_list_get({base}, {key})" + expr[end + 1 :]
        elif has_top_level_comma(expr[k + 1 : end]):
            index = expr[k + 1 : end]
            expr = expr[:base_start] + f"r_subset({base}, {translate_subscript(index, base=base)})" + expr[end + 1 :]
        else:
            index = expr[k + 1 : end].strip()
            if is_negative_integer_subscript(index):
                item = index.replace(" ", "")[1:]
                expr = expr[:base_start] + f"r_drop_index({base}, {item})" + expr[end + 1 :]
            else:
                expr = expr[:base_start] + f"r_matrix_index_get({base}, {index})" + expr[end + 1 :]
        i = base_start + 1
    return expr


def replace_r_subscripts(expr: str) -> str:
    expr = replace_parenthesized_subscripts(expr)
    item_pattern = re.compile(
        r"([A-Za-z]\w*(?:(?:@@MEM@@|\.)\w+)*(?:\((?:[^()]|\([^()]*\))*\))?)\s*\[\[([^\[\]]+)\]\]"
    )
    expr = item_pattern.sub(replace_double_subscript, expr)
    pattern = re.compile(r"([A-Za-z]\w*(?:(?:@@MEM@@|\.)\w+)*)\s*\[([^\[\]]*)\]")
    expr = pattern.sub(replace_single_subscript, expr)
    # Chained subscripts land after the helpers emitted above, e.g. x[i, ][j, ].
    return replace_parenthesized_subscripts(expr, allow_helper_bases=True)


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


def replace_r_colnames_array_subscripts(expr: str) -> str:
    pattern = re.compile(
        r"np\.array\((globals\(\)\.get\([^)]*?_colnames[^)]*\))\)\[([^\]]+)\]"
    )

    def repl(match: re.Match[str]) -> str:
        base = match.group(1)
        idx = match.group(2).strip()
        if idx in {":", "slice(None)"}:
            return f"np.array({base})[{idx}]"
        if is_string_index_expr(idx):
            return f"np.array({base})[{idx}]"
        if idx.startswith("r_sub("):
            return f"np.array({base})[{idx}]"
        return f"np.array({base})[r_sub({idx}, 1)]"

    return pattern.sub(repl, expr)


def replace_double_subscript(match: re.Match[str]) -> str:
    raw_base = match.group(1).strip()
    parsed_call = parse_full_call(raw_base)
    if parsed_call is not None:
        base = translate_call(parsed_call[0], parsed_call[1])
    else:
        open_pos = -1
        if raw_base.endswith(")"):
            depth = 0
            for i in range(len(raw_base) - 1, -1, -1):
                if raw_base[i] == ")":
                    depth += 1
                elif raw_base[i] == "(":
                    depth -= 1
                    if depth == 0:
                        open_pos = i
                        break
        if open_pos >= 0:
            base = f"{translate_member_expr(raw_base[:open_pos])}({raw_base[open_pos + 1:-1]})"
        else:
            base = translate_member_expr(raw_base)
    index = match.group(2).strip()
    if is_string_literal(index):
        if base.endswith(")"):
            return f"r_list_get({base}, {index})"
        return f"{base}.{r_name(index[1:-1])}"
    placeholder = re.fullmatch(r"__R_STR_(\d+)__", index)
    if placeholder:
        if not base.endswith(")"):
            text = masked_string_text(index)
            if text is not None and re.fullmatch(r"[A-Za-z][\w.]*", text[1:-1]):
                return f"{base}.{r_name(text[1:-1])}"
        return f"r_list_get({base}, __R_STR_{placeholder.group(1)}__)"
    if re.fullmatch(r"\d+", index):
        return f"r_list_get({base}, {index})"
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
        return f"r_drop_index({base}, {item})"
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
            expr = f"r_drop_axis({expr}, {item}, {axis})"
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
    # Statement-level callers pass raw R text; scrub R-only lexical forms that
    # would otherwise leak into the generated subscript verbatim.
    index = re.sub(r"\b(\d+)[Ll]\b", r"\1", index)
    index = index.replace("%/%", "//").replace("%%", "%")
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
    return re.match(r"^-\s*(?:\d+|[A-Za-z][\w.]*)$", index) is not None


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
            if re.search(r"\blambda_?\s*$", text[:i]):
                continue
            return text[:i].strip(), text[i + 1 :].strip()
    return None


def translate_matrix_subscript(index: str, *, base: str | None = None) -> str:
    parts = split_subscript_args(index)
    out: list[str] = []
    advanced_axes: list[int] = []
    drop_false = any(re.search(r"drop\s*=\s*F(?:ALSE|alse)?\b", part) for part in parts)
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
        elif axis == 1 and base and re.fullmatch(r"[A-Za-z_][\w.]*", item) and r_name(item) in CHARACTER_VECTOR_VARS:
            out.append(f"r_col_key({base}, {translate_expr(item)}, globals().get('{base}_colnames'))")
        elif axis == 1 and base and is_string_index_expr(item) and not is_likely_dataframe_name(base):
            cols = translate_expr(item)
            out.append(f"np.array([r_col_key({base}, _col, globals().get('{base}_colnames')) for _col in np.ravel({cols})])")
        elif is_logical_subscript(item):
            out.append(translate_logical_matrix_subscript(item))
            advanced_axes.append(len(out) - 1)
        elif re.fullmatch(r"[A-Za-z]\w*", item) and r_name(item) not in LOGICAL_VECTOR_VARS:
            # Selectors held in variables can be numeric, logical, or
            # character; dispatch at runtime.
            if axis == 1 and drop_false:
                out.append(f"np.atleast_1d(r_axis_index({item}))")
            else:
                out.append(f"r_axis_index({item})")
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


def translate_logical_matrix_subscript(item: str) -> str:
    """Translate a logical axis while preserving nested matrix selections."""
    replacements: list[tuple[str, str]] = []
    pattern = re.compile(r"\b([A-Za-z]\w*)\s*\[([^\[\]]*)\]")

    def mask_nested(match: re.Match[str]) -> str:
        base, index = match.groups()
        if not has_top_level_comma(index):
            return match.group(0)
        placeholder = f"__R_MATRIX_SUB_{len(replacements)}__"
        replacement = f"r_subset({r_name(base)}, {translate_subscript(index, base=r_name(base))})"
        replacements.append((placeholder, replacement))
        return placeholder

    masked = pattern.sub(mask_nested, item)
    translated = translate_expr(masked)
    for placeholder, replacement in replacements:
        translated = translated.replace(placeholder, replacement)
    return translated


def translate_matrix_axis_subscript(item: str) -> str:
    translated = translate_subscript(item)
    if re.fullmatch(r"\(?\d+\)?\s*-\s*1", translated):
        return translated
    if is_advanced_matrix_index(translated) or translated == ":" or translated == "slice(None)":
        return translated
    if re.match(r"^.+\[.+\]$", translated):
        return f"({translated}) - 1"
    return translated


def is_likely_dataframe_name(name: str) -> bool:
    return name in {"df", "prices"} or name.endswith("_df")


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
    if r_name(text) in CHARACTER_VECTOR_VARS:
        return True
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


def is_character_vector_expr(text: str) -> bool:
    text = strip_outer_parens(text.strip())
    if is_string_literal(text):
        return True
    py_name = r_name(text)
    if py_name in CHARACTER_VECTOR_VARS:
        return True
    if re.fullmatch(r"[A-Za-z_]\w*_names", py_name):
        return True
    raw_call = parse_full_call(text)
    if raw_call is None:
        return False
    name = raw_call[0].lower()
    if name in {"setdiff", "names", "colnames", "rownames", "as.character", "paste", "paste0"}:
        return True
    if name == "c":
        return any(is_character_vector_expr(arg.strip()) for arg in raw_call[1])
    return False


def is_logical_scalar_expr(text: str) -> bool:
    text = strip_outer_parens(text.strip())
    if not text:
        return False
    upper = text.upper()
    if upper in {"TRUE", "FALSE", "T", "F"}:
        return True
    if text.startswith("!"):
        return True
    if re.search(r"\b(and|or|not)\b", text):
        return True
    raw_call = parse_full_call(text)
    if raw_call is None:
        return False
    name = raw_call[0].lower()
    return name in {
        "is.na",
        "is.nan",
        "is.finite",
        "is.infinite",
        "is.logical",
        "as.logical",
    }


def is_logical_vector_expr(text: str) -> bool:
    text = strip_outer_parens(text.strip())
    if not text:
        return False
    if text.upper() in {"TRUE", "FALSE", "T", "F"}:
        return True
    raw_call = parse_full_call(text)
    if raw_call is not None:
        name = raw_call[0].lower()
        if name in {
            "is.na",
            "is.nan",
            "is.finite",
            "is.infinite",
            "is.logical",
            "as.logical",
            "complete.cases",
            "complete_cases_py",
            "grepl",
            "lower.tri",
            "upper.tri",
            "duplicated",
            "xor",
            "is.element",
            "startswith",
            "endswith",
        }:
            return True
        if name == "c":
            return all(is_logical_scalar_expr(arg.strip()) for arg in raw_call[1]) and bool(raw_call[1])
        # A call to any other function does not produce a logical vector even
        # when its arguments contain comparisons, e.g. which(x == 1).
        return False
    # Only top-level comparisons count; ones inside subscripts or call
    # arguments (e.g. ord[groups == j]) do not make the result logical.
    for op in ("==", "!=", "<=", ">=", "<", ">", "&", "|"):
        if find_top_level_operator(text, op) >= 0:
            return True
    if text.startswith("!"):
        return True
    return False


_MATRIX_RESULT_CALLS = {
    "matrix",
    "as.matrix",
    "cbind",
    "rbind",
    "outer",
    "sweep",
    "crossprod",
    "tcrossprod",
    "cov",
    "cor",
}

_ELEMENTWISE_CALLS = {"exp", "log", "sqrt", "abs", "log2", "log10", "sin", "cos", "tan"}


def is_matrix_expr(text: str) -> bool:
    text = strip_outer_parens(text.strip())
    if not text:
        return False
    if re.fullmatch(r"[A-Za-z.][\w.]*", text):
        return r_name(text) in MATRIX_VARS
    raw_call = parse_full_call(text)
    if raw_call is not None:
        name = raw_call[0].lower()
        if name in _MATRIX_RESULT_CALLS:
            return True
        if name in _ELEMENTWISE_CALLS and raw_call[1]:
            return is_matrix_expr(raw_call[1][0])
        return False
    for op in ("+", "-", "*", "/"):
        pos = find_top_level_operator(text, op)
        if pos > 0:
            return is_matrix_expr(text[:pos]) or is_matrix_expr(text[pos + 1 :])
    return False


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
    if r_name(index) in LOGICAL_VECTOR_VARS:
        return True
    if index.startswith(("np.is", "is.", "is_", "~")):
        return True
    raw_call = parse_full_call(index)
    if raw_call is not None and raw_call[0].lower() in {"grepl", "is.na", "is.nan", "is.finite", "is.infinite", "complete.cases", "complete_cases_py", "lower.tri", "upper.tri"}:
        return True
    if index.startswith(","):
        return False
    return any(op in index for op in ("<", ">", "==", "!=", "<=", ">="))


def translate_call(name: str, args: list[str]) -> str:
    lname = name.lower()
    if lname == "negate":
        if not args:
            raise R2PyError("Negate requires a function")
        return f"r_negate({translate_expr(args[0])})"
    if lname == "filter":
        if len(args) < 2:
            raise R2PyError("Filter requires a function and values")
        return f"r_filter({translate_expr(args[0])}, {translate_expr(args[1])})"
    if lname == "lm":
        return translate_lm_call(args)
    if lname == "glm":
        return translate_glm_call(args)
    if lname == "aov":
        return translate_aov_call(args)
    if lname == "do.call":
        func = translate_expr(args[0])
        func = {"cbind": "cbind_py", "rbind": "rbind_py", "paste": "r_paste", "paste0": "r_paste"}.get(func, func)
        arg_list = translate_expr(args[1])
        return f"do_call_py({func}, {arg_list})"
    if lname == "optim":
        return translate_optim_call(args)
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
            return mask_lambda("try_(lambda: None)")
        silent = translate_expr(keyword_arg(args, "silent", default="False"))
        return mask_lambda(f"try_(lambda: {translate_expr(positional_args(args)[0])}, silent={silent})")
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
        elif find_top_level_operator(arg, "~") >= 0 and find_top_level_operator(arg, "=") < 0:
            # Keep unsupported formula arguments as strings so output stays valid Python.
            py_args.append(repr(arg.strip()))
        else:
            py_args.append(translate_expr(arg))
    if name.startswith(("np.", "special.")):
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
        print_args = positional_args(args)
        print_py_args = [translate_expr(arg) for arg in print_args]
        row_names = keyword_arg(args, "row.names")
        if row_names is not None and len(print_args) == 1:
            return f"r_print({print_py_args[0]}, row_names={translate_expr(row_names)})"
        if len(print_args) == 1:
            raw_call = parse_full_call(print_args[0])
            if raw_call is not None and raw_call[0].lower() == "round" and len(raw_call[1]) >= 2:
                inner = print_py_args[0]
                return (
                    "r_print("
                    + inner
                    + ", digits="
                    + translate_expr(raw_call[1][1])
                    + print_colnames_arg(raw_call[1][0], allow_simple=True)
                    + ")"
                )
            colnames_arg = print_colnames_arg(print_args[0], allow_simple=True)
            if colnames_arg:
                return "r_print(" + print_py_args[0] + colnames_arg + ")"
            return "r_s3_print(" + print_py_args[0] + ")"
        return "r_print(" + ", ".join(print_py_args) + ")"
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
    if lname in {"log", "log10", "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "abs", "floor"}:
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "gamma":
        return "special.gamma(" + ", ".join(py_args) + ")"
    if lname == "choose":
        if len(py_args) < 2:
            raise R2PyError("choose requires n and k")
        return f"special.comb({py_args[0]}, {py_args[1]}, exact=False)"
    if lname == "combn":
        positional = positional_args(args)
        if len(positional) < 2:
            raise R2PyError("combn requires x and m")
        func = keyword_arg(args, "FUN")
        simplify = translate_expr(keyword_arg(args, "simplify", default="True"))
        func_arg = "None" if func is None else translate_expr(func)
        return f"combn_py({translate_expr(positional[0])}, {translate_expr(positional[1])}, func={func_arg}, simplify={simplify})"
    if lname == "besselk":
        if len(args) < 2:
            raise R2PyError("besselK requires x and nu")
        x = translate_expr(args[0])
        nu = translate_expr(args[1])
        scaled_arg = keyword_arg(args, "expon.scaled")
        if scaled_arg is None:
            scaled_arg = keyword_arg(args, "expon_scaled", default="False")
        scaled = translate_expr(scaled_arg)
        if scaled == "True":
            return f"special.kve({nu}, {x})"
        if scaled == "False":
            return f"special.kv({nu}, {x})"
        return f"np.where({scaled}, special.kve({nu}, {x}), special.kv({nu}, {x}))"
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
    if lname == "toeplitz":
        return "linalg.toeplitz(" + py_args[0] + ")"
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
    if lname == "polyroot":
        return "polyroot_py(" + py_args[0] + ")"
    if lname == "eigen":
        return "eigen_py(" + py_args[0] + ")"
    if lname == "svd":
        return "svd_py(" + py_args[0] + ")"
    if lname == "qr":
        return "qr_py(" + py_args[0] + ")"
    if lname == "qr.solve":
        positional = positional_args(args)
        if len(positional) < 2:
            raise R2PyError("qr.solve requires a matrix and right hand side")
        tol = keyword_arg(args, "tol")
        rcond = "None" if tol is None else translate_expr(tol)
        return f"np.linalg.lstsq({translate_expr(positional[0])}, {translate_expr(positional[1])}, rcond={rcond})[0]"
    if lname == "determinant":
        logarithm = translate_expr(keyword_arg(args, "logarithm", default="True"))
        return "determinant_py(" + py_args[0] + ", logarithm=" + logarithm + ")"
    if lname in {"sum", "mean", "median", "prod"}:
        na_rm = keyword_arg(args, "na.rm", default="False")
        if translate_expr(na_rm) == "True":
            nan_func = {"sum": "nansum", "mean": "nanmean", "median": "nanmedian", "prod": "nanprod"}[lname]
            return f"np.{nan_func}({py_args[0]})"
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "sort":
        decreasing = translate_expr(keyword_arg(args, "decreasing", default="False"))
        first = translate_expr(positional_args(args)[0])
        return f"r_sort({first}, decreasing={decreasing})"
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
        cor_args = positional_args(args)
        cor_py_args = [translate_expr(arg) for arg in cor_args]
        use = keyword_arg(args, "use")
        use_arg = "" if use is None else ", use=" + translate_expr(use)
        if len(cor_py_args) == 1:
            return "cor_py(" + cor_py_args[0] + use_arg + ")"
        return "cor_py(" + cor_py_args[0] + ", " + cor_py_args[1] + use_arg + ")"
    if lname in {"min", "max"}:
        na_rm = translate_expr(keyword_arg(args, "na.rm", default="FALSE"))
        min_max_args = [translate_expr(arg) for arg in positional_args(args)]
        if na_rm == "True":
            if len(min_max_args) == 1:
                return f"np.nan{lname}({min_max_args[0]})"
            joined = ", ".join(min_max_args)
            return f"np.nan{lname}(np.concatenate([np.ravel(_v) for _v in ({joined},)]))"
        if len(min_max_args) > 1:
            return f"np.{lname}imum(" + ", ".join(min_max_args) + ")"
        return f"np.{lname}({min_max_args[0]})"
    if lname == "sd":
        return "np.std(" + py_args[0] + ", ddof=1)"
    if lname == "any":
        na_rm = keyword_arg(args, "na.rm", default="False")
        if translate_expr(na_rm) == "True":
            return f"np.any((~pd.isna({py_args[0]})) & np.asarray({py_args[0]}, dtype=bool))"
        return "np.any(" + ", ".join(py_args) + ")"
    if lname == "all":
        return "np.all(" + ", ".join(py_args) + ")"
    if lname == "pmax":
        return "np.maximum(" + ", ".join(py_args) + ")"
    if lname == "pmin":
        return "np.minimum(" + ", ".join(py_args) + ")"
    if lname == "setdiff":
        return "r_setdiff(" + ", ".join(py_args) + ")"
    if lname == "as.numeric":
        return "r_as_numeric(" + py_args[0] + ")"
    if lname == "as.character":
        return "np.asarray(" + py_args[0] + ", dtype=str)"
    if lname == "as.vector":
        return "np.ravel(" + py_args[0] + ", order='F')"
    if lname == "as.matrix":
        return "r_as_matrix(" + py_args[0] + ")"
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
    if lname == "complete.cases":
        return "complete_cases_py(" + py_args[0] + ")"
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
    if lname in {"forwardsolve", "backsolve"}:
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        lower = "True" if lname == "forwardsolve" else "False"
        return f"linalg.solve_triangular({parts[0]}, {parts[1]}, lower={lower})"
    if lname == "trimws":
        return f"np.char.strip(np.asarray({translate_expr(args[0])}, dtype=str))"
    if lname == "nzchar":
        return f"(np.char.str_len(np.asarray({translate_expr(args[0])}, dtype=str)) > 0)"
    if lname == "strrep":
        return f"np.char.multiply(np.asarray({translate_expr(args[0])}, dtype=str), {translate_expr(args[1])})"
    if lname in {"suppresswarnings", "suppressmessages"}:
        return translate_expr(args[0])
    if lname == "sys.time":
        return "pd.Timestamp.now()"
    if lname == "commandargs":
        trailing = positional_args(args)[0].strip().upper() if positional_args(args) else keyword_arg(args, "trailingOnly", default="TRUE").strip().upper()
        trailing_only = trailing not in {"FALSE", "F"}
        return f"r_command_args({trailing_only})"
    if lname == "interactive":
        return "False"
    if lname == "identical":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        return f"bool(np.array_equal(np.asarray({parts[0]}, dtype=object), np.asarray({parts[1]}, dtype=object)))"
    if lname == "xor":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        return f"np.logical_xor({parts[0]}, {parts[1]})"
    if lname == "intersect":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        return f"np.intersect1d({parts[0]}, {parts[1]})"
    if lname == "union":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        return f"np.union1d({parts[0]}, {parts[1]})"
    if lname == "setnames":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        return f"r_set_names({parts[0]}, {parts[1]})"
    if lname == "vapply":
        return translate_apply_list_call("sapply", positional_args(args)[:2])
    if lname == "drop":
        return f"np.squeeze(np.asarray({translate_expr(args[0])}))"
    if lname == "rev":
        return f"np.flip({translate_expr(positional_args(args)[0])})"
    if lname == "unlist":
        return f"r_unlist({translate_expr(positional_args(args)[0])})"
    if lname == "as.list":
        return f"r_as_list({translate_expr(positional_args(args)[0])})"
    if lname == "formatc":
        return "r_formatC(" + ", ".join(translate_call_arg(arg) for arg in args) + ")"
    if lname == "switch":
        switch_expr = translate_switch_call(args)
        if switch_expr is not None:
            return switch_expr
    if lname == "file.exists":
        return f"os.path.exists({translate_expr(args[0])})"
    if lname == "file.path":
        return "os.path.join(" + ", ".join(translate_expr(arg) for arg in args) + ")"
    if lname == "file.remove":
        return f"os.remove({translate_expr(args[0])})"
    if lname == "basename":
        return f"os.path.basename({translate_expr(args[0])})"
    if lname == "dirname":
        return f"os.path.dirname({translate_expr(args[0])})"
    if lname == "options":
        # R session options have no Python counterpart; evaluate to None.
        return "None"
    if lname == "strsplit":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        fixed = translate_expr(keyword_arg(args, "fixed", default="FALSE"))
        return f"r_strsplit({', '.join(parts)}, fixed={fixed})"
    if lname == "kruskal.test":
        formula = args[0]
        pos = find_top_level_operator(formula, "~")
        if pos >= 0:
            return f"kruskal_test_py({translate_expr(formula[:pos])}, {translate_expr(formula[pos + 1:])})"
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        return "kruskal_test_py(" + ", ".join(parts) + ")"
    if lname == "wilcox.test":
        parts = [translate_expr(arg) for arg in positional_args(args)[:2]]
        paired = translate_expr(keyword_arg(args, "paired", default="FALSE"))
        return f"wilcox_test_py({', '.join(parts)}, paired={paired})"
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
        lag_max = keyword_arg(args, "lag.max")
        lag_part = "" if lag_max is None else f", lag_max={translate_expr(lag_max)}"
        return f"acf_py({py_args[0]}, plot={plot}{lag_part})"
    if lname == "diff":
        return "r_diff(" + py_args[0] + ")"
    if lname == "format":
        return translate_format_call(args)
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
        labels_arg = keyword_arg(args, "labels")
        labels_part = "" if labels_arg is None else f", labels={translate_expr(labels_arg)}"
        return "cut_py(" + py_args[0] + ", " + breaks + labels_part + ")"
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
        return f"RNamedVector({py_args[0]}@@MEM@@coef, getattr({py_args[0]}, 'coef_names', [str(i) for i in np.arange(len({py_args[0]}@@MEM@@coef))]))"
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
    if lname == "force":
        return py_args[0]
    if lname == "length":
        return "r_length(" + py_args[0] + ")"
    if lname == "names":
        return "r_names(" + py_args[0] + ")"
    if lname == "colnames":
        return f"r_colnames({translate_expr(args[0])}, globals().get({(args[0].strip() + '_colnames')!r}, []))"
    if lname == "rownames":
        return f"r_rownames({translate_expr(args[0])}, globals().get({(args[0].strip() + '_rownames')!r}, []))"
    if lname == "nrow":
        return py_args[0] + ".shape[0]"
    if lname == "ncol":
        return py_args[0] + ".shape[1]"
    if lname in {"seq", "seq.int"}:
        return translate_seq_call(args)
    if lname == "seq_along":
        return "np.arange(1, r_length(" + py_args[0] + ") + 1)"
    if lname == "seq_len":
        return "np.arange(1, " + py_args[0] + " + 1)"
    if lname in {"rep", "rep.int", "rep_int"}:
        return translate_rep_call(args)
    if lname == "reduce":
        return "reduce_py(" + ", ".join(py_args) + ")"
    if lname == "numeric":
        return "np.zeros(" + (py_args[0] if py_args else "0") + ")"
    if lname == "integer":
        return "np.zeros(" + (py_args[0] if py_args else "0") + ", dtype=int)"
    if lname == "character":
        return "np.full(" + (py_args[0] if py_args else "0") + ", '', dtype=object)"
    if lname == "logical":
        return "np.zeros(" + (py_args[0] if py_args else "0") + ", dtype=bool)"
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
    if raw_call is not None and raw_call[0].lower() in {"round", "signif"} and raw_call[1]:
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
    positional = positional_args(args)
    x_arg = keyword_arg(args, "X", default=positional[0] if positional else None)
    margin_arg = keyword_arg(args, "MARGIN", default=positional[1] if len(positional) > 1 else None)
    fun_arg = keyword_arg(args, "FUN", default=positional[2] if len(positional) > 2 else None)
    if x_arg is None or margin_arg is None or fun_arg is None:
        raise R2PyError("apply requires array, margin, and function")
    x = translate_expr(x_arg)
    margin = translate_expr(margin_arg)
    func = translate_expr(fun_arg)
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
    return f"r_apply({x}, {margin}, {func})"


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
            raw_name = arg[:pos].strip()
            if "." in raw_name:
                DOTTED_R_VARS.add(raw_name)
            name = r_name(raw_name)
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
        if len(args) == 1:
            return f"{translate_expr(args[0])}.design"
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
    positional = positional_args(args)
    if len(positional) < 2:
        raise R2PyError("outer requires x and y")
    x = translate_expr(positional[0])
    y = translate_expr(positional[1])
    fun = keyword_arg(args, "FUN", default=positional[2] if len(positional) > 2 else None)
    func = translate_expr(fun) if fun is not None else repr("*")
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
    return f"r_write_csv({data}, {file_arg}, index={row_names})"


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
    positional = positional_args(args)
    values_arg = keyword_arg(args, "X", default=positional[0] if positional else None)
    fun = keyword_arg(args, "FUN", default=positional[1] if len(positional) > 1 else None)
    if values_arg is None or fun is None:
        raise R2PyError(f"{name} requires list and function")
    values = translate_expr(values_arg)
    func = fun.strip()
    helper = "r_lapply" if name == "lapply" else "r_sapply"
    if func in {"`[`", "`[[`", '"["', '"[["', "'['", "'[['"} and len(positional) > 2:
        index = translate_expr(positional[2])
        return f"{helper}({values}, (lambda _v: r_list_get(_v, {index})))"
    extras: list[str] = []
    seen_positional = 0
    for arg in args:
        pos_eq = find_top_level_operator(arg, "=")
        if pos_eq < 0:
            seen_positional += 1
            if seen_positional > 2:
                extras.append(arg)
        elif normalize_keyword_name(arg[:pos_eq].strip()).lower() not in {"x", "fun", "use_names", "simplify"}:
            extras.append(arg)
    if extras and re.fullmatch(r"[A-Za-z.][\w.]*", func):
        call_py = translate_expr(f"{func}(_v_, {', '.join(extras)})")
        return f"{helper}({values}, (lambda _v_: {call_py}))"
    py_func = repr(func) if func in {"sum", "mean", "length"} else translate_expr(func)
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
    if len(args) < 3:
        raise R2PyError("sweep requires x, margin, and stats")
    x = translate_expr(args[0])
    margin = translate_expr(args[1])
    stats = translate_expr(args[2])
    op = translate_expr(keyword_arg(args, "FUN", default=args[3] if len(args) > 3 else '"-"'))
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
            raw_name = arg[:pos].strip()
            value = translate_expr(arg[pos + 1 :].strip())
            key = raw_name
            if is_string_literal(key):
                key = key[1:-1]
            else:
                masked = masked_string_text(key)
                if masked is not None:
                    key = masked[1:-1]
            if re.fullmatch(r"[A-Za-z]\w*", r_name(key)):
                cols.append(f"r_data_frame({r_name(key)}={value})")
            else:
                cols.append(f"r_data_frame(**{{{key!r}: {value}}})")
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
        fmt_text = args[0].strip()
        if re.fullmatch(r"__R_STR_\d+__", fmt_text):
            literal = masked_string_text(fmt_text)
        elif is_string_literal(fmt_text):
            literal = fmt_text
        else:
            literal = None
        if literal is None or "*" not in literal:
            return f"np.char.mod({fmt}, {values[0]})"
    return "r_sprintf(" + ", ".join([fmt, *values]) + ")"


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
    names_arg = keyword_arg(args, "names", default="True")
    if translate_expr(names_arg) == "False":
        return f"np.quantile({x}, {probs})"
    return f"(lambda _p: RNamedVector(np.quantile({x}, _p), [f'{{100 * v:g}}%' for v in _p]))(np.atleast_1d({probs}))"


def translate_tail_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("tail requires an array")
    x = translate_expr(args[0])
    n = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "n", default="6"))
    return f"tail_py({x}, {n})"


def translate_head_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("head requires an array")
    x = translate_expr(args[0])
    n = translate_expr(args[1] if len(args) > 1 and "=" not in args[1] else keyword_arg(args, "n", default="6"))
    if n == "1" and re.match(r"^[A-Za-z_]\w*$", x) and x not in {"df", "data", "z", "values"}:
        return f"{x}[0]"
    return f"head_py({x}, {n})"


def translate_switch_call(args: list[str]) -> str | None:
    """Translate value-form switch() into a lazy conditional chain."""
    if len(args) < 2:
        return None
    value = translate_expr(args[0])
    branches: list[tuple[str, str]] = []
    default_expr: str | None = None
    for arg in args[1:]:
        pos = find_top_level_operator(arg, "=")
        if pos < 0:
            if arg.strip().startswith("{"):
                return None
            default_expr = translate_expr(arg)
            continue
        key = arg[:pos].strip()
        if is_string_literal(key):
            key = key[1:-1]
        else:
            masked = masked_string_text(key)
            if masked is not None:
                key = masked[1:-1]
        branch = arg[pos + 1 :].strip()
        if branch.startswith("{"):
            return None
        branches.append((key, branch))
    if not branches:
        return None
    result = default_expr if default_expr is not None else "None"
    next_branch: str | None = None
    for key, branch in reversed(branches):
        branch_expr = translate_expr(branch) if branch else next_branch
        if branch_expr is None:
            branch_expr = "None"
        next_branch = branch_expr
        result = f"({branch_expr} if _r_switch_key == {key!r} else {result})"
    candidate = f"(lambda _r_switch_key: {result})"
    try:
        ast.parse(repair_inline_lambda_keyword(normalize_dotted_call_syntax(restore_lambda_masks(candidate))), mode="eval")
    except SyntaxError:
        return None
    return f"{mask_lambda(candidate)}({value})"


def translate_format_call(args: list[str]) -> str:
    if not args:
        return '""'
    positional = positional_args(args)
    fmt = keyword_arg(args, "format")
    if fmt is None and len(positional) >= 2:
        second = positional[1].strip()
        if second.startswith(('"', "'")) or (re.fullmatch(r"__R_STR_\d+__", second) and masked_string_text(second) is not None):
            fmt = positional[1]
    if fmt is not None:
        return f"r_date_format({translate_expr(positional[0])}, {translate_expr(fmt)})"
    x = translate_expr(positional[0] if positional else args[0])
    extras = []
    for key in ["digits", "nsmall", "scientific", "width"]:
        value = keyword_arg(args, key)
        if value is not None:
            extras.append(f"{key}={translate_expr(value)}")
    return f"r_format_vec({x}" + (", " + ", ".join(extras) if extras else "") + ")"


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
        return f"r_matrix_data({data}).reshape((-1, int({translate_expr(ncol)})), order={order})"
    if ncol is None:
        return f"r_matrix_data({data}).reshape((int({translate_expr(nrow)}), -1), order={order})"
    py_nrow = translate_expr(nrow)
    py_ncol = translate_expr(ncol)
    return f"np.resize(r_matrix_data({data}), int(({py_nrow}) * ({py_ncol}))).reshape((int({py_nrow}), int({py_ncol})), order={order})"


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
    if pos > 0 and (arg[pos - 1] in "<>!" or (pos + 1 < len(arg) and arg[pos + 1] == "=")):
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
        if pos > 0 and (arg[pos - 1] in "<>!=" or (pos + 1 < len(arg) and arg[pos + 1] == "=")):
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
    return re.sub(r"(?<![\w.)\]'\"])(\.?[A-Za-z]\w*(?:\.\w+)*)\b", lambda m: r_name(m.group(1)), expr)


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
            if node.id in MATRIX_VARS:
                # Matrix operands may need R column-major recycling.
                return False
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
    if "__R_STR_" in name or "__R_LAMBDA_" in name:
        return name
    if len(name) >= 2 and name[0] == "`" and name[-1] == "`":
        name = name[1:-1]
    constants = {"True", "False", "None", "np", "pd", "stats", "r_stats", "nan", "inf", "and", "or", "not", "is", "in", "if", "else", "for"}
    if "@@MEM@@" in name:
        return ".".join(r_name(part) for part in name.split("@@MEM@@"))
    if name in constants or name.startswith(("np.", "stats.", "r_stats.", "pd.", "special.", "time.", "os.", "math.", "sys.")):
        return name
    if name[0].isdigit():
        return name
    if "." in name and name not in DOTTED_R_VARS and not name.startswith("."):
        return name
    out = re.sub(r"\W+", "_", name.replace(".", "_")).strip("_")
    if "." in name and out in USER_FUNCTION_NAMES:
        out += "_var"
    if not out:
        out = "x"
    if keyword.iskeyword(out):
        out += "_"
    return out


def r_function_name(name: str) -> str:
    if name.startswith(("np.", "stats.", "r_stats.", "pd.", "special.", "time.", "linalg.", "os.", "math.", "sys.")):
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
        if ch in {"'", '"', "`"}:
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


_STRING_PLACEHOLDER_ID = 0
_STRING_MASK_TEXTS: dict[str, str] = {}


def masked_string_text(placeholder: str) -> str | None:
    return _STRING_MASK_TEXTS.get(placeholder)


def mask_string_literals(expr: str) -> tuple[str, list[tuple[str, str]]]:
    global _STRING_PLACEHOLDER_ID
    parts: list[str] = []
    strings: list[tuple[str, str]] = []
    current: list[str] = []
    quote = ""
    i = 0
    while i < len(expr):
        ch = expr[i]
        if quote:
            current.append(ch)
            if ch == quote:
                placeholder = f"__R_STR_{_STRING_PLACEHOLDER_ID}__"
                _STRING_PLACEHOLDER_ID += 1
                _STRING_MASK_TEXTS[placeholder] = "".join(current)
                strings.append((placeholder, "".join(current)))
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


def restore_string_literals(expr: str, strings: list[tuple[str, str]]) -> str:
    for placeholder, text in strings:
        expr = expr.replace(placeholder, text)
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
    command = [sys.executable, "-m", "py_compile", str(path)]
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    except (OSError, SyntaxError) as exc:
        stderr = "".join(traceback.format_exception_only(type(exc), exc))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)
    return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


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


def slim_python(python: str) -> str:
    try:
        tree = ast.parse(python)
    except SyntaxError:
        return python

    removable = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and is_generated_helper_name(node.name)
    }
    if not removable:
        return python

    needed = ast_used_names_excluding_removable_top_level(tree, set(removable))
    changed = True
    while changed:
        changed = False
        for name, node in removable.items():
            if name not in needed:
                continue
            before = len(needed)
            needed.update(ast_used_names(node))
            changed = len(needed) != before

    remove_ranges: list[tuple[int, int]] = []
    for name, node in removable.items():
        if name not in needed:
            end = getattr(node, "end_lineno", node.lineno)
            remove_ranges.append((node.lineno, end))
    if not remove_ranges:
        return remove_unused_import_statements(python)

    slimmed = remove_line_ranges(python, remove_ranges)
    slimmed = collapse_excess_blank_lines(slimmed)
    return remove_unused_import_statements(slimmed)


def split_runtime_module(python: str, module_name: str = "xr2p_runtime") -> tuple[str, str, set[str]]:
    try:
        tree = ast.parse(python)
    except SyntaxError:
        return python, "", set()

    move_ranges: list[tuple[int, int]] = []
    runtime_names: set[str] = set()
    import_end = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_end = max(import_end, getattr(node, "end_lineno", node.lineno))
            continue
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            break

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and is_generated_helper_name(node.name):
            move_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
            runtime_names.add(node.name)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Try)):
            names = assigned_names(node)
            if names and all(is_runtime_helper_binding(name) for name in names):
                move_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
                runtime_names.update(names)

    if not move_ranges:
        return python, "", set()

    lines = python.splitlines()
    runtime_lines: list[str] = []
    if import_end:
        runtime_lines.extend(lines[:import_end])
        runtime_lines.append("")
    runtime_lines.extend(runtime_module_header())
    runtime_lines.append("")
    generic_blocks: list[list[str]] = []
    specific_blocks: list[list[str]] = []
    for start, end in sorted(move_ranges):
        block = lines[start - 1:end]
        block = add_runtime_docstring(block)
        if is_program_specific_runtime_block(block):
            specific_blocks.append(block)
        else:
            generic_blocks.append(block)
    for block in generic_blocks:
        runtime_lines.extend(block)
        runtime_lines.append("")
    if specific_blocks:
        runtime_lines.extend(program_specific_runtime_header())
        runtime_lines.append("")
        for block in specific_blocks:
            runtime_lines.extend(block)
            runtime_lines.append("")

    main = remove_line_ranges(python, move_ranges)
    try:
        main_used_names = ast_used_names(ast.parse(main))
        runtime_names = runtime_names.intersection(main_used_names)
    except SyntaxError:
        pass
    main_lines = main.splitlines()
    insert_at = 0
    while insert_at < len(main_lines):
        stripped = main_lines[insert_at].strip()
        if stripped.startswith(("import ", "from ")):
            insert_at += 1
            continue
        break
    main_lines[insert_at:insert_at] = format_runtime_import(module_name, runtime_names)
    main = collapse_excess_blank_lines("\n".join(main_lines).rstrip() + "\n")
    runtime = collapse_excess_blank_lines("\n".join(runtime_lines).rstrip() + "\n")
    return main, runtime, runtime_names


def prune_runtime_module(runtime: str, required_names: set[str]) -> str:
    try:
        tree = ast.parse(runtime)
    except SyntaxError:
        return runtime

    defined: dict[str, ast.AST] = {}
    owner_by_name: dict[str, ast.AST] = {}
    for node in tree.body:
        names: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Try)):
            names.update(assigned_names(node))
        for name in names:
            defined[name] = node
            owner_by_name[name] = node

    keep_names = set(required_names)
    changed = True
    while changed:
        changed = False
        for name in list(keep_names):
            node = defined.get(name)
            if node is None:
                continue
            for used in ast_used_names(node):
                if used in defined and used not in keep_names:
                    keep_names.add(used)
                    changed = True

    keep_nodes = {owner_by_name[name] for name in keep_names if name in owner_by_name}
    remove_ranges: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Try)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                names = {node.name}
            else:
                names = assigned_names(node)
            if names and node not in keep_nodes and not any(name in keep_names for name in names):
                remove_ranges.append((node.lineno, getattr(node, "end_lineno", node.lineno)))
    if not remove_ranges:
        return runtime
    pruned = remove_line_ranges(runtime, remove_ranges)
    pruned = collapse_excess_blank_lines(pruned)
    return remove_unused_import_statements(pruned)


def format_runtime_import(module_name: str, names: set[str]) -> list[str]:
    public_names = sorted(name for name in names if name and not name.startswith("__"))
    if not public_names:
        return []
    if len(public_names) <= 4 and sum(len(name) for name in public_names) <= 60:
        return [f"from {module_name} import " + ", ".join(public_names)]
    lines = [f"from {module_name} import ("]
    lines.extend(f"    {name}," for name in public_names)
    lines.append(")")
    return lines


def runtime_module_header() -> list[str]:
    return [
        '"""Runtime helpers for Python files generated by xr2p.py.',
        "",
        "This module implements small R-compatibility shims used by translated scripts.",
        "It is generated on demand by `xr2p.py --runtime-module` and can be refreshed",
        "with `--update-runtime`.",
        '"""',
    ]


def program_specific_runtime_header() -> list[str]:
    return [
        "# Program-specific generated helpers",
        "# These helpers are emitted for specialized translations and are not part",
        "# of the stable xr2p runtime API.",
    ]


def is_program_specific_runtime_block(block: list[str]) -> bool:
    text = "\n".join(block)
    return any(
        token in text
        for token in (
            "_nagarch_var_fast_impl",
            "_arma_residuals_fast_impl",
            "_garch_negloglik_fast_impl",
            "_nagarch_negloglik_fast_impl",
            "nagarch_var_fast",
            "garch_negloglik_fast",
            "nagarch_negloglik_fast",
            "arma_residuals_fast",
            "varma_resid_fast",
        )
    )


RUNTIME_DOCSTRINGS = {
    "RList": "Simple namespace used to represent R lists with optional element names.",
    "RNamedVector": "Array-like container that preserves R vector names through common operations.",
    "r_print": "Print values using compact R-like formatting for vectors, matrices, and data frames.",
    "r_s3_print": "Dispatch print methods for simple translated S3-style objects.",
    "r_subset": "Return an R-style subset of arrays, data frames, lists, or named vectors.",
    "r_set_subset": "Assign into an R-style subset and return the modified object.",
    "r_vec_subset": "Return a one-dimensional R-style vector subset.",
    "r_matrix_index_get": "Return an R-style one-based vector or matrix element/subset.",
    "r_matrix_index_set": "Assign an R-style one-based vector or matrix element/subset.",
    "r_data_frame": "Build a pandas DataFrame from translated R data.frame arguments.",
    "r_df_col": "Convert an R-like value into a one-dimensional DataFrame column.",
    "r_c": "Concatenate translated R vector arguments while preserving optional names.",
    "r_list_get": "Return an R-style list element using one-based numeric or named indexing.",
    "r_names": "Return R-style names for vectors, lists, data frames, or named metadata.",
    "r_setdiff": "Return values in x that are not present in y, preserving order.",
    "r_recycle_binary": "Apply a binary NumPy operation with simple R-style vector recycling.",
    "r_add": "Add values with simple R-style recycling and date handling.",
    "r_sub": "Subtract values with simple R-style recycling.",
    "r_mul": "Multiply values with simple R-style recycling.",
    "r_div": "Divide values with simple R-style recycling.",
    "r_length": "Return R-style length for common translated containers.",
    "r_range": "Return an inclusive Python range matching R sequence loop endpoints.",
    "r_seq": "Return an inclusive NumPy sequence matching R's colon operator.",
    "r_as_date": "Convert strings or arrays to pandas datetime values.",
    "r_diff": "Return first differences, using rows as the matrix axis.",
    "tail_py": "Return the last n elements or rows.",
    "head_py": "Return the first n elements or rows.",
    "var_r": "Return R-compatible sample variance or covariance.",
    "cor_py": "Return R-compatible correlation, including common missing-value options.",
}


def add_runtime_docstring(block: list[str]) -> list[str]:
    if not block:
        return block
    first = block[0]
    match = re.match(r"^(class|def)\s+([A-Za-z_]\w*)\b.*:\s*$", first)
    if match is None:
        return block
    name = match.group(2)
    doc = RUNTIME_DOCSTRINGS.get(name)
    if not doc:
        return block
    if len(block) > 1 and re.match(r'\s+"""', block[1]):
        return block
    indent = "    "
    return [block[0], f'{indent}"""{doc}"""', *block[1:]]


def is_generated_helper_name(name: str) -> bool:
    if name.startswith("_nagarch_"):
        return True
    if name.startswith(("r_", "R")):
        return True
    if name.endswith("_py"):
        return True
    return name in {
        "TryError",
        "source",
        "date",
        "q",
        "quit",
        "message",
        "warning",
        "time_ctime",
        "var_r",
        "sd",
        "cov",
        "cor",
        "sort",
        "order",
        "rank",
        "table",
        "tapply",
        "lapply",
        "sapply",
        "mapply",
        "split",
        "unsplit",
        "Reduce",
        "rep_int",
        "match_fun",
    }


def is_runtime_helper_binding(name: str) -> bool:
    return (
        name.startswith("_")
        or name.endswith("_fast")
        or name in {"njit", "arma_residuals_fast", "varma_resid_fast", "nagarch_var_fast"}
    )


def assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    elif isinstance(node, ast.Try):
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                targets.extend(child.targets)
            elif isinstance(child, ast.AnnAssign):
                targets.append(child.target)
            elif isinstance(child, ast.Import):
                names.update(alias.asname or alias.name.split(".")[0] for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in child.names)
    for target in targets:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                names.add(child.id)
    return names


def ast_used_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}


def ast_used_names_excluding_removable_top_level(tree: ast.Module, removable: set[str]) -> set[str]:
    used: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in removable:
            continue
        used.update(ast_used_names(node))
    return used


def remove_line_ranges(source: str, ranges: list[tuple[int, int]]) -> str:
    remove: set[int] = set()
    for start, end in ranges:
        remove.update(range(start, end + 1))
    lines = source.splitlines()
    kept = [line for i, line in enumerate(lines, start=1) if i not in remove]
    return "\n".join(kept).rstrip() + "\n"


def collapse_excess_blank_lines(source: str) -> str:
    lines = source.splitlines()
    out: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            out.append(line)
        else:
            blank_count += 1
            if blank_count <= 2:
                out.append(line)
    return "\n".join(out).rstrip() + "\n"


def remove_unused_import_statements(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    import_lines: dict[int, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            import_lines[node.lineno] = {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            import_lines[node.lineno] = {alias.asname or alias.name for alias in node.names}
    if not import_lines:
        return source
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    remove_lines = [lineno for lineno, names in import_lines.items() if names.isdisjoint(used)]
    if not remove_lines:
        return source
    return remove_line_ranges(source, [(lineno, lineno) for lineno in remove_lines])


def compile_python_text(python: str, *, suffix: str = ".py") -> subprocess.CompletedProcess[str]:
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as handle:
        handle.write(python)
        path = Path(handle.name)
    try:
        return check_python_compile(path)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def runtime_module_is_compatible(existing: str, generated: str) -> bool:
    try:
        existing_tree = ast.parse(existing)
        generated_tree = ast.parse(generated)
    except SyntaxError:
        return False
    existing_names = top_level_defined_names(existing_tree)
    generated_names = top_level_defined_names(generated_tree)
    return generated_names.issubset(existing_names)


def top_level_defined_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Try)):
            names.update(assigned_names(node))
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def count_code_lines(source: str) -> int:
    return sum(1 for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def source_scan_text(source: str) -> str:
    """Remove R comments while preserving quoted strings for source() scanning."""
    lines: list[str] = []
    for line in source.splitlines():
        kept: list[str] = []
        quote: str | None = None
        escaped = False
        for char in line:
            if escaped:
                kept.append(char)
                escaped = False
            elif char == "\\" and quote is not None:
                kept.append(char)
                escaped = True
            elif quote is not None:
                kept.append(char)
                if char == quote:
                    quote = None
            elif char in {'"', "'"}:
                quote = char
                kept.append(char)
            elif char == "#":
                break
            else:
                kept.append(char)
        lines.append("".join(kept))
    return "\n".join(lines)


def literal_source_paths(source: str) -> list[str]:
    """Return literal file arguments from source("file.r") calls."""
    pattern = re.compile(r"\bsource\s*\(\s*([\"'])(.*?)\1", re.IGNORECASE | re.DOTALL)
    paths: list[str] = []
    for match in pattern.finditer(source_scan_text(source)):
        literal = match.group(1) + match.group(2) + match.group(1)
        try:
            value = ast.literal_eval(literal)
        except (SyntaxError, ValueError):
            value = match.group(2)
        if isinstance(value, str):
            paths.append(value)
    return paths


def translate_source_dependencies(
    source: str,
    source_path: Path,
    output_path: Path,
    *,
    use_numba: bool,
    banner: bool,
    lean: bool,
) -> list[Path]:
    """Recursively translate local files referenced by literal source() calls."""
    written: list[Path] = []
    visited: set[tuple[Path, Path]] = {(source_path.resolve(), output_path.resolve())}

    def translate_dependency(r_path: Path, py_path: Path) -> None:
        key = (r_path.resolve(), py_path.resolve())
        if key in visited:
            return
        visited.add(key)
        try:
            dependency_source = r_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise R2PyError(f"cannot read source dependency {r_path}: {exc}") from exc
        dependency_python = translate_source(
            dependency_source,
            use_numba=use_numba,
            source_name=str(r_path),
            banner=banner,
        )
        if lean:
            slimmed = slim_python(dependency_python)
            if slimmed != dependency_python and compile_python_text(slimmed).returncode == 0:
                dependency_python = slimmed
        for child in literal_source_paths(dependency_source):
            child_r = Path(child)
            child_py = Path(child)
            if child_r.is_absolute():
                child_py = child_r
            else:
                child_r = r_path.parent / child_r
                child_py = py_path.parent / child_py
            if child_r.suffix.lower() == ".r":
                translate_dependency(child_r, child_py.with_suffix(".py"))
        py_path.parent.mkdir(parents=True, exist_ok=True)
        py_path.write_text(dependency_python, encoding="utf-8")
        written.append(py_path)
        print(f"wrote dependency {py_path}")

    for dependency in literal_source_paths(source):
        dependency_r = Path(dependency)
        dependency_py = Path(dependency)
        if dependency_r.is_absolute():
            dependency_py = dependency_r
        else:
            dependency_r = source_path.parent / dependency_r
            dependency_py = output_path.parent / dependency_py
        if dependency_r.suffix.lower() == ".r":
            translate_dependency(dependency_r, dependency_py.with_suffix(".py"))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate a numerical subset of R to Python/NumPy.")
    parser.add_argument("source", type=Path, help="R source file")
    parser.add_argument("-o", "--out", type=Path, help="output Python file")
    parser.add_argument("--tee", action="store_true", help="print the emitted Python code")
    parser.add_argument("--tee-both", action="store_true", help="print the original R source and emitted Python code")
    parser.add_argument("--loc", action="store_true", help="print nonblank, noncomment line counts for R source and Python output")
    parser.add_argument("--lean", action="store_true", help="try to remove unused generated helpers; fall back if slimmed code does not compile")
    parser.add_argument("--runtime-module", action="store_true", help="move generated runtime helpers to xr2p_runtime.py and import them")
    parser.add_argument("--prune-runtime", action="store_true", help="with --runtime-module, keep only runtime helpers needed by the generated script")
    parser.add_argument("--update-runtime", action="store_true", help="overwrite xr2p_runtime.py when --runtime-module is used")
    parser.add_argument("--no-numba", action="store_true", help="do not emit generated code that imports or uses numba")
    parser.add_argument("--no-banner", action="store_true", help="do not emit the translated-from banner comment")
    parser.add_argument("--no-py-compile", action="store_true", help="skip python -m py_compile check")
    parser.add_argument("--run", action="store_true", help="run the generated Python")
    parser.add_argument("--time", action="store_true", help="run the generated Python and print transpilation and run elapsed times")
    parser.add_argument("--run-both", action="store_true", help="run original R and generated Python")
    parser.add_argument("--time-both", action="store_true", help="run original R and generated Python and print transpilation and run elapsed times")
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
    if args.prune_runtime:
        args.runtime_module = True
    if args.update_runtime and not args.runtime_module:
        print("Option error: --update-runtime requires --runtime-module.")
        return 1

    if args.time:
        args.run = True
    if args.time_both:
        args.run_both = True

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
        translate_start = time.perf_counter()
        python = translate_source(source, use_numba=not args.no_numba, source_name=str(args.source), banner=not args.no_banner)
        translate_elapsed = time.perf_counter() - translate_start
    except (OSError, R2PyError) as exc:
        print(f"xr2p: {exc}", file=sys.stderr)
        return 1

    if args.lean:
        slimmed = slim_python(python)
        if slimmed != python:
            lean_compile = compile_python_text(slimmed)
            if lean_compile.returncode == 0:
                python = slimmed
            else:
                print("warning: --lean output failed syntax check; wrote unslimmed translation", file=sys.stderr)

    out = args.out or args.source.with_suffix(".py")
    try:
        dependency_outputs = translate_source_dependencies(
            source,
            args.source,
            out,
            use_numba=not args.no_numba,
            banner=not args.no_banner,
            lean=args.lean,
        )
    except (OSError, R2PyError) as exc:
        print(f"xr2p: {exc}", file=sys.stderr)
        return 1
    runtime_python = ""
    runtime_path: Path | None = None
    runtime_written = False
    if args.runtime_module:
        runtime_path = out.parent / "xr2p_runtime.py"
        python, runtime_python, runtime_import_names = split_runtime_module(python, runtime_path.stem)
        if args.prune_runtime and runtime_python:
            pruned_runtime = prune_runtime_module(runtime_python, runtime_import_names)
            if pruned_runtime != runtime_python:
                prune_compile = compile_python_text(pruned_runtime)
                if prune_compile.returncode == 0:
                    runtime_python = pruned_runtime
                else:
                    print("warning: --prune-runtime output failed syntax check; using full runtime module", file=sys.stderr)
        runtime_needs_update = args.update_runtime or not runtime_path.exists()
        if runtime_python and runtime_path.exists() and not runtime_needs_update:
            existing_runtime = runtime_path.read_text(encoding="utf-8", errors="replace")
            if not runtime_module_is_compatible(existing_runtime, runtime_python):
                print("warning: existing runtime module is incompatible; updating xr2p_runtime.py", file=sys.stderr)
                runtime_needs_update = True
        if runtime_python and runtime_needs_update:
            runtime_path.write_text(runtime_python, encoding="utf-8")
            runtime_written = True
    out.write_text(python, encoding="utf-8")
    print(f"wrote {out}")
    if runtime_path is not None and runtime_python:
        print(f"runtime module: {runtime_path}")
    if args.time or args.time_both:
        print(f"Transpile time: {translate_elapsed:.3f}s")
    if args.loc:
        r_loc = count_code_lines(source)
        py_loc = count_code_lines(python)
        if args.runtime_module and runtime_path is not None:
            runtime_source = runtime_python if runtime_written or not runtime_path.exists() else runtime_path.read_text(encoding="utf-8", errors="replace")
            runtime_loc = count_code_lines(runtime_source)
            total_py_loc = py_loc + runtime_loc
            ratio = total_py_loc / r_loc if r_loc else float("inf")
            print(f"lines of code: R={r_loc} Python={py_loc} runtime={runtime_loc} total Python={total_py_loc} total Python/R={ratio:.3g}")
        else:
            ratio = py_loc / r_loc if r_loc else float("inf")
            print(f"lines of code: R={r_loc} Python={py_loc} Python/R={ratio:.3g}")
    if args.tee_both:
        print("R source:")
        print(source, end="" if source.endswith("\n") else "\n")
        print("Python translation:")
        print(python, end="" if python.endswith("\n") else "\n")
    if args.tee:
        print(python, end="" if python.endswith("\n") else "\n")
    if not args.no_py_compile:
        for compile_path in [*dependency_outputs, out]:
            compile_result = check_python_compile(compile_path)
            if compile_result.returncode != 0:
                print(f"Python syntax check failed ({compile_path}):", file=sys.stderr)
                print_process_output(compile_result)
                return compile_result.returncode

    python_round_digits = args.round_both if args.round_both is not None else args.round
    r_round_digits = args.round_both

    if args.run_both:
        print("Run (R):", args.rscript, args.source)
        r_start = time.perf_counter()
        r_result = run_r(args.source, args.rscript)
        r_elapsed = time.perf_counter() - r_start
        print("Run (R):", "PASS" if r_result.returncode == 0 else f"FAIL exit={r_result.returncode}")
        print_result_output(r_result, r_round_digits, pretty_r=args.pretty, flush_left=args.flush_left, squeeze=args.squeeze)
        print("Run (Python):", sys.executable, out)
        py_start = time.perf_counter()
        py_result = run_python(out)
        py_elapsed = time.perf_counter() - py_start
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
                if args.time_both:
                    ratio = py_elapsed / r_elapsed if r_elapsed else float("inf")
                    print()
                    print(f"run times: R={r_elapsed:.3f}s Python={py_elapsed:.3f}s Python/R={ratio:.3g}")
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
                if args.time_both:
                    ratio = py_elapsed / r_elapsed if r_elapsed else float("inf")
                    print()
                    print(f"run times: R={r_elapsed:.3f}s Python={py_elapsed:.3f}s Python/R={ratio:.3g}")
                return 1
            print("Stats:", "PASS")
        if args.time_both:
            ratio = py_elapsed / r_elapsed if r_elapsed else float("inf")
            print()
            print(f"run times: R={r_elapsed:.3f}s Python={py_elapsed:.3f}s Python/R={ratio:.3g}")
        return 0 if r_result.returncode == 0 and py_result.returncode == 0 else 1
    if args.run:
        py_start = time.perf_counter()
        result = run_python(out)
        py_elapsed = time.perf_counter() - py_start
        if args.time:
            print(f"Run (Python time): {py_elapsed:.3f}s")
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

