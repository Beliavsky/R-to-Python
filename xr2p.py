#!/usr/bin/env python3
"""Experimental R-to-Python/NumPy transpiler for numerical scripts.

This is intentionally a small subset translator.  It is meant to cover simple
numeric R programs like the early xr2f.py test cases, not general R.
"""

from __future__ import annotations

import argparse
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
    for original in source.splitlines():
        line = strip_r_comment(original).strip()
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
    if "stats." in python:
        python = python.replace("import numpy as np\n", "import numpy as np\nfrom scipy import stats\n", 1)
    return python


def zero_base_unused_counter_loops(python: str) -> str:
    lines = python.splitlines()
    for i, line in enumerate(lines):
        match = re.match(r"^(\s*)for\s+([A-Za-z]\w*)\s+in\s+np\.arange\(1,\s*(.+?)\s*\+\s*1\):\s*$", line)
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


def translate_statement(line: str) -> list[str]:
    one_line_for = re.match(r"for\s*\((\w+)\s+in\s+(.+?)\)\s+(.+)$", line)
    if one_line_for:
        name, values, body = one_line_for.groups()
        return [f"for {name} in {translate_expr(values)}:", *[INDENT + part for part in translate_statement(body)]]

    for_match = re.match(r"for\s*\((\w+)\s+in\s+(.+?)\)\s*$", line)
    if for_match:
        name, values = for_match.groups()
        return [f"for {name} in {translate_expr(values)}:"]

    if_match = re.match(r"if\s*\((.+)\)\s*$", line)
    if if_match:
        return [f"if {translate_expr(if_match.group(1))}:"]

    while_match = re.match(r"while\s*\((.+)\)\s*$", line)
    if while_match:
        return [f"while {translate_expr(while_match.group(1))}:"]

    full_call = parse_full_call(line)
    if full_call is not None and full_call[0].lower() in {"print", "cat"}:
        name, args = full_call
        return [translate_call(name, args)]

    assign = split_assignment(line)
    if assign is not None:
        lhs, rhs = assign
        return [f"{lhs} = {translate_expr(rhs)}"]

    return [translate_expr(line)]


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


def split_assignment(line: str) -> tuple[str, str] | None:
    for op in ("<-", "="):
        pos = find_top_level_operator(line, op)
        if pos >= 0:
            lhs = line[:pos].strip()
            rhs = line[pos + len(op) :].strip()
            if re.match(r"^[A-Za-z.]\w*$", lhs):
                return r_name(lhs), rhs
    return None


def translate_expr(expr: str) -> str:
    expr = expr.strip().rstrip(";")
    expr, strings = mask_string_literals(expr)
    expr = translate_expr_code(expr)
    expr = restore_string_literals(expr, strings)
    return expr


def translate_expr_code(expr: str) -> str:
    expr = expr.replace("<-", "=")
    expr = expr.replace("%%", "%")
    expr = expr.replace("%*%", "@")
    expr = replace_power(expr)
    expr = replace_r_constants(expr)
    expr = replace_ranges(expr)
    expr = replace_calls(expr)
    expr = replace_names(expr)
    return expr


def replace_r_constants(expr: str) -> str:
    replacements = {
        "TRUE": "True",
        "FALSE": "False",
        "NULL": "None",
        "NA": "np.nan",
        "NaN": "np.nan",
        "Inf": "np.inf",
    }
    for old, new in replacements.items():
        expr = re.sub(rf"\b{old}\b", new, expr)
    return expr


def replace_power(expr: str) -> str:
    return expr.replace("^", "**")


def replace_ranges(expr: str) -> str:
    pattern = re.compile(r"(?<![\w.])(\w+|\d+)\s*:\s*(\w+|\d+)(?![\w.])")

    def repl(match: re.Match[str]) -> str:
        start, stop = match.groups()
        return f"np.arange({r_name(start)}, {r_name(stop)} + 1)"

    return pattern.sub(repl, expr)


def replace_calls(expr: str) -> str:
    previous = None
    while previous != expr:
        previous = expr
        expr = replace_innermost_call(expr)
    return expr


def replace_innermost_call(expr: str) -> str:
    pattern = re.compile(r"(?<![\w.])([A-Za-z]\w*(?:\.\w+)*)\s*\(([^()]*)\)")
    for match in pattern.finditer(expr):
        name = match.group(1)
        if name.startswith(("np.", "stats.")):
            continue
        args = split_args(match.group(2))
        translated = translate_call(name, args)
        if translated != match.group(0):
            return expr[: match.start()] + translated + expr[match.end() :]
    return expr


def translate_call(name: str, args: list[str]) -> str:
    py_args = [translate_expr(arg) for arg in args]
    lname = name.lower()
    if name.startswith("np."):
        return name + "(" + ", ".join(py_args) + ")"
    if lname == "c":
        return "np.array([" + ", ".join(py_args) + "])"
    if lname == "matrix":
        return translate_matrix_call(args)
    if lname == "print":
        return "print(" + ", ".join(py_args) + ")"
    if lname == "cat":
        return "print(" + ", ".join(py_args) + ', end="")'
    if lname == "sqrt":
        return f"np.sqrt({py_args[0]})"
    if lname in {"log", "exp", "sin", "cos", "tan", "abs"}:
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname in {"sum", "mean", "min", "max", "var", "median"}:
        return f"np.{lname}(" + ", ".join(py_args) + ")"
    if lname == "sd":
        return "np.std(" + py_args[0] + ", ddof=1)"
    if lname == "length":
        return "len(" + py_args[0] + ")"
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
        return "np.repeat(" + ", ".join(py_args) + ")"
    if lname == "numeric":
        return "np.zeros(" + py_args[0] + ")"
    if lname == "integer":
        return "np.zeros(" + py_args[0] + ", dtype=int)"
    if lname == "set.seed":
        return "np.random.seed(" + py_args[0] + ")"
    if lname == "runif":
        return translate_runif_call(args)
    if lname == "rnorm":
        return translate_rnorm_call(args)
    if lname in {"dnorm", "pnorm", "qnorm"}:
        return translate_normal_dist_call(lname, args)
    if lname in {"dt", "pt", "qt"}:
        return translate_t_dist_call(lname, args)
    return r_name(name) + "(" + ", ".join(py_args) + ")"


def translate_matrix_call(args: list[str]) -> str:
    if not args:
        raise R2PyError("matrix requires at least one argument")
    data = translate_expr(args[0])
    nrow = keyword_arg(args, "nrow", default=args[1] if len(args) >= 2 else None)
    ncol = keyword_arg(args, "ncol", default=args[2] if len(args) >= 3 else None)
    if nrow is None and ncol is None:
        return f"np.array({data})"
    if nrow is None:
        return f"np.array({data}).reshape((-1, {translate_expr(ncol)}), order='F')"
    if ncol is None:
        return f"np.array({data}).reshape(({translate_expr(nrow)}, -1), order='F')"
    return f"np.resize(np.array({data}), {translate_expr(nrow)} * {translate_expr(ncol)}).reshape(({translate_expr(nrow)}, {translate_expr(ncol)}), order='F')"


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
    return f"np.random.normal({mean}, {sd}, size={n})"


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
        if pos >= 0 and arg[:pos].strip() == name:
            return arg[pos + 1 :].strip()
    return default


def replace_names(expr: str) -> str:
    return re.sub(r"(?<![\w.])([A-Za-z]\w*(?:\.\w+)*)\b", lambda m: r_name(m.group(1)), expr)


def r_name(name: str) -> str:
    constants = {"True", "False", "None", "np", "stats", "nan", "inf"}
    if name in constants or name.startswith("np.") or name.startswith("stats."):
        return name
    if name[0].isdigit():
        return name
    return name.replace(".", "_")


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


def run_python(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(path)], text=True, capture_output=True)


def run_r(path: Path, rscript: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([rscript, str(path)], text=True, capture_output=True)


def print_process_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate a numerical subset of R to Python/NumPy.")
    parser.add_argument("source", type=Path, help="R source file")
    parser.add_argument("-o", "--out", type=Path, help="output Python file")
    parser.add_argument("--tee", action="store_true", help="print the emitted Python code")
    parser.add_argument("--run", action="store_true", help="run the generated Python")
    parser.add_argument("--run-both", action="store_true", help="run original R and generated Python")
    parser.add_argument("--rscript", default="rscript", help="command used to run R scripts")
    args = parser.parse_args(argv)

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
    if args.run_both:
        print("Run (R):", args.rscript, args.source)
        r_result = run_r(args.source, args.rscript)
        print("Run (R):", "PASS" if r_result.returncode == 0 else f"FAIL exit={r_result.returncode}")
        print_process_output(r_result)
        print("Run (Python):", sys.executable, out)
        py_result = run_python(out)
        print("Run (Python):", "PASS" if py_result.returncode == 0 else f"FAIL exit={py_result.returncode}")
        print_process_output(py_result)
        return 0 if r_result.returncode == 0 and py_result.returncode == 0 else 1
    if args.run:
        result = run_python(out)
        print_process_output(result)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
