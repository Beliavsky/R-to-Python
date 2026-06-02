#!/usr/bin/env python3
"""Batch runner for xr2p.py over R files.

This is deliberately small: it is a coverage-measurement helper for the
translator, not a full build system.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

from xr2p import R2PyError, run_python, translate_source


XR2F_PYTEST_CASES = [
    "xarray.r",
    "xbare.r",
    "xc.r",
    "xfunc.r",
    "xhello.r",
    "xlist.r",
    "xlist_core.r",
    "xlm.r",
    "xloop.r",
    "xmatrix.r",
    "xna.r",
    "xnumeric.r",
    "xouter.r",
    "xpaste.r",
    "xr2f_smoke.R",
    "xreg_fit.r",
    "xrunif.r",
    "xseq.r",
    "xt.r",
    "xtf.r",
]


def expand_inputs(items: list[str], *, include_xr2f_pytest_corpus: bool = False, recursive: bool = False) -> list[Path]:
    out: list[Path] = []
    if include_xr2f_pytest_corpus:
        root = Path(r"C:\python\R-to-Fortran")
        out.extend(root / name for name in XR2F_PYTEST_CASES)
    for item in items:
        if item.startswith("@"):
            list_path = Path(item[1:])
            for raw in list_path.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    out.extend(expand_inputs([line], recursive=recursive))
            continue
        path = Path(item)
        if path.is_dir():
            pattern = "**/*.r" if recursive else "*.r"
            out.extend(sorted(path.glob(pattern)))
            pattern_upper = "**/*.R" if recursive else "*.R"
            out.extend(sorted(path.glob(pattern_upper)))
            continue
        matches = sorted(glob.glob(item, recursive=recursive))
        out.extend(Path(m) for m in matches) if matches else out.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        key = path.resolve() if path.exists() else path
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def output_path_for(source: Path, *, out_dir: Path, roots: list[Path]) -> Path:
    rel = None
    source_resolved = source.resolve()
    for root in roots:
        try:
            rel = source_resolved.relative_to(root.resolve())
            break
        except ValueError:
            continue
    if rel is None:
        rel = Path(source.name)
    return (out_dir / rel).with_suffix(".py")


def input_roots(items: list[str]) -> list[Path]:
    roots = []
    for item in items:
        if item.startswith("@"):
            continue
        path = Path(item)
        if path.is_dir():
            roots.append(path)
    return roots


def format_failure(exc: Exception, *, source: Path, out_path: Path, python: str | None) -> str:
    if isinstance(exc, SyntaxError):
        line_no = exc.lineno or 0
        detail = [f"syntax error in {out_path}, line {line_no}: {exc.msg}"]
        if exc.text:
            text = exc.text.rstrip("\n")
            detail.append(text)
            if exc.offset:
                detail.append(" " * max(exc.offset - 1, 0) + "^")
        if python and line_no:
            lines = python.splitlines()
            start = max(line_no - 3, 1)
            stop = min(line_no + 2, len(lines))
            detail.append("generated context:")
            for number in range(start, stop + 1):
                marker = ">" if number == line_no else " "
                detail.append(f"{marker} {number:5d}: {lines[number - 1]}")
        detail.append(f"source: {source}")
        return "\n".join(detail)
    return str(exc)


def main(argv: list[str] | None = None) -> int:
    started_at = perf_counter()
    parser = argparse.ArgumentParser(description="Run xr2p.py over many R files.")
    parser.add_argument("inputs", nargs="*", help="R files, globs, or @lists")
    parser.add_argument("--xr2f-pytest-corpus", action="store_true", help=r"include C:\python\R-to-Fortran pytest examples")
    parser.add_argument("--recursive", "-r", action="store_true", help="recursively expand directory inputs and ** globs")
    parser.add_argument("--limit", type=int, metavar="N", help="process at most the first N expanded inputs")
    parser.add_argument("--out-dir", type=Path, help="write generated Python files under this directory")
    parser.add_argument("--summary-csv", type=Path, help="write pass/fail summary as CSV")
    parser.add_argument("--check-syntax", action="store_true", help="parse generated Python with ast.parse")
    parser.add_argument("--run", action="store_true", help="run generated Python")
    parser.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs, include_xr2f_pytest_corpus=args.xr2f_pytest_corpus, recursive=args.recursive)
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit requires a nonnegative integer")
        paths = paths[: args.limit]
    if not paths:
        parser.error("no inputs specified")
    n_ok = 0
    failures: list[tuple[Path, str]] = []
    rows: list[dict[str, str]] = []
    roots = input_roots(args.inputs)
    with tempfile.TemporaryDirectory(prefix="xr2p_batch_") as td:
        temp = args.out_dir or Path(td)
        temp.mkdir(parents=True, exist_ok=True)
        for path in paths:
            out_path = output_path_for(path, out_dir=temp, roots=roots)
            python: str | None = None
            try:
                source = path.read_text(encoding="utf-8-sig")
                python = translate_source(source)
                if args.out_dir:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(python, encoding="utf-8")
                if args.check_syntax:
                    ast.parse(python, filename=str(path.with_suffix(".py")))
                if args.run:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(python, encoding="utf-8")
                    result = run_python(out_path)
                    if result.returncode != 0:
                        raise R2PyError((result.stderr or result.stdout or f"exit code {result.returncode}").strip())
            except Exception as exc:
                detail = format_failure(exc, source=path, out_path=out_path, python=python)
                failures.append((path, detail))
                rows.append({"status": "FAIL", "source": str(path), "output": str(out_path), "error": detail})
                print(f"FAIL {path}: {detail}")
            else:
                n_ok += 1
                rows.append({"status": "PASS", "source": str(path), "output": str(out_path), "error": ""})
                if not args.quiet:
                    print(f"PASS {path}")

    print(f"summary: {n_ok} passed, {len(failures)} failed, {len(paths)} total")
    print(f"finished: {datetime.now().isoformat(timespec='seconds')}")
    print(f"elapsed: {perf_counter() - started_at:.2f} seconds")
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["status", "source", "output", "error"])
            writer.writeheader()
            writer.writerows(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
