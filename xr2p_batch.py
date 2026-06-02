#!/usr/bin/env python3
"""Batch runner for xr2p.py over R files.

This is deliberately small: it is a coverage-measurement helper for the
translator, not a full build system.
"""

from __future__ import annotations

import argparse
import ast
import glob
import sys
import tempfile
from pathlib import Path

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


def expand_inputs(items: list[str], *, include_xr2f_pytest_corpus: bool = False) -> list[Path]:
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
                    out.extend(expand_inputs([line]))
            continue
        matches = sorted(glob.glob(item))
        out.extend(Path(m) for m in matches) if matches else out.append(Path(item))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run xr2p.py over many R files.")
    parser.add_argument("inputs", nargs="*", help="R files, globs, or @lists")
    parser.add_argument("--xr2f-pytest-corpus", action="store_true", help=r"include C:\python\R-to-Fortran pytest examples")
    parser.add_argument("--check-syntax", action="store_true", help="parse generated Python with ast.parse")
    parser.add_argument("--run", action="store_true", help="run generated Python")
    parser.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs, include_xr2f_pytest_corpus=args.xr2f_pytest_corpus)
    if not paths:
        parser.error("no inputs specified")
    n_ok = 0
    failures: list[tuple[Path, str]] = []
    with tempfile.TemporaryDirectory(prefix="xr2p_batch_") as td:
        temp = Path(td)
        for path in paths:
            try:
                source = path.read_text(encoding="utf-8-sig")
                python = translate_source(source)
                if args.check_syntax:
                    ast.parse(python, filename=str(path.with_suffix(".py")))
                if args.run:
                    out_path = temp / path.with_suffix(".py").name
                    out_path.write_text(python, encoding="utf-8")
                    result = run_python(out_path)
                    if result.returncode != 0:
                        raise R2PyError((result.stderr or result.stdout or f"exit code {result.returncode}").strip())
            except Exception as exc:
                failures.append((path, str(exc)))
                print(f"FAIL {path}: {exc}")
            else:
                n_ok += 1
                if not args.quiet:
                    print(f"PASS {path}")

    print(f"summary: {n_ok} passed, {len(failures)} failed, {len(paths)} total")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
