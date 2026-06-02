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
import re
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


def r_pass_sources(summary_csv: Path) -> list[Path]:
    with summary_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "status" not in (reader.fieldnames or []) or "source" not in (reader.fieldnames or []):
            raise ValueError("--only-r-pass CSV must have status and source columns")
        return [Path(row["source"]) for row in reader if row.get("status", "").upper() == "PASS" and row.get("source")]


def r_output_sources(summary_csv: Path) -> list[Path]:
    with summary_csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "status" not in fieldnames or "source" not in fieldnames or "stdout_bytes" not in fieldnames:
            raise ValueError("--only-r-output CSV must have status, source, and stdout_bytes columns")
        out: list[Path] = []
        for row in reader:
            if row.get("status", "").upper() != "PASS" or not row.get("source"):
                continue
            if row.get("uses_graphics", "0") in {"1", "true", "TRUE", "True", "yes", "YES", "Yes"}:
                continue
            try:
                stdout_bytes = int(row.get("stdout_bytes", "0"))
            except ValueError:
                stdout_bytes = 0
            if stdout_bytes > 0:
                out.append(Path(row["source"]))
        return out


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


def source_warnings(source: str) -> str:
    warnings: list[str] = []
    func_re = re.compile(r"([A-Za-z]\w*)\s*(?:<-|=)\s*function\s*\(([^)]*)\)", re.IGNORECASE)
    for match in func_re.finditer(source):
        name = match.group(1)
        params = []
        for raw_param in match.group(2).split(","):
            param = raw_param.strip().split("=", 1)[0].strip()
            if re.fullmatch(r"[a-z]\w*", param):
                params.append(param)
        if not params:
            continue
        body = source[match.end() :]
        next_func = func_re.search(body)
        if next_func is not None:
            body = body[: next_func.start()]
        for param in params:
            upper = param.upper()
            if upper != param and re.search(rf"\b{re.escape(upper)}\b", body):
                warnings.append(f"{name}: references global-looking {upper} while parameter is {param}")
    return "; ".join(dict.fromkeys(warnings))


def extract_source_dependencies(source: str, source_path: Path) -> list[Path]:
    source_root = source_path
    deps: list[Path] = []
    marker = "/r_src/"
    # Basic extraction: source("file.R") or source('file.R')
    source_re = re.compile(r"\bsource\s*\(\s*([\"'])(.*?)\1", re.IGNORECASE | re.MULTILINE)
    for match in source_re.finditer(source):
        raw = match.group(2).strip()
        if not raw:
            continue
        raw_path = raw.replace("\\", "/")
        candidate: Path | None = None
        if marker in raw_path:
            tail = raw_path.split(marker, 1)[1]
            root = source_root.parent
            while root.name.lower() != "r_src" and root != root.parent:
                root = root.parent
            if root.name.lower() == "r_src":
                candidate = root / tail
            else:
                candidate = Path(raw_path)
        else:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = source_root.parent / candidate
        if candidate is None:
            continue
        if candidate.suffix == "":
            for ext in [".R", ".r"]:
                candidate_with_ext = candidate.with_suffix(ext)
                if candidate_with_ext.exists():
                    deps.append(candidate_with_ext.resolve())
                    break
        else:
            deps.append(candidate.resolve())
    # dedupe, keep order
    uniq: list[Path] = []
    seen: set[Path] = set()
    for dep in deps:
        if dep not in seen and dep.exists():
            seen.add(dep)
            uniq.append(dep)
    return uniq


def expand_source_dependencies(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    queue: list[Path] = []
    for path in paths:
        if path.exists():
            queue.append(path.resolve())
        else:
            queue.append(path)
    while queue:
        path = queue.pop(0)
        if path in seen:
            continue
        if not path.exists():
            continue
        seen.add(path)
        expanded.append(path)
        try:
            source = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        for dep in extract_source_dependencies(source, path):
            if dep not in seen:
                queue.append(dep)
    return expanded


def main(argv: list[str] | None = None) -> int:
    started_at = perf_counter()
    parser = argparse.ArgumentParser(description="Run xr2p.py over many R files.")
    parser.add_argument("inputs", nargs="*", help="R files, globs, or @lists")
    parser.add_argument("--xr2f-pytest-corpus", action="store_true", help=r"include C:\python\R-to-Fortran pytest examples")
    parser.add_argument("--recursive", "-r", action="store_true", help="recursively expand directory inputs and ** globs")
    parser.add_argument("--limit", type=int, metavar="N", help="process at most the first N expanded inputs")
    parser.add_argument("--skip", type=int, metavar="N", help="skip the first N expanded inputs after filtering")
    parser.add_argument("--max-fail", type=int, metavar="N", help="stop after N failures")
    parser.add_argument("--out-dir", type=Path, help="write generated Python files under this directory")
    parser.add_argument("--summary-csv", type=Path, help="write pass/fail summary as CSV")
    parser.add_argument("--only-r-pass", type=Path, metavar="CSV", help="only process sources marked PASS in an xrbatch.py summary CSV")
    parser.add_argument("--only-r-output", type=Path, metavar="CSV", help="only process sources marked PASS with stdout_bytes > 0 in an xrbatch.py summary CSV")
    parser.add_argument("--check-syntax", action="store_true", help="parse generated Python with ast.parse")
    parser.add_argument("--run", action="store_true", help="run generated Python")
    parser.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs, include_xr2f_pytest_corpus=args.xr2f_pytest_corpus, recursive=args.recursive)
    dependency_paths: list[Path] = []
    if args.only_r_output:
        passed = r_output_sources(args.only_r_output)
        dependency_paths = r_pass_sources(args.only_r_output)
        if paths:
            allowed = {path.resolve() if path.exists() else path for path in passed}
            paths = [path for path in paths if (path.resolve() if path.exists() else path) in allowed]
        else:
            paths = passed
    elif args.only_r_pass:
        passed = r_pass_sources(args.only_r_pass)
        if paths:
            allowed = {path.resolve() if path.exists() else path for path in passed}
            paths = [path for path in paths if (path.resolve() if path.exists() else path) in allowed]
        else:
            paths = passed
    selected_total = len(paths)
    skipped = 0
    if args.skip is not None:
        if args.skip < 0:
            parser.error("--skip requires a nonnegative integer")
        skipped += min(args.skip, len(paths))
        paths = paths[args.skip :]
    if args.limit is not None:
        if args.limit < 0:
            parser.error("--limit requires a nonnegative integer")
        if len(paths) > args.limit:
            skipped += len(paths) - args.limit
        paths = paths[: args.limit]
    if args.max_fail is not None and args.max_fail < 0:
        parser.error("--max-fail requires a nonnegative integer")
    if not paths:
        parser.error("no inputs specified")
    n_ok = 0
    n_fail_selected = 0
    failures: list[tuple[Path, str]] = []
    rows: list[dict[str, str]] = []
    stopped_early = False
    roots = input_roots(args.inputs)
    with tempfile.TemporaryDirectory(prefix="xr2p_batch_") as td:
        temp = args.out_dir or Path(td)
        temp.mkdir(parents=True, exist_ok=True)
        all_dependency_paths = expand_source_dependencies(list(dict.fromkeys(dependency_paths + paths)))
        for dep_path in all_dependency_paths:
            dep_out_path = output_path_for(dep_path, out_dir=temp, roots=roots)
            dep_python: str | None = None
            warnings = ""
            dep_started_at = perf_counter()
            try:
                dep_source = dep_path.read_text(encoding="utf-8-sig")
                warnings = source_warnings(dep_source)
                dep_python = translate_source(dep_source)
                ast.parse(dep_python, filename=str(dep_path.with_suffix(".py")))
                dep_out_path.parent.mkdir(parents=True, exist_ok=True)
                dep_out_path.write_text(dep_python, encoding="utf-8")
            except Exception as exc:
                elapsed_sec = perf_counter() - dep_started_at
                detail = format_failure(exc, source=dep_path, out_path=dep_out_path, python=dep_python)
                failures.append((dep_path, detail))
                rows.append({"status": "FAIL", "source": str(dep_path), "output": str(dep_out_path), "elapsed_sec": f"{elapsed_sec:.6f}", "warnings": warnings, "error": detail})
                print(f"FAIL {dep_path}: {detail}")
                if args.max_fail is not None and len(failures) >= args.max_fail:
                    stopped_early = True
                    skipped += len(paths)
                    break
        if stopped_early:
            paths = []
        for path in paths:
            out_path = output_path_for(path, out_dir=temp, roots=roots)
            python: str | None = None
            warnings = ""
            file_started_at = perf_counter()
            try:
                source = path.read_text(encoding="utf-8-sig")
                warnings = source_warnings(source)
                python = translate_source(source)
                if args.out_dir:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(python, encoding="utf-8")
                if args.check_syntax:
                    ast.parse(python, filename=str(path.with_suffix(".py")))
                if args.run:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(python, encoding="utf-8")
                    result = run_python(out_path, cwd=path.parent)
                    if result.returncode != 0:
                        raise R2PyError((result.stderr or result.stdout or f"exit code {result.returncode}").strip())
            except Exception as exc:
                elapsed_sec = perf_counter() - file_started_at
                detail = format_failure(exc, source=path, out_path=out_path, python=python)
                n_fail_selected += 1
                failures.append((path, detail))
                rows.append({"status": "FAIL", "source": str(path), "output": str(out_path), "elapsed_sec": f"{elapsed_sec:.6f}", "warnings": warnings, "error": detail})
                print(f"FAIL {path}: {detail}")
                if args.max_fail is not None and len(failures) >= args.max_fail:
                    stopped_early = True
                    skipped += len(paths) - (n_ok + n_fail_selected)
                    break
            else:
                elapsed_sec = perf_counter() - file_started_at
                n_ok += 1
                rows.append({"status": "PASS", "source": str(path), "output": str(out_path), "elapsed_sec": f"{elapsed_sec:.6f}", "warnings": warnings, "error": ""})
                if warnings and not args.quiet:
                    print(f"WARN {path}: {warnings}")
                if not args.quiet:
                    print(f"PASS {path}")

    print(f"summary: {n_ok} passed, {len(failures)} failed, {skipped} skipped, {selected_total} total")
    if stopped_early:
        print(f"stopped: reached --max-fail {args.max_fail}")
    print(f"finished: {datetime.now().isoformat(timespec='seconds')}")
    print(f"elapsed: {perf_counter() - started_at:.2f} seconds")
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["status", "source", "output", "elapsed_sec", "warnings", "error"])
            writer.writeheader()
            writer.writerows(rows)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
