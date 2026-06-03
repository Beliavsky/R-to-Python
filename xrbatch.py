#!/usr/bin/env python3
"""Batch runner for original R scripts.

This is a preflight helper for xr2p.py.  It identifies which R examples run
successfully in the local R environment before spending time on translation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter


def expand_inputs(items: list[str], *, recursive: bool = False, pattern: str = "*.r") -> list[Path]:
    out: list[Path] = []
    patterns = [pattern]
    if pattern == "*.r":
        patterns.append("*.R")
    for item in items:
        if item.startswith("@"):
            list_path = Path(item[1:])
            for raw in list_path.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    out.extend(expand_inputs([line], recursive=recursive, pattern=pattern))
            continue
        path = Path(item)
        if path.is_dir():
            for pat in patterns:
                out.extend(sorted(path.glob(("**/" if recursive else "") + pat)))
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


def input_roots(items: list[str]) -> list[Path]:
    roots: list[Path] = []
    for item in items:
        if item.startswith("@"):
            continue
        path = Path(item)
        if path.is_dir():
            roots.append(path)
    return roots


def log_base_for(source: Path, *, log_dir: Path, roots: list[Path]) -> Path:
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
    return log_dir / rel


def write_logs(
    source: Path,
    result: subprocess.CompletedProcess[str] | None,
    *,
    log_dir: Path | None,
    roots: list[Path],
) -> tuple[str, str]:
    if log_dir is None or result is None:
        return "", ""
    base = log_base_for(source, log_dir=log_dir, roots=roots)
    stdout_path = base.with_suffix(base.suffix + ".stdout.txt")
    stderr_path = base.with_suffix(base.suffix + ".stderr.txt")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return str(stdout_path), str(stderr_path)


def output_sizes(result: subprocess.CompletedProcess[str] | None) -> tuple[int, int]:
    if result is None:
        return 0, 0
    return len((result.stdout or "").encode("utf-8")), len((result.stderr or "").encode("utf-8"))


GRAPHICS_RE = re.compile(
    r"(?<![\w.])(?:png|pdf|jpeg|jpg|tiff|bmp|svg|dev\.new|dev\.off|plot|lines|points|hist|barplot|contour|persp|image)\s*\(",
    re.IGNORECASE,
)


def uses_graphics(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8-sig")
    except OSError:
        return False
    return GRAPHICS_RE.search(source) is not None


def trim_output(text: str, *, lines: int = 20) -> str:
    text = text.strip()
    if not text:
        return ""
    parts = text.splitlines()
    if len(parts) <= lines:
        return "\n".join(parts)
    return "\n".join(parts[:lines] + [f"... ({len(parts) - lines} more lines)"])


def run_rscript(path: Path, *, rscript: str, timeout: float | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [rscript, str(path.name)],
        cwd=path.parent,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def failure_detail(path: Path, exc: BaseException | None, result: subprocess.CompletedProcess[str] | None) -> str:
    if exc is not None:
        if isinstance(exc, subprocess.TimeoutExpired):
            return f"timeout after {exc.timeout} seconds"
        return str(exc)
    if result is None:
        return "unknown failure"
    output = trim_output((result.stderr or result.stdout or "").strip())
    if output:
        return output
    return f"exit code {result.returncode}"


def main(argv: list[str] | None = None) -> int:
    started_at = perf_counter()
    parser = argparse.ArgumentParser(description="Run many original R scripts with Rscript.")
    parser.add_argument("inputs", nargs="*", help="R files, globs, directories, or @lists")
    parser.add_argument("--recursive", "-r", action="store_true", help="recursively expand directory inputs and ** globs")
    parser.add_argument("--limit", type=int, metavar="N", help="process at most the first N expanded inputs")
    parser.add_argument("--max-fail", type=int, metavar="N", help="stop after N failures")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds before one R script is stopped; default: 60")
    parser.add_argument("--pattern", default="*.r", help='directory input glob pattern; default: "*.r" plus "*.R"')
    parser.add_argument("--rscript", default="rscript", help='Rscript executable; default: "rscript"')
    parser.add_argument("--summary-csv", type=Path, help="write pass/fail summary as CSV")
    parser.add_argument("--log-dir", type=Path, help="write stdout/stderr logs for each R script under this directory")
    parser.add_argument("--quiet", action="store_true", help="only print failures and the summary")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 0:
        parser.error("--limit requires a nonnegative integer")
    if args.max_fail is not None and args.max_fail < 0:
        parser.error("--max-fail requires a nonnegative integer")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout requires a positive number")

    paths = expand_inputs(args.inputs, recursive=args.recursive, pattern=args.pattern)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        parser.error("no inputs specified")

    n_ok = 0
    failures: list[tuple[Path, str]] = []
    rows: list[dict[str, str]] = []
    stopped_early = False
    roots = input_roots(args.inputs)

    for path in paths:
        result: subprocess.CompletedProcess[str] | None = None
        exc: BaseException | None = None
        file_started_at = perf_counter()
        try:
            result = run_rscript(path, rscript=args.rscript, timeout=args.timeout)
        except BaseException as e:
            exc = e
        elapsed_sec = perf_counter() - file_started_at
        stdout_log, stderr_log = write_logs(path, result, log_dir=args.log_dir, roots=roots)
        stdout_bytes, stderr_bytes = output_sizes(result)
        graphics_flag = "1" if uses_graphics(path) else "0"

        if exc is not None or result is None or result.returncode != 0:
            detail = failure_detail(path, exc, result)
            failures.append((path, detail))
            rows.append({"status": "FAIL", "source": str(path), "returncode": "" if result is None else str(result.returncode), "elapsed_sec": f"{elapsed_sec:.6f}", "stdout_bytes": str(stdout_bytes), "stderr_bytes": str(stderr_bytes), "uses_graphics": graphics_flag, "stdout_log": stdout_log, "stderr_log": stderr_log, "error": detail})
            print(f"FAIL {path}: {detail}")
            if args.max_fail is not None and len(failures) >= args.max_fail:
                stopped_early = True
                break
            continue

        n_ok += 1
        rows.append({"status": "PASS", "source": str(path), "returncode": str(result.returncode), "elapsed_sec": f"{elapsed_sec:.6f}", "stdout_bytes": str(stdout_bytes), "stderr_bytes": str(stderr_bytes), "uses_graphics": graphics_flag, "stdout_log": stdout_log, "stderr_log": stderr_log, "error": ""})
        if not args.quiet:
            print(f"PASS {path}")

    print(f"summary: {n_ok} passed, {len(failures)} failed, {len(paths)} total")
    if stopped_early:
        print(f"stopped: reached --max-fail {args.max_fail}")
    print(f"finished: {datetime.now().isoformat(timespec='seconds')}")
    print(f"elapsed: {perf_counter() - started_at:.2f} seconds")

    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["status", "source", "returncode", "elapsed_sec", "stdout_bytes", "stderr_bytes", "uses_graphics", "stdout_log", "stderr_log", "error"])
            writer.writeheader()
            writer.writerows(rows)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
