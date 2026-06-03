#!/usr/bin/env python3
"""Rewrite Burkardt R source() paths to local relative paths.

Burkardt examples often contain paths such as:

    source("/home/john/public_html/r_src/adamsbashforth/adamsbashforth.R")

When the corpus is downloaded locally, those paths should be relative to the
current file, for example:

    source("../adamsbashforth/adamsbashforth.R")
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


SOURCE_RE = re.compile(
    r"""
    source
    \s*\(
    \s*
    (?P<quote>["'])
    (?P<path>/home/john/public_html/r_src/[^"']+)
    (?P=quote)
    \s*
    \)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def rel_source_path(raw_path: str, *, source_file: Path, root: Path) -> str:
    marker = "/r_src/"
    _, _, tail = raw_path.partition(marker)
    if not tail:
        return raw_path
    target = root / tail
    return Path(os.path.relpath(target, source_file.parent)).as_posix()


def rewrite_text(text: str, *, source_file: Path, root: Path) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        quote = match.group("quote")
        fixed = rel_source_path(match.group("path"), source_file=source_file, root=root)
        return f"source({quote}{fixed}{quote})"

    return SOURCE_RE.sub(repl, text), count


def r_files(root: Path) -> list[Path]:
    return sorted(
        path
        for pattern in ("**/*.r", "**/*.R")
        for path in root.glob(pattern)
        if path.is_file()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite Burkardt source() paths to relative local paths.")
    parser.add_argument("root", type=Path, help="Burkardt r_src directory")
    parser.add_argument("--apply", action="store_true", help="modify files in place; default is dry-run")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    changed = 0
    replacements = 0
    for path in r_files(root):
        text = path.read_text(encoding="utf-8-sig")
        new_text, count = rewrite_text(text, source_file=path, root=root)
        if count == 0:
            continue
        changed += 1
        replacements += count
        print(f"{'UPDATE' if args.apply else 'DRY'} {path}: {count}")
        if args.apply:
            path.write_text(new_text, encoding="utf-8")

    print(f"summary: {changed} files, {replacements} replacements")
    if not args.apply:
        print("dry-run only; rerun with --apply to modify files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
