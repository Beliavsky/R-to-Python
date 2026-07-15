from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from xr2p import translate_source


ROOT = Path(__file__).resolve().parents[1]


COMPILE_FIXTURES = [
    "xappend_replace.r",
    "xassign_by_character_name.r",
    "xindex_matrix_assign.r",
    "xindexing.r",
    "xinf_nan.r",
    "xlist_matrices.r",
    "xmatrix_column_major.r",
    "xnames_preserved.r",
    "xregex.r",
    "xstop_warning.r",
    "xsimple_naive_bayes.r",
    "xkalman_filter.r",
    "xem_mix_normals.r",
    "xtibble.r",
    "xinline_lambda.r",
    "xmisc_ops.r",
]


RUN_FIXTURES = [
    "xappend_replace.r",
    "xassign_by_character_name.r",
    "xindex_matrix_assign.r",
    "xindexing.r",
    "xinf_nan.r",
    "xlist_matrices.r",
    "xmatrix_column_major.r",
    "xnames_preserved.r",
    "xregex.r",
    "xstop_warning.r",
    "xinline_lambda.r",
    "xmisc_ops.r",
]


def translate_to_tmp(source_name: str, tmp_path: Path) -> Path:
    source = ROOT / "fixtures" / source_name
    if not source.exists():
        pytest.skip(f"fixture not found: {source}")
    python = translate_source(source.read_text(encoding="utf-8-sig"))
    out = tmp_path / source.with_suffix(".py").name
    out.write_text(python, encoding="utf-8")
    return out


@pytest.mark.parametrize("source_name", COMPILE_FIXTURES)
def test_recent_feature_fixtures_compile(source_name: str, tmp_path: Path) -> None:
    out = translate_to_tmp(source_name, tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(out)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source_name", RUN_FIXTURES)
def test_recent_feature_fixtures_run(source_name: str, tmp_path: Path) -> None:
    out = translate_to_tmp(source_name, tmp_path)
    result = subprocess.run(
        [sys.executable, str(out)],
        text=True,
        capture_output=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
