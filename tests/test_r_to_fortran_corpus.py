from pathlib import Path

import pytest

from xr2p import translate_source
from xr2p_batch import XR2F_PYTEST_CASES


XR2F_ROOT = Path(r"C:\python\R-to-Fortran")


@pytest.mark.parametrize("name", XR2F_PYTEST_CASES)
def test_translates_r_to_fortran_pytest_examples(name: str) -> None:
    source = XR2F_ROOT / name
    if not source.exists():
        pytest.skip(f"R-to-Fortran corpus file not found: {source}")

    out = translate_source(source.read_text(encoding="utf-8-sig"))

    assert out.startswith("import numpy as np")
    assert "\n" in out
