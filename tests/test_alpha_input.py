import os
import warnings

import numpy as np
import pandas as pd
import pytest

from tlc.tlc import tlc

ALPHA_CSV = os.path.join(os.path.dirname(__file__),
                         "..", "examples", "Sb2Se3", "alpha.csv")


@pytest.fixture
def ref():
    """Reference result from CSV file path."""
    t = tlc(1.419, thickness=500, alpha=ALPHA_CSV)
    t.calculate()
    return t


def test_dataframe_matches_csv(ref):
    """A DataFrame produces the same results as reading from CSV."""
    df = pd.read_csv(ALPHA_CSV)
    t = tlc(1.419, thickness=500, alpha=df)
    t.calculate()
    assert t.efficiency == pytest.approx(ref.efficiency, rel=1e-10)
    assert t.j_sc == pytest.approx(ref.j_sc, rel=1e-10)
    assert t.v_oc == pytest.approx(ref.v_oc, rel=1e-10)


def test_ndarray_matches_csv(ref):
    """A NumPy array produces the same results as reading from CSV."""
    df = pd.read_csv(ALPHA_CSV)
    arr = df[["E", "alpha"]].values
    t = tlc(1.419, thickness=500, alpha=arr)
    t.calculate()
    assert t.efficiency == pytest.approx(ref.efficiency, rel=1e-10)


def test_alpha_file_deprecated():
    """Passing alpha_file= still works but emits FutureWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t = tlc(1.419, thickness=500, alpha_file=ALPHA_CSV)
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)
        assert "alpha_file is deprecated" in str(w[0].message)
    t.calculate()
    assert t.efficiency > 0


def test_both_alpha_and_alpha_file_raises():
    """Passing both alpha= and alpha_file= raises ValueError."""
    with pytest.raises(ValueError, match="Cannot pass both"):
        tlc(1.419, thickness=500, alpha="foo.csv", alpha_file="bar.csv")
