import pytest
from tlc.tlc import tlc


@pytest.fixture
def sq_1p34():
    t = tlc(1.34, l_sq=True)
    t.calculate_rad()
    return t


def test_sq_efficiency_1p34eV(sq_1p34):
    assert 33.0 < sq_1p34.efficiency * 100 < 34.5


def test_sq_jsc_1p1eV():
    t = tlc(1.1, l_sq=True)
    t.calculate_rad()
    assert 42 < t.j_sc < 46


def test_sq_voc_1p34eV(sq_1p34):
    assert 1.0 < sq_1p34.v_oc < 1.15


def test_sq_ff_reasonable(sq_1p34):
    assert 0.85 < sq_1p34.ff < 0.92


def test_sq_voc_increases_with_gap():
    t_low = tlc(1.1, l_sq=True)
    t_low.calculate_rad()
    t_high = tlc(1.5, l_sq=True)
    t_high.calculate_rad()
    assert t_high.v_oc > t_low.v_oc


def test_sq_absorptivity_step(sq_1p34):
    import numpy as np
    # SQ limit: step-function absorptivity at E_gap
    below_gap = sq_1p34.absorptivity[sq_1p34.Es < 1.34 - 0.05]
    above_gap = sq_1p34.absorptivity[sq_1p34.Es > 1.34 + 0.05]
    assert np.all(below_gap == 0)
    assert np.all(above_gap == 1)


def test_sq_jsc_positive(sq_1p34):
    assert sq_1p34.j_sc > 0


def test_sq_j0_rad_positive(sq_1p34):
    assert sq_1p34.j0_rad > 0
    assert sq_1p34.j0_rad < sq_1p34.j_sc * 1e-10


def test_results_property(sq_1p34):
    expected_keys = {"j_sc", "j0_rad", "v_oc", "v_max", "j_max", "ff", "efficiency"}
    r = sq_1p34.results
    assert isinstance(r, dict)
    assert set(r.keys()) == expected_keys


def test_results_before_calculate_raises():
    t = tlc(1.5, l_sq=True)
    with pytest.raises(RuntimeError, match="No results yet"):
        _ = t.results


def test_to_dataframe(sq_1p34):
    import pandas as pd
    df = sq_1p34.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "E_gap" in df.columns
    assert "efficiency" in df.columns
