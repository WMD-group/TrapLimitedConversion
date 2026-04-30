import pytest
import numpy as np
from tlc.tlc import tlc, Trap
from tlc.defect_data import DefectData


E_GAP = 1.2


def _make_defect_data(traps, n0=1e10, p0=1e16):
    return DefectData(
        n0=n0, p0=p0, fermi_level=0.3, e_gap=E_GAP,
        temperature=300, N_n=1e18, N_p=1e18, traps=traps,
    )


def _calc_with_srh(defect_data):
    t = tlc(E_GAP, l_sq=True)
    t.calculate_SRH_from_data(defect_data)
    t.calculate_rad()
    return t


def _sq_efficiency():
    t = tlc(E_GAP, l_sq=True)
    t.calculate_rad()
    return t.efficiency


def test_zero_trap_gives_sq_limit():
    trap = Trap.single_level("zero", E_t=0.5, N_t=0, q_initial=0, q_final=-1,
                             C_p=1e-7, C_n=1e-8)
    t = _calc_with_srh(_make_defect_data([trap]))
    sq_eff = _sq_efficiency()
    assert abs(t.efficiency - sq_eff) / sq_eff < 1e-4


def test_single_trap_reduces_efficiency():
    trap = Trap.single_level("V_test", E_t=0.5, N_t=1e15, q_initial=0,
                             q_final=-1, C_p=1e-7, C_n=1e-8)
    t = _calc_with_srh(_make_defect_data([trap]))
    sq_eff = _sq_efficiency()
    assert t.efficiency < sq_eff


def test_higher_concentration_lower_efficiency():
    trap_low = Trap.single_level("V_lo", E_t=0.5, N_t=1e14, q_initial=0,
                                 q_final=-1, C_p=1e-7, C_n=1e-8)
    trap_high = Trap.single_level("V_hi", E_t=0.5, N_t=1e16, q_initial=0,
                                  q_final=-1, C_p=1e-7, C_n=1e-8)
    t_low = _calc_with_srh(_make_defect_data([trap_low]))
    t_high = _calc_with_srh(_make_defect_data([trap_high]))
    assert t_high.efficiency < t_low.efficiency


def test_two_traps_worse_than_one():
    trap1 = Trap.single_level("V_1", E_t=0.4, N_t=1e15, q_initial=0,
                              q_final=-1, C_p=1e-7, C_n=1e-8)
    trap2 = Trap.single_level("V_2", E_t=0.7, N_t=1e15, q_initial=0,
                              q_final=-1, C_p=1e-7, C_n=1e-8)
    t_one = _calc_with_srh(_make_defect_data([trap1]))
    t_two = _calc_with_srh(_make_defect_data([trap1, trap2]))
    assert t_two.efficiency < t_one.efficiency


def test_r_srh_positive():
    trap = Trap.single_level("V_test", E_t=0.5, N_t=1e15, q_initial=0,
                             q_final=-1, C_p=1e-7, C_n=1e-8)
    t = _calc_with_srh(_make_defect_data([trap]))
    # R_SRH should be non-negative for forward bias (V > 0)
    forward_mask = t.Vs > 0.05
    assert np.all(t.R_SRH[forward_mask] >= 0)
