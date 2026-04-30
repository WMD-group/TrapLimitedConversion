import pytest
from tlc.tlc import tlc, Trap
from tlc.defect_data import DefectData


def test_trap_str_repr():
    trap = Trap.single_level("V_Cd", E_t=0.5, N_t=1e15,
                             q_initial=0, q_final=-1, C_p=1e-7, C_n=1e-8)
    s = str(trap)
    r = repr(trap)
    assert "V_Cd" in s
    assert "V_Cd" in r
    assert "(0/-1/-)" in s  # q3=None shows as "-"


def test_trap_single_level_q3_none():
    trap = Trap.single_level("V_Cd", E_t=0.5, N_t=1e15,
                             q_initial=0, q_final=-1, C_p=1e-7, C_n=1e-8)
    assert trap.q3 is None


def test_defect_data_validation():
    with pytest.raises(AssertionError):
        DefectData(n0=-1, p0=1e16, fermi_level=0.3, e_gap=1.2,
                   temperature=300, N_n=1e18, N_p=1e18, traps=[])


def test_tlc_invalid_egap():
    with pytest.raises(ValueError):
        tlc(0.1, l_sq=True)


def test_tlc_invalid_temperature():
    with pytest.raises(ValueError):
        tlc(1.5, T=0, l_sq=True)


def test_sq_limit_classmethod():
    t1 = tlc.sq_limit(1.5)
    t1.calculate_rad()
    t2 = tlc(1.5, l_sq=True)
    t2.calculate_rad()
    assert abs(t1.efficiency - t2.efficiency) < 1e-10
    assert abs(t1.j_sc - t2.j_sc) < 1e-10
    assert abs(t1.v_oc - t2.v_oc) < 1e-10


def test_calculate_rad_before_srh():
    t = tlc(1.2, l_sq=True)
    t.calculate_rad()
    sq = tlc.sq_limit(1.2)
    sq.calculate_rad()
    assert abs(t.efficiency - sq.efficiency) / sq.efficiency < 1e-4
