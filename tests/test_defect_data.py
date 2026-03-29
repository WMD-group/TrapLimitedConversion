import numpy as np
import scipy.constants as scpc
import pytest

from tlc.defect_data import DefectData


def test_from_effective_masses_matches_manual():
    """from_effective_masses() agrees with hand-computed n0, p0, N_n, N_p."""
    m_e, m_h = 0.2, 0.8
    fermi_level, e_gap, T = 0.3, 1.2, 300.0

    kT = scpc.k * T
    kT_eV = kT / scpc.eV
    N_n = 2 * (2 * np.pi * m_e * scpc.m_e * kT / scpc.h**2)**1.5 * 1e-6
    N_p = 2 * (2 * np.pi * m_h * scpc.m_e * kT / scpc.h**2)**1.5 * 1e-6
    n0 = N_n * np.exp(-(e_gap - fermi_level) / kT_eV)
    p0 = N_p * np.exp(-fermi_level / kT_eV)

    data_manual = DefectData(
        n0=n0, p0=p0, fermi_level=fermi_level, e_gap=e_gap,
        temperature=T, N_n=N_n, N_p=N_p, traps=[],
    )
    data_auto = DefectData.from_effective_masses(
        fermi_level=fermi_level, e_gap=e_gap, temperature=T,
        m_e=m_e, m_h=m_h,
    )

    assert data_auto.N_n == pytest.approx(data_manual.N_n, rel=1e-12)
    assert data_auto.N_p == pytest.approx(data_manual.N_p, rel=1e-12)
    assert data_auto.n0 == pytest.approx(data_manual.n0, rel=1e-12)
    assert data_auto.p0 == pytest.approx(data_manual.p0, rel=1e-12)
