from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as scpc

if TYPE_CHECKING:
    from tlc.tlc import Trap


@dataclass
class DefectData:
    """Container for equilibrium carrier data and trap properties.

    This is the interface between external defect tools (e.g. doped)
    and the TLC SRH calculator. Users can create this directly with
    numerical values or use ``defect_data_from_doped()``.

    Parameters
    ----------
    n0 : float
        Equilibrium electron concentration (cm^-3).
    p0 : float
        Equilibrium hole concentration (cm^-3).
    fermi_level : float
        Fermi level position from VBM (eV).
    e_gap : float
        Band gap (eV).
    temperature : float
        Operating temperature (K).
    N_n : float
        Effective conduction band DOS (cm^-3).
    N_p : float
        Effective valence band DOS (cm^-3).
    traps : list[Trap]
        List of Trap objects with N_t (concentration) already set.

    Examples
    --------
    >>> from tlc import Trap, DefectData
    >>> trap = Trap.single_level("V_Cd", E_t=0.5, N_t=1e15,
    ...                          q_initial=0, q_final=-1,
    ...                          C_p=1e-7, C_n=1e-8)
    >>> data = DefectData(n0=1e10, p0=1e16, fermi_level=0.3,
    ...                   e_gap=1.2, temperature=300,
    ...                   N_n=1e18, N_p=1e18, traps=[trap])
    """

    n0: float
    p0: float
    fermi_level: float
    e_gap: float
    temperature: float
    N_n: float
    N_p: float
    traps: list = field(default_factory=list)

    def __post_init__(self):
        assert self.n0 >= 0, f"n0 must be non-negative, got {self.n0}"
        assert self.p0 >= 0, f"p0 must be non-negative, got {self.p0}"
        assert self.e_gap > 0, f"e_gap must be positive, got {self.e_gap}"
        assert self.temperature > 0, f"temperature must be positive, got {self.temperature}"

    @classmethod
    def from_effective_masses(
        cls,
        fermi_level: float,
        e_gap: float,
        temperature: float,
        m_e: float,
        m_h: float,
        traps: list | None = None,
    ) -> DefectData:
        """Create DefectData by computing DOS and carrier concentrations
        from effective masses.

        Parameters
        ----------
        fermi_level : float
            Fermi level position from VBM (eV).
        e_gap : float
            Band gap (eV).
        temperature : float
            Operating temperature (K).
        m_e : float
            Electron effective mass (in units of free electron mass m0).
        m_h : float
            Hole effective mass (in units of free electron mass m0).
        traps : list[Trap] or None
            List of Trap objects. Defaults to empty list.

        Returns
        -------
        DefectData
            Instance with computed ``N_n``, ``N_p``, ``n0``, ``p0``.

        Examples
        --------
        >>> data = DefectData.from_effective_masses(
        ...     fermi_level=0.3, e_gap=1.2, temperature=300,
        ...     m_e=0.2, m_h=0.8)
        >>> print(f"n0={data.n0:.2e}, p0={data.p0:.2e}")
        """
        kT = scpc.k * temperature  # J
        kT_eV = kT / scpc.eV      # eV

        # Effective density of states (m⁻³), then convert to cm⁻³
        N_n = 2 * (2 * np.pi * m_e * scpc.m_e * kT / scpc.h**2)**1.5 * 1e-6
        N_p = 2 * (2 * np.pi * m_h * scpc.m_e * kT / scpc.h**2)**1.5 * 1e-6

        # Equilibrium carrier concentrations (cm⁻³)
        n0 = N_n * np.exp(-(e_gap - fermi_level) / kT_eV)
        p0 = N_p * np.exp(-fermi_level / kT_eV)

        return cls(
            n0=float(n0),
            p0=float(p0),
            fermi_level=fermi_level,
            e_gap=e_gap,
            temperature=temperature,
            N_n=float(N_n),
            N_p=float(N_p),
            traps=traps if traps is not None else [],
        )
