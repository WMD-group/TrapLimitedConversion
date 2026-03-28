from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
