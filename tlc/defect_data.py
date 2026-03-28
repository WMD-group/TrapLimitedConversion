from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tlc.tlc import Trap


@dataclass
class DefectData:
    """Container for equilibrium carrier data and trap properties.

    This is the interface between external defect tools (e.g. doped)
    and the TLC SRH calculator. Users can create this directly with
    numerical values or use the from_doped() helper.

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
