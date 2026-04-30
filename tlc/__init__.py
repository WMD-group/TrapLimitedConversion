"""Trap limited conversion efficiency analysis for photovoltaic materials"""

from tlc.tlc import tlc as TLC  # promoted canonical name
from tlc.tlc import tlc, Trap   # backward-compatible lowercase
from tlc.defect_data import DefectData

from tlc._version import __version__


def defect_data_from_doped(*args, **kwargs):
    """Re-export from tlc.doped_interface (requires doped)."""
    from tlc.doped_interface import defect_data_from_doped as _fn
    return _fn(*args, **kwargs)


__all__ = ["TLC", "tlc", "Trap", "DefectData", "defect_data_from_doped"]
