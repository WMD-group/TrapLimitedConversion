"""Interface between doped DefectThermodynamics and TLC DefectData."""

from __future__ import annotations

import numpy as np

from tlc.defect_data import DefectData
from tlc.tlc import Trap, kb_in_eV_per_K


def defect_data_from_doped(
    defect_thermo,
    trap_config: dict,
    temperature: float = 300.0,
    anneal_temperature: float | None = None,
) -> DefectData:
    """Create DefectData from a doped DefectThermodynamics object.

    Parameters
    ----------
    defect_thermo : doped.thermodynamics.DefectThermodynamics
        Parsed defect thermodynamics from doped.
    trap_config : dict
        Maps defect names to capture coefficients. Format::

            {
                "V_Sb": {
                    "transitions": [
                        {"q1": 0, "q2": -1, "E_t1": 0.5,
                         "E_t2": 0.0, "g": 1,
                         "C_p1": 1e-7, "C_n1": 1e-8,
                         "C_p2": 0, "C_n2": 0},
                    ]
                },
            }

    temperature : float
        Operating temperature in K (default 300).
    anneal_temperature : float or None
        Annealing temperature in K. If provided, uses the frozen defect
        approximation via get_fermi_level_and_concentrations.
        If None, uses equilibrium at ``temperature``.

    Returns
    -------
    DefectData
        Container with equilibrium carriers, effective DOS, and Trap objects.

    Examples
    --------
    >>> from doped.thermodynamics import DefectThermodynamics
    >>> thermo = DefectThermodynamics.from_json("defect_thermo.json")
    >>> config = {"V_Cd": {"transitions": [
    ...     {"q1": 0, "q2": -1, "E_t1": 0.5, "C_p1": 1e-7, "C_n1": 1e-8}
    ... ]}}
    >>> data = defect_data_from_doped(thermo, config, temperature=300)
    """
    try:
        from doped.thermodynamics import DefectThermodynamics as _DT  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "doped is required for defect_data_from_doped. "
            "Install with: pip install doped"
        ) from exc

    e_gap = defect_thermo.band_gap

    # Get Fermi level, carrier concentrations, and defect concentrations
    if anneal_temperature is not None:
        fermi_level, n0, p0, conc_df = (
            defect_thermo.get_fermi_level_and_concentrations(
                annealing_temperature=anneal_temperature,
                quenched_temperature=temperature,
                per_charge=True,
            )
        )
    else:
        fermi_level, n0, p0 = defect_thermo.get_equilibrium_fermi_level(
            temperature=temperature,
            return_concs=True,
        )
        conc_df = defect_thermo.get_equilibrium_concentrations(
            temperature=temperature,
            fermi_level=fermi_level,
            per_charge=True,
        )

    # Effective DOS from inverted Boltzmann relation
    kT = kb_in_eV_per_K * temperature
    N_p = p0 / np.exp(-fermi_level / kT)
    N_n = n0 / np.exp(-(e_gap - fermi_level) / kT)

    # Build Trap objects from trap_config + doped concentrations
    traps = []
    for defect_name, config in trap_config.items():
        for trans in config["transitions"]:
            q1 = trans["q1"]
            q2 = trans["q2"]
            q3 = trans.get("q3", None)
            charge_states = {q1, q2} if q3 is None else {q1, q2, q3}

            # Filter conc_df for this defect and relevant charge states
            mask = (
                conc_df["Defect"].str.contains(defect_name)
                & conc_df["Charge"].isin(charge_states)
            )
            N_t = conc_df.loc[mask, "Concentration (cm^-3)"].sum()

            trap = Trap(
                name=defect_name,
                E_t1=trans["E_t1"],
                E_t2=trans.get("E_t2", 0.0),
                N_t=N_t,
                q1=q1,
                q2=q2,
                q3=q3,
                g=trans.get("g", 1),
                C_p1=trans["C_p1"],
                C_p2=trans.get("C_p2", 0),
                C_n1=trans["C_n1"],
                C_n2=trans.get("C_n2", 0),
            )
            traps.append(trap)

    return DefectData(
        n0=n0,
        p0=p0,
        fermi_level=fermi_level,
        e_gap=e_gap,
        temperature=temperature,
        N_n=N_n,
        N_p=N_p,
        traps=traps,
    )
