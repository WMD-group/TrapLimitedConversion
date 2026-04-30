# TrapLimitedConversion (TLC)

## What this project is
A Python package for calculating solar energy conversion limits of inorganic
crystals, accounting for defect-mediated non-radiative recombination
(Shockley-Read-Hall theory). It computes the Trap-Limited Conversion (TLC)
efficiency, which extends the Shockley-Queisser radiative limit by including
realistic absorption spectra and SRH recombination losses.

## Key physics
- Shockley-Queisser (SQ) detailed balance: J_sc from AM1.5G solar spectrum,
  J0_rad from blackbody radiation, J-V curve, V_oc, fill factor, efficiency
- Beer-Lambert absorptivity from optical absorption coefficient α(E) and
  film thickness
- Shockley-Read-Hall (SRH) non-radiative recombination: single-level and
  two-level (three charge state) defect transitions
- Self-consistent Fermi level solver for equilibrium carrier concentrations

## File structure
- `tlc/tlc.py` — main TLC calculator class (J_sc, J0_rad, J-V, SRH, plotting)
- `tlc/defect_data.py` — DefectData dataclass and from_effective_masses() factory
- `tlc/doped_interface.py` — defect_data_from_doped() bridge to doped
- `tlc/__init__.py` — public exports (TLC, tlc, Trap, DefectData, defect_data_from_doped)
- `data/ASTMG173.csv` — NREL AM1.5G reference solar spectrum
- `examples/tlc.ipynb` — tutorial notebook (docs/tutorial.ipynb is a symlink to this)
- `examples/Sb2Se3/` — Sb2Se3 example data

## Units convention
- Energy: eV
- Current density: mA/cm²
- Carrier concentrations: cm⁻³
- Absorption coefficient: cm⁻¹
- Thickness: nm (converted to cm internally via 1e-7)
- Solar irradiance: W m⁻² eV⁻¹ (converted from W m⁻² nm⁻¹)

## Constants
Use scipy.constants throughout. Key values:
- kb_in_eV_per_K = 8.6173303e-5 eV/K
- sun_power = 100 mW/cm² (AM1.5G standard)

## Code style
- Python 3.10+
- NumPy-style docstrings
- snake_case for functions and variables
- Type hints on all public functions
- No pydantic — keep it simple with dataclasses where needed

## Testing
- pytest for all tests
- Key sanity checks:
  - SQ limit for 1.34 eV gap → ~33.7% efficiency
  - SQ limit for 1.5 eV gap → ~30% efficiency
  - J_sc for 1.1 eV → ~44 mA/cm²

## Sanity check command (run after EVERY change)
python -c "from tlc import TLC; t = TLC.sq_limit(1.5); t.calculate(); print(t)"

## Public API
Primary classes and entry points (all importable from `tlc`):
- `TLC` — main calculator class (canonical uppercase name; `tlc` is a backward-compat alias)
  - `TLC.sq_limit(E_gap, ...)` — SQ-limit mode (step-function absorptivity)
  - `TLC(E_gap, alpha=..., defect_data=...)` — realistic absorption mode
  - `TLC.calculate()` — run J-V calculation (auto-computes SRH if defect_data set)
  - `TLC.results` — dict of output quantities
  - `TLC.to_dataframe()` — single-row DataFrame for parameter sweeps
- `Trap` — defect trap level for SRH
  - `Trap.single_level(...)` — two charge state trap (preferred)
  - `Trap.two_level(...)` — three charge state trap
- `DefectData` — container for equilibrium carrier data and trap list
  - `DefectData.from_effective_masses(...)` — compute n0/p0/N_n/N_p from m_e, m_h
- `defect_data_from_doped(...)` — bridge from doped DefectThermodynamics

## Current stage
Stage 3 complete (v0.4.0). See REFACTOR_LOG.md.
All changes should be minimal cleanup — no new features, no architecture changes.
The goal is making the existing code clean, consistent, and well-documented.

## Rules for changes
1. ONE task per session — do not refactor multiple things at once
2. After every change, run the sanity check command above
3. Do not change public API signatures unless explicitly asked. Deprecate with FutureWarning first.
4. Do not add new dependencies unless explicitly asked
5. Preserve existing unit conventions (CGS: cm⁻³, mA/cm², cm⁻¹)
6. Git commit after every successful change with a descriptive message
7. If unsure about physics, explain your understanding and ask me to confirm
