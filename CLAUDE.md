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
- `tlc/scfermi.py` — self-consistent Fermi level solver (to be replaced by doped)
- `data/ASTMG173.csv` — NREL AM1.5G reference solar spectrum
- `examples/` — Cu2ZnSnS4 and Sb2Se3 example calculations

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

## Current stage
Pre-release polish for v0.4.0. See REFACTOR_LOG.md.
All changes should be minimal cleanup — no new features, no architecture changes.
The goal is making the existing code clean, consistent, and well-documented.

## Rules for changes
1. ONE task per session — do not refactor multiple things at once
2. After every change, run the sanity check command above
3. Do not change function signatures unless explicitly asked
4. Do not add new dependencies unless explicitly asked
5. Preserve existing unit conventions (CGS: cm⁻³, mA/cm², cm⁻¹)
6. Git commit after every successful change with a descriptive message
7. If unsure about physics, explain your understanding and ask me to confirm
