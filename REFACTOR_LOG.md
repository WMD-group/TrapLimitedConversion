# TLC Refactoring Log

## Current Stage: 3 (In Progress)
## Current Step: 3.6 done

## Completed Steps
- 1.1: Fix Trap.__str__ typos (E_t → E_t1, elf → self) ✓
- 1.3: Fix efficiency formula operator precedence for concentrated light ✓
- 1.5: Remove DOS normalization overwrite in scfermi._run ✓
- 1.6: main_interpolate uses Tanneal parameter instead of hardcoded 853 ✓
- 1.7: Scfermi.from_file uses cls for classmethod ✓
- 1.9: Remove debug print statements from __get_delta_n ✓
- 1.10: np.arange float step → np.linspace for reproducibility ✓
- 2.1: Add pyproject.toml, create tests directory ✓
- 2.2: Add SQ-limit regression tests (8 tests, all passing) ✓
- 2.3: Add CI test workflow for Python 3.10 and 3.12 ✓
- 2.4: Add DefectData dataclass for SRH input ✓
- 2.5: Add doped interface (defect_data_from_doped) ✓
- 2.6: Add calculate_SRH_from_data() with DefectData support ✓
- 2.7: Configurable alpha_file, add sq_limit classmethod, remove poscar/totdos from __init__ ✓
- 2.8: Trap uses keyword args, add single_level factory ✓
- 2.9: Add SRH integration tests (5 tests, all passing) ✓
- 2.10: Remove scfermi dependency from tlc.py (-116 lines) ✓
- 2.11: Clean package exports, move scfermi to legacy, version 0.4.0 ✓
- 3.1: Add unified calculate() method with optional defect_data at init ✓
- 3.2: Add results property and to_dataframe() method ✓
- 3.3: Add ax= parameter to all plot methods for composability ✓
- 3.4: Accept DataFrame/array for absorption data, deprecate alpha_file ✓
- 3.5: Promote TLC as canonical class name, add __all__ ✓
- 3.6: Add DefectData.from_effective_masses() classmethod ✓

## Skipped
- 1.4: __find_max_point — original sign convention is internally consistent, not a bug
- 1.8: occupation clamping — current clamp at 1 is defensible as dilute-limit guard. Full fix requires Fermi-Dirac statistics. Moot since scfermi.py is dropped in Stage 2.

## Deferred
- 1.2: octal literal fix → deferred to Stage 2 (Trap class cleanup)

## Notes
- Full plan: tlc_refactor_plan.md
- Bug reference: see test branch bugs.md
