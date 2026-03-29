[![Tests](https://github.com/WMD-group/TrapLimitedConversion/actions/workflows/tests.yml/badge.svg)](https://github.com/WMD-group/TrapLimitedConversion/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/263363730.svg)](https://zenodo.org/badge/latestdoi/263363730)

# Trap-Limited Conversion Efficiency

Tools for calculating the solar energy conversion limits of inorganic crystals. The approach relies on defect-mediated non-radiative recombination values calculated from [CarrierCapture.jl](https://github.com/WMD-group/CarrierCapture.jl) or similar packages such as [NonRad](https://github.com/mturiansky/nonrad).

Tutorials can be found on the [docs](https://traplimitedconversion.readthedocs.io) site.

We acknowledge that some code related to radiative detailed balance was adapted from https://github.com/marcus-cmc/Shockley-Queisser-limit, while the AM1.5g solar spectrum `ASTMG173.csv` is from [NREL](https://www.nrel.gov/grid/solar-resource/spectra.html).

## Installation

```bash
pip install -e .          # core
pip install -e ".[doped]" # with doped integration
pip install -e ".[dev]"   # with pytest and ruff
```

## Quick Start

### Shockley-Queisser limit

```python
from tlc import TLC

t = TLC.sq_limit(1.34)
t.calculate()
print(f"Efficiency: {t.efficiency*100:.1f}%")  # ~33.7%
```

### TLC with direct defect input

```python
from tlc import TLC, Trap, DefectData

trap = Trap.single_level("V_Cd", E_t=0.5, N_t=1e15,
                         q_initial=0, q_final=-1,
                         C_p=1e-7, C_n=1e-8)
data = DefectData(n0=1e10, p0=1e16, fermi_level=0.3,
                  e_gap=1.2, temperature=300,
                  N_n=1e18, N_p=1e18, traps=[trap])
t = TLC(1.2, alpha="alpha.csv", defect_data=data)
t.calculate()
print(t.results)
```

### TLC with doped integration

```python
from doped.thermodynamics import DefectThermodynamics
from tlc.doped_interface import defect_data_from_doped

thermo = DefectThermodynamics.from_json("defect_thermo.json")
trap_config = {"V_Cd": {"transitions": [
    {"q1": 0, "q2": -1, "E_t1": 0.5, "C_p1": 1e-7, "C_n1": 1e-8}
]}}
data = defect_data_from_doped(thermo, trap_config, temperature=300,
                              anneal_temperature=900)
t = TLC(1.2, alpha="alpha.csv", defect_data=data)
t.calculate()
```

The lowercase `tlc` and `calculate_rad()` are still supported for backward compatibility.

See the [tutorial notebook](https://traplimitedconversion.readthedocs.io/en/latest/Tutorials.html) for the full walkthrough.

## Related Packages

* [Doped](https://doped.readthedocs.io) - pre- and post-processing of point defect calculations

* [ShakeNBreak](https://shakenbreak.readthedocs.io) - approach to find symmetry broken solutions

* [SC-Fermi](https://github.com/jbuckeridge/sc-fermi) / [py-SC-Fermi](https://github.com/bjmorgan/py-sc-fermi) - equilibrium self-consistent Fermi level in Fortran / Python

* [Wannier90](http://www.wannier.org) - allows calculation of optical absorption with dense k-point sampling

## Used in

* The original method is reported in ["Upper limit to the photovoltaic efficiency of imperfect crystals from first principles"](https://pubs.rsc.org/en/content/articlelanding/2020/ee/d0ee00291g)

* An update to include the optical absorption spectrum in ["Ab initio calculation of the detailed balance limit to the photovoltaic efficiency of single p-n junction kesterite solar cells"](https://aip.scitation.org/doi/10.1063/5.0049143)

* Application to CdTe in ["Rapid recombination by cadmium vacancies in CdTe"](https://pubs.acs.org/doi/10.1021/acsenergylett.1c00380)

* Application to Sb<sub>2</sub>Se<sub>3</sub> in ["Upper efficiency limit of Sb<sub>2</sub>Se<sub>3</sub> solar cells"](https://arxiv.org/abs/2402.04434)

## Development

The project is hosted on [Github](https://github.com/WMD-group/traplimitedconversion). Please use the [issue tracker](https://github.com/WMD-group/TrapLimitedConversion/issues) for feature requests, bug reports, and more general questions. If you would like to contribute, please do so via a pull request.
