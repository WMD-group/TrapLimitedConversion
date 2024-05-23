[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/263363730.svg)](https://zenodo.org/badge/latestdoi/263363730)

# Trap-Limited Conversion Efficiency

Tools for calculating the solar energy conversion limits of inorganic crystals. The approach relies on defect-mediated non-radiative recombination values calculated from [CarrierCapture.jl](https://github.com/WMD-group/CarrierCapture.jl) or similar packages such as [NonRad](https://github.com/mturiansky/nonrad).

Tutorials can be found on the [docs](https://traplimitedconversion.readthedocs.io/en/latest/index.html) site.

We acknowledge that some code related to radiative detailed balance was adapted from https://github.com/marcus-cmc/Shockley-Queisser-limit, while the AM1.5g solar spectrum `ASTMG173.csv` is from [NREL](https://www.nrel.gov/grid/solar-resource/spectra.html). 

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

The project is hosted on [Github](https://github.com/WMD-group/traplimitedconversion). Please use the [issue tracker](https://github.com/WMD-group/carriercapture/issues/) for feature requests, bug reports, and more general questions. If you would like to contribute, please do so via a pull request.
