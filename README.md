[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/263363730.svg)](https://zenodo.org/badge/latestdoi/263363730)

# Trap-Limited Conversion Efficiency

A collections of scripts first written for calculating the solar energy conversion limits of inorganic crystals. It relies on defect-mediated non-radiative recombination values calculated from [CarrierCapture.jl](https://github.com/WMD-group/CarrierCapture.jl). 

***Package under construction*** 

There are currently three folders:

## [TLC](tlc)

An approach to calculate an upper-limit to photovoltaic efficiency based on an equilibrium population of defects with pre-calculated carrier capture coefficients. The folder contains a worked example for Cu<sub>2</sub>ZnSnSe<sub>4</sub> in the Jupyter Notebook `TLC_CZTSe.ipynb`.

The method has been reported in ["Upper limit to the photovoltaic efficiency of imperfect crystals from first principles", Energy & Environmental Science, 2020](https://pubs.rsc.org/en/content/articlelanding/2020/ee/d0ee00291g).

## [aTLC](atlc)

The original TLC implementation assumed the Shockley–Queisser limit for radiative processes (i.e. above the band gap, all photons are fully absorbed). Here we use a frequency-dependent optical absorption coefficient to calculate the thickness-dependent direct absorption and electron-hole recombination. This approach results in a more realistic estimate of the short-circuit current and device performance limits. The original TLC behaviour is obtained by setting `l_sq=true`. The folder contains a worked example for Cu<sub>2</sub>ZnSnS<sub>4</sub>  in the Jupyter Notebook `aTLC.ipynb`.

The method has been reported in ["Ab initio calculation of the detailed balance limit to the photovoltaic efficiency of single p-n junction kesterite solar cells", 2021](https://aip.scitation.org/doi/10.1063/5.0049143).

## [Wannier90 Absorption](wannier90-absorption)

This code calculates the optical absorption function based on the output of [Wannier90](http://www.wannier.org). The advantage of this procedure is that dense k-point sampling is possible. This approach can provide convergence in the optical properties that is not always possible using standard methods. The output is used as part of [aTLC](atlc). The folder contains a script `wannier_alpha.py` and a set of reference data for Cu<sub>2</sub>ZnSnSe<sub>4</sub> that has been output from Wannier90.  

### Dependencies 

In addition to standard python libraries (`pip install pandas numpy matplotlib scipy pickle seaborn copy re subprocess`), the codes rely on output from [CarrierCapture.jl](https://github.com/WMD-group/CarrierCapture.jl) and [SC-Fermi](https://github.com/jbuckeridge/sc-fermi). We acknowledge that `SQlimit.py` is adapted from https://github.com/marcus-cmc/Shockley-Queisser-limit, while the AM1.5g solar spectrum `ASTMG173.csv` is taken from https://www.nrel.gov/grid/solar-resource/spectra.html. 

### Development

The project is hosted on [Github](https://github.com/WMD-group/traplimitedconversion). Please use the [issue tracker](https://github.com/WMD-group/carriercapture/issues/) for feature requests, bug reports, and more general questions. If you would like to contribute, please do so via a pull request.

