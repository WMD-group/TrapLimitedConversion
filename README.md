[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Trap-Limited Conversion Efficiency

A collections of scripts written by Dr Sunghyun Kim for calculating the solar energy conversion limits of inorganic materials. It builds on the [CarrierCapture.jl](https://github.com/WMD-group/CarrierCapture.jl) package.
There are plans for the underlying code to be refactored into a coherent workflow. 
There are currently three folders:

## [TLC](tlc)

An approach to calculate an upper-limit to photovoltaic efficiency based on an equilibrium population of defects with pre-calculated carrier capture coefficients. The method has been reported in ["Upper limit to the photovoltaic efficiency of imperfect crystals from first principles", Energy & Environmental Science, 2020](https://pubs.rsc.org/en/content/articlelanding/2020/ee/d0ee00291g)

## [ATLC](atlc)

The original TLC implementation assumed the Shockley–Queisser limit for radiative processes (e.g. above the band gap, all photons are fully absorbed). Here we use a frequency-dependent optical absorption coefficient to calculate the thickness-dependent direct absorption and electron-hole recombination. This approach results in a more realistic estimate of the short-circuit current and device performance limits. The original TLC behaviour is obtained by setting `l_sq=true`.

## [Wannier90 Absorption](wannier90-absorption)

This code calculates the optical absorption function based on the output of [Wannier90](http://www.wannier.org). The advantage of this procedure is that dense k-point sampling is possible. This can provide convergence in the dielectric and optical properties that is not always possible using standard methods. The output is used as part of ATLC.

### Dependencies 

In addition to standard python libraries (`pip install pandas numpy matplotlib scipy pickle seaborn copy re subprocess`), `SQlimit.py` is taken from https://github.com/marcus-cmc/Shockley-Queisser-limit, and the notebook uses output from [CarrierCapture.jl](https://github.com/WMD-group/CarrierCapture.jl) and [SC-Fermi](https://github.com/jbuckeridge/sc-fermi).

### Development

The project is hosted on [Github](https://github.com/WMD-group/traplimitedconversion).
Please use the [issue tracker](https://github.com/WMD-group/carriercapture/issues/) for feature requests, bug reports and more general questions.
If you would like to contribute, please do so via a pull request.




