Quickstart
==========

Shockley-Queisser Limit
-----------------------

The simplest calculation --- the radiative efficiency limit for a given band gap::

   from tlc import TLC

   t = TLC.sq_limit(1.34)  # band gap in eV
   t.calculate()
   print(t)

This gives the maximum efficiency assuming only radiative recombination
and perfect step-function absorption.

TLC with Realistic Absorption
------------------------------

Use a calculated absorption coefficient from DFT::

   from tlc import TLC

   t = TLC(1.2, thickness=2000, alpha="path/to/alpha.csv")
   t.calculate()
   print(t)

The ``alpha`` argument accepts a file path (str/Path), a DataFrame with
columns ``E`` and ``alpha``, or a NumPy array of shape ``(N, 2)``.

TLC with Defect Recombination
------------------------------

Include SRH non-radiative recombination::

   from tlc import TLC, Trap, DefectData

   # Define trap properties
   trap = Trap.single_level(
       name='V_Cd',
       E_t=0.5,           # trap level from VBM (eV)
       N_t=1e15,          # trap concentration (cm^-3)
       q_initial=0,
       q_final=-1,
       C_p=1e-7,          # hole capture coefficient (cm^3/s)
       C_n=1e-8,          # electron capture coefficient (cm^3/s)
   )

   # Carrier concentrations from Fermi level calculation
   data = DefectData(
       n0=1e10,           # equilibrium electrons (cm^-3)
       p0=1e16,           # equilibrium holes (cm^-3)
       fermi_level=0.3,   # from VBM (eV)
       e_gap=1.2,         # band gap (eV)
       temperature=300,   # K
       N_n=1e18,          # effective CB DOS (cm^-3)
       N_p=1e18,          # effective VB DOS (cm^-3)
       traps=[trap],
   )

   t = TLC(1.2, l_sq=True, defect_data=data)
   t.calculate()
   print(t)

Using doped for Carrier Concentrations
---------------------------------------

If you use `doped <https://doped.readthedocs.io>`_ for defect
thermodynamics::

   from tlc.doped_interface import defect_data_from_doped

   # defect_thermo is a doped DefectThermodynamics object
   data = defect_data_from_doped(
       defect_thermo=defect_thermo,
       trap_config={
           "V_Cd": {
               "transitions": [{
                   "q1": 0, "q2": -1,
                   "E_t1": 0.5, "g": 1,
                   "C_p1": 1e-7, "C_n1": 1e-8,
               }]
           }
       },
       temperature=300,
       anneal_temperature=900,
   )

   t = TLC(1.2, l_sq=True, defect_data=data)
   t.calculate()
   print(t)
