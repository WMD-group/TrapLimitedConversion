TrapLimitedConversion (TLC)
===========================

``TrapLimitedConversion`` computes solar energy conversion limits of
inorganic crystals, accounting for defect-mediated non-radiative
recombination via Shockley-Read-Hall (SRH) theory.

Quick Start
-----------

Shockley-Queisser limit::

   from tlc import tlc
   t = tlc.sq_limit(1.34)
   t.calculate_rad()
   print(t)  # ~33.7% efficiency

TLC with defect data::

   from tlc import tlc, Trap, DefectData
   trap = Trap.single_level('V_Cd', E_t=0.5, N_t=1e15,
                            q_initial=0, q_final=-1,
                            C_p=1e-7, C_n=1e-8)
   data = DefectData(n0=1e10, p0=1e16, fermi_level=0.3,
                     e_gap=1.2, temperature=300,
                     N_n=1e18, N_p=1e18, traps=[trap])
   t = tlc(1.2, l_sq=True)
   t.calculate_SRH_from_data(data)
   t.calculate_rad()
   print(t)

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Usage

   installation
   quickstart
   Tutorials

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :hidden:
   :caption: Information

   changelog
