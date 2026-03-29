TrapLimitedConversion (TLC)
===========================

``TrapLimitedConversion`` computes solar energy conversion limits of
inorganic crystals, accounting for defect-mediated non-radiative
recombination via Shockley-Read-Hall (SRH) theory.

Key features
------------

- **Shockley-Queisser limit** with step-function or realistic absorptivity
- **SRH recombination** from single-level and two-level defect traps
- **One-step workflow**: pass ``defect_data`` at construction time
- **Parameter sweeps** with ``to_dataframe()``
- **Composable plots** via ``ax=`` on all plot methods
- Integration with `doped <https://doped.readthedocs.io>`_ for carrier concentrations

See the :ref:`tutorial <Tutorials>` for usage examples.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Usage

   installation
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
