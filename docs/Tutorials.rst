.. _Tutorials:

Tutorials
=========

The tutorial notebook below walks through the full TLC workflow:

- **Shockley-Queisser limit** --- radiative efficiency for a given band gap
- **Realistic absorption** --- using a DFT-calculated absorption coefficient
- **Trap-limited conversion** --- including SRH defect recombination
- **Plotting** --- composable matplotlib plots with ``ax=``
- **Parameter sweeps** --- collecting results with ``to_dataframe()``

The ``alpha`` input accepts a file path (str/Path), a ``pd.DataFrame``
with columns ``E`` (energy in eV) and ``alpha`` (absorption coefficient
in cm\ :sup:`-1`), or a NumPy array of shape ``(N, 2)``.

For integration with `doped <https://doped.readthedocs.io>`_, see
``defect_data_from_doped()`` in the :doc:`API reference <api>`.

.. toctree::
   :maxdepth: 1

   tutorial.ipynb
