Changelog
=========

v0.4.0 (2026)
--------------

Bug Fixes
~~~~~~~~~
- Fixed ``Trap.__str__`` crash (typo referencing non-existent attributes)
- Fixed efficiency formula operator precedence for concentrated light
- Fixed DOS normalization overwrite in scfermi solver
- Fixed hardcoded ``Tanneal=853`` ignoring user parameter
- Fixed ``@classmethod`` using ``self`` instead of ``cls``
- Fixed ``q3=0`` sentinel conflicting with real charge state 0; now uses ``q3=None`` for single-level traps
- Removed debug print statements
- Replaced ``np.arange`` float step with ``np.linspace``

New Features
~~~~~~~~~~~~
- ``DefectData`` dataclass for direct SRH input
- ``defect_data_from_doped()`` integration with doped package
- ``Trap.single_level()`` factory for common single-level traps
- ``tlc.sq_limit()`` classmethod for quick SQ calculations
- Configurable ``alpha_file`` path
- Keyword arguments for ``Trap`` class

Refactoring
~~~~~~~~~~~
- Removed dependency on internal ``scfermi.py`` (moved to ``tlc/legacy/``)
- Removed ``pymatgen`` as required dependency
- Removed ``setup.py`` (replaced by ``pyproject.toml``)
- Removed ``pkg_resources`` deprecation warning
- Added ``pyproject.toml`` with modern packaging
- Added pytest test suite (SQ limit + SRH integration tests)
- Added GitHub Actions CI
