# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'TLC'
release = '0.4.0-dev'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'myst_nb',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

myst_enable_extensions = [
    "html_admonition",
    "html_image",
    "dollarmath",
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

html_context = {
    "display_github": True,
    "github_user": "WMD-group",
    "github_repo": "TrapLimitedConversion",
    "github_version": "refactor-v2",
    "conf_py_path": "/docs/",
}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for autodoc -----------------------------------------------------

autoclass_content = "both"

# -- Options for nb extension ------------------------------------------------

nb_execution_mode = "off"
myst_heading_anchors = 2
