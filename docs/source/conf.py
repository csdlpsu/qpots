# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
from unittest.mock import MagicMock

MOCK_MODULES = ["matlab.engine"]
sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'qPOTS: Batch Pareto Optimal Thompson Sampling'
copyright = '2025, Kade E. Carlson, Ashwin Renganathan'
author = 'Kade E. Carlson, Ashwin Renganathan'
release = '1.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Enables Google/NumPy docstrings
    'sphinx.ext.viewcode'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
