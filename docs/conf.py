# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'reggae'
copyright = '2024, Joel Ong, Martin Nielsen, Emily Hatt, Guy Davies, Bill Chaplin'
author = 'Joel Ong, Martin Nielsen, Emily Hatt, Guy Davies, Bill Chaplin'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinx.ext.mathjax',
    'myst_parser'
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "smartquotes",
    "strikethrough",
]

autoapi_dirs = ['../reggae']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
