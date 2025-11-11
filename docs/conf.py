# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'papylio'
copyright = '2025 - Chirlmin Joo lab'
# author = 'Ivo Severins, Sung Hyun Kim, Carolien Bastiaanssen, Iason Katechis, Margreet Docter, Roy Simons, Pim America, '
html_logo = '_static/logo.png'

# from git import Repo
from pathlib2 import Path
# raise ValueError(Path(__file__).parent.parent)
# repo = Repo(Path(__file__).parent.parent)

# sha = repo.head.object.hexsha

# The full version, including alpha/beta/rc tags
# release = f'develop-{sha[0:7]}'
import papylio
release = papylio.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary']
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.coverage",
    # "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.todo",
    # "sphinx.ext.autosectionlabel",
    # "sphinx.ext.githubpages",
    "nbsphinx",
    # "myst_nb",
    # "IPython.sphinxext.ipython_directive",
    # "IPython.sphinxext.ipython_console_highlighting",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# autodoc_member_order = 'groupwise'
autosummary_generate = True
autoclass_content = "class"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme' # 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Napoleon settings for docstring processing -------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
# napoleon_type_aliases = {
#     "scalar": ":term:`scalar`",
#     "sequence": ":term:`sequence`",
#     "callable": ":py:func:`callable`",
#     "file-like": ":term:`file-like <file-like object>`",
#     "array-like": ":term:`array-like <array_like>`",
#     "Path": "~~pathlib.Path",
# }