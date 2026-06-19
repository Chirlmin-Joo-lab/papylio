"""Papylio: Single-Molecule Data Analysis and Visualization Platform.

A comprehensive Python package for analyzing, visualizing, and extracting insights
from single-molecule microscopy data, including FRET analysis, trace extraction,
hidden Markov modeling, and sequencing data integration.

Main Classes
------------
- Experiment: Collection of data files in an experiment
- File: Single-molecule data file with coordinate and trace analysis
- Movie: Microscopy image sequence/movie loader
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

_LAZY_IMPORTS = {
    "Experiment": "papylio.experiment",
    "File": "papylio.file",
    "Movie": "papylio.movie",
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache it so the next access is instant, no __getattr__ call
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())

def get_version():
    import subprocess, datetime

    repo_root = Path(__file__).parent.parent

    __tag__ = subprocess.check_output(
        ["git", "-C", str(repo_root), "describe", "--tags", "--abbrev=0"],
        text=True,
    ).strip()[1:]

    __commit__ = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True
    ).strip()[:9]

    __date__ = datetime.datetime.now().strftime("%Y%m%d")

    return __tag__ + "+g" + __commit__ + ".d" + __date__

try:
    from ._version import version as __version__
except ImportError:
    try:
        __version__ = get_version()
    except (LookupError, ImportError):
        __version__ = '0.0.0'

from papylio.configuration import setup_user_config, load_user_microscope_classes
setup_user_config()
load_user_microscope_classes()