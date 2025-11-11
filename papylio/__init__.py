import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from papylio.experiment import Experiment
from papylio.file import File
# from papylio.molecule import Molecule
from papylio.movie.movie import Movie

# import papylio.sequencing.sequencing
#
# Experiment = type(Experiment.__name__, (Experiment,) + tuple(Experiment.plugins), {})

try:
    from ._version import version as __version__
except ImportError:
    try:
        import setuptools_scm
        __version__ = setuptools_scm.get_version(version_scheme="only-version", local_scheme="node-and-date",
                                                 root='..', relative_to=__file__)
    except (LookupError, ImportError):
        __version__ = '0.0.0'