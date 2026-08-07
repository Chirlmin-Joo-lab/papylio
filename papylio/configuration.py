"""
papylio/config.py

Manages the Papylio user configuration directory and default config file.
Call `setup_user_config()` on application startup.
"""

from pathlib import Path
import importlib.util
import sys

from platformdirs import user_config_dir

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # backport
    except ImportError:
        raise ImportError("Install 'tomli' for Python < 3.11: pip install tomli")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_DIR = Path(user_config_dir("papylio",  appauthor=False))
CONFIG_FILE = CONFIG_DIR / "config.toml"
MICROSCOPES_DIR = CONFIG_DIR / "microscopes"


# ---------------------------------------------------------------------------
# Default content
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = """\
# Papylio configuration
# ---------------------

# Strings in file names to ignore when searching for files
excluded_names: [_ave, _max, _corrections, _dwells, _dwell_analysis, darkfield, flatfield, _sequencing_data, _sequencing_match]

# Folders not to be searched for files
excluded_folders: [Analysis, Sequencing data, Results]
  
# Name of the default microscope to use
# default_microscope = 
"""

EXAMPLE_MICROSCOPE = '''\
"""
Example Papylio microscope profile.

Copy this file, rename it to match your microscope (e.g. my_scope.py),
and fill in the parameters. The class name does not matter — Papylio
discovers all Movie subclasses defined in this directory. Subclass can be one of the built-in Movie types (e.g. TIFMovie, ND2) or a subclass of Movie.

The `parse_metadata` method is optional. If your microscope embeds
metadata in the file (e.g. MicroManager TIFF), override it to extract
values dynamically. Return a dict; any keys returned here will override
the class-level defaults for that acquisition.
"""

from papylio import TIFMovie


class MyScopeMovie(TIFMovie):
    microscope = "MyScope"
    version = "10-06-2026"

    rotation = 1

    channels = ['green', 'red']
    channel_arrangement = np.array([[[0, 1]]])

    illuminations = ['green', 'red']
    default_illumination = 0

    psf_size = 1.01
'''

README = """\
Papylio user configuration directory
=====================================

config.toml
    General settings for Papylio (default microscope, GUI preferences, etc.).
    Edit this file to change application-wide defaults.

microscopes/
    One .py file per microscope profile.
    Each file defines a MicroscopeProfile subclass with the optical and
    acquisition parameters for that microscope, plus an optional
    parse_metadata() method for extracting metadata from raw files.

    Papylio discovers all profiles (except example) in this directory automatically.
    To add a new microscope:
      1. Copy and rename microscopes/example.py.
      2. Fill in the parameters
      3. Reload profiles in the GUI (or restart Papylio)

    Profiles can be shared by simply exchanging the .py file.
"""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_user_config(force: bool = False) -> None:
    """
    Create the Papylio config directory structure if it does not exist.

    Parameters
    ----------
    force:
        If True, overwrite the config file and example microscope even if
        they already exist. Existing custom microscope files are never touched.
    """
    _create_directories()
    _write_readme(force=force)
    _write_default_config(force=force)
    _write_example_microscope(force=force)


def _create_directories() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MICROSCOPES_DIR.mkdir(parents=True, exist_ok=True)


def _write_readme(force: bool = False) -> None:
    readme_path = CONFIG_DIR / "README.txt"
    if force or not readme_path.exists():
        readme_path.write_text(README, encoding="utf-8")


def _write_default_config(force: bool = False) -> None:
    if force or not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(DEFAULT_CONFIG)


def _write_example_microscope(force: bool = False) -> None:
    example_path = MICROSCOPES_DIR / "example.py"
    if force or not example_path.exists():
        example_path.write_text(EXAMPLE_MICROSCOPE)


# ---------------------------------------------------------------------------
# Reading config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.toml, falling back to defaults for any missing keys."""
    if not CONFIG_FILE.exists():
        user_config = tomllib.loads(DEFAULT_CONFIG)

    with open(CONFIG_FILE, "rb") as f:
        user_config = tomllib.load(f)

    return user_config


# ---------------------------------------------------------------------------
# Loading microscope profiles
# ---------------------------------------------------------------------------

def load_user_microscope_classes() -> None:
    """Import all .py files in the user microscopes directory so their
    classes register themselves as Movie subclasses."""

    microscopes = []

    for py_file in sorted(MICROSCOPES_DIR.glob("*.py")):
        if py_file.stem.startswith("example"):
            continue
        try:
            module_name = f"papylio.user_microscopes.{py_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            microscopes.append(py_file.stem)
        except Exception as e:
            print(
                f"[papylio] Warning: could not load microscope class "
                f"'{py_file.name}': {e}",
                file=sys.stderr,
            )

    print('Loaded microscopes:', str(microscopes).replace('[','').replace(']','').replace("'",""))

# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def open_config_dir() -> None:
    """Open the config directory in the system file manager."""
    import subprocess, os

    if sys.platform == "win32":
        os.startfile(CONFIG_DIR)
    elif sys.platform == "darwin":
        subprocess.run(["open", CONFIG_DIR])
    else:
        subprocess.run(["xdg-open", CONFIG_DIR])