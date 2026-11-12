# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
REPO_ROOT = os.path.abspath(os.path.join(SPEC_DIR, '..', '..'))
version_file = os.path.join(REPO_ROOT, 'papylio', '_version.py')

import papylio
version_str = papylio.get_version()

print(f"Current working directory: {os.getcwd()}", flush=True)
print(f"Spec file: {SPEC}", flush=True)
print(f"Repository root: {REPO_ROOT}", flush=True)
print(f"Papylio imported from: {papylio.__file__}", flush=True)
print(f"Version to write: {version_str}", flush=True)
print(f"Writing version file: {version_file}", flush=True)

with open(version_file, 'w') as f:
    f.write(f'version = "{version_str}"\n')

print(f"Version file exists: {os.path.isfile(version_file)}", flush=True)
with open(version_file, encoding='utf-8') as f:
    print(f"Version file contents: {f.read().strip()}", flush=True)

# Ensure datas uses the correct absolute path
datas = [
    (os.path.join(REPO_ROOT, 'papylio'), 'papylio'),
    (version_file, 'papylio')
]

# Explicitly include the generated file in datas
# datas = [
#    ('..\\..\\papylio', 'papylio'),
#    (version_file, 'papylio')  # This forces _version.py into the papylio folder
#]

binaries = []
hiddenimports = ['papylio', 'numpy', 'matplotlib', 'pathlib2', 'cv2', 'tabulate', 'scipy',
                  'skimage', 'skimage.transform', 'yaml', 'pandas', 'seaborn', 'nd2reader',
                  'xarray', 'netCDF4', 'h5netcdf', 'dask', 'bottleneck', 'tifffile', 'tqdm',
                  'PySide2', 'numba', 'matchpoint', 'objectlist', 'networkx']

marimo_packages = set(['marimo', 'click', 'jedi', 'markdown', 'pymdownx', 'tomlkit', 'yaml', 'uvicorn',
                       'starlette', 'multipart', 'websockets', 'loro', 'typing_extensions', 'docutils',
                       'psutil', 'itsdangerous', 'narwhals', 'packaging', 'msgspec', 'zmq', 'nbformat'])

jupyter_packages = set(['jupyterlab', 'jupyter_events', 'jupyterlab_server', 'notebook', 'jupyter_client', 'zmq',
		                'jupyter_builder', "async_lru", "httpx", "ipykernel", "jinja2", "jupyter_core", "jupyter_server",
                        "jupyter_lsp", "jupyterlab_server", "notebook_shim", "packaging", "tomli", "tornado", "traitlets",
                        "typing_extensions"])

papylio_packages = set(['papylio', 'pomegranate', 'dask_image', 'xarray', 'numpy', 'matplotlib'])

packages = papylio_packages.union(marimo_packages).union(jupyter_packages)

for pkg in packages:
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]
	
a = Analysis(
    ['..\\..\\papylio\\gui\\start.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='papylio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['..\\..\\papylio\\gui\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='papylio',
)