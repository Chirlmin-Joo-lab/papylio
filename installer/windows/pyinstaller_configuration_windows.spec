# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
datas = [('..\\..\\papylio', 'papylio')]
binaries = []
hiddenimports = ['papylio', 'numpy', 'matplotlib', 'pathlib2', 'cv2', 'tabulate', 'scipy',
                  'skimage', 'skimage.transform', 'yaml', 'pandas', 'seaborn', 'nd2reader',
                  'xarray', 'netCDF4', 'h5netcdf', 'dask', 'bottleneck', 'tifffile', 'tqdm',
                  'PySide2', 'numba', 'matchpoint', 'objectlist', 'networkx']
for pkg in ['papylio', 'pomegranate', 'dask_image', 'marimo',
		    'starlette', 'uvicorn', 'websockets', 'click', 'jinja2', 'xarray', 'numpy', 'matplotlib',
		    'jupyterlab', 'jupyter_events', 'jupyterlab_server', 'jupyter_server', 'notebook', 'jupyter_client',
		    'jupyter_core', 'ipykernel', 'zqm', 'jupyter_lsp', 'notebook_shim']:
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