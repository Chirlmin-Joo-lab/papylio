import pytest
import xarray as xr
import tifffile
import numpy as np
from PySide2.QtWidgets import QApplication
import sys
from multiprocessing import Process, freeze_support

from pathlib2 import Path

# print(__file__)

# sys.path.append(Path(__file__).parent.parent.parent)
from papylio.gui.main import MainWindow

def test_GUI(shared_datadir):
    print(shared_datadir)
    freeze_support()

    app = QApplication(sys.argv)

    window = MainWindow(shared_datadir / 'BN_TIRF')
    window.show()

    app.exec_()


