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

@pytest.fixture
def experiment_hj(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'Papylio example dataset - analyzed')

@pytest.fixture
def file_hj(experiment_hj):
    return experiment_hj.files.select('HJ')[0]

def test_classification_widget(file_hj):
    from papylio.gui.classification_widget import ClassificationWidget
    app = QApplication(sys.argv)
    win = ClassificationWidget()
    win.file = file_hj
    win.resize(850, 600)
    win.show()
    sys.exit(app.exec_())
