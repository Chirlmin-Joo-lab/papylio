import pytest
import xarray as xr
import tifffile
import numpy as np
from PySide6.QtWidgets import QApplication
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

    window = MainWindow(shared_datadir / 'Papylio example dataset - analyzed')
    window.show()

    app.exec_()

@pytest.fixture
def experiment_hj(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'Papylio example dataset - analyzed')

@pytest.fixture
def file_hj(experiment_hj):
    return experiment_hj.files.select('HJ')[0]

def test_trace_plot(file_hj):
    file_hj.show_traces()

def test_trace_plot_arguments_v0p9(file_hj):
    file_hj.show_traces(plot_variables=['intensity_total', 'intensity', 'FRET', 'classification'],
                        ylims=[(0, 35000), (0, 35000), (0, 1), (-2.5,1.5)],
                        colours=[('k'), ('g', 'r'), ('b'), ('k')], selected=False, height=5)

def test_trace_plot_two_illuminations(file_hj):
    ds = file_hj.dataset
    ds.illumination[:] = [0, ] * 100 + [1, ] * 200 + [0, ] * 100
    ds.classification_FRET.attrs['plot_settings'] = '{"active": true, "plot_range": [-0.05, 1.05], "color": ["b"]}'
    ds.to_netcdf(file_hj.absoluteFilePath.with_suffix('.nc'))
    file_hj.show_traces()

def test_classification_widget(file_hj):
    from papylio.gui.classification_widget import ClassificationWidget
    app = QApplication(sys.argv)
    win = ClassificationWidget()
    win.file = file_hj
    win.resize(850, 600)
    win.show()
    sys.exit(app.exec_())
