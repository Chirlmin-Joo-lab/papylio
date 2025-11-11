import pytest
from PySide2 import QtWidgets
import sys
from papylio.trace_correction import TraceCorrectionWindow


@pytest.fixture
def experiment(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'BN_TIRF_output_test_file')


@pytest.fixture
def file(experiment):
    return experiment.files[0]


def test_trace_plot(file):
    # app = QtWidgets.QApplication.instance()
    # if not app:
    #     app = QtWidgets.QApplication(sys.argv)
    frame = TraceCorrectionWindow(file.intensity)
    # frame.show()
    # frame.activateWindow()
    # frame.raise_()
    # app.exec_()
