import pytest
import numpy as np
import xarray as xr
import sys

@pytest.fixture
def dataset(shared_datadir):
    return xr.open_dataset(shared_datadir / 'BN_TIRF_output_test_file' / 'TIRF 561 0001.nc' )

def test_trace_plot(dataset):
    from papylio.trace_plot import TracePlotWindow
    frame = TracePlotWindow(dataset)
