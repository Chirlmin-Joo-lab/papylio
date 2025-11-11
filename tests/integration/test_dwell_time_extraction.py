import numpy as np
import pytest
import xarray as xr
from pytest_datadir.plugin import shared_datadir

from papylio.analysis.dwell_time_extraction import dwell_times_from_classification

@pytest.fixture
def dwells(shared_datadir):
    return xr.load_dataset(shared_datadir / 'Papylio example dataset - analyzed' / 'ssHJ1' / 'ssHJ1 TIRF 561 0001_dwells.nc')