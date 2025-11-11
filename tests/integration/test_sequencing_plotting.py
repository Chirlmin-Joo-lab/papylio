import numpy as np
import xarray as xr
from papylio.plugins.sequencing.plotting import double_mutations, plot_double_mutations
import pytest

@pytest.fixture
def general_sequence():
    return 'ACTGACTG'

@pytest.fixture
def simulated_data(general_sequence):
    dm = double_mutations(general_sequence, add_reference=True)
    da = xr.DataArray(np.arange(len(dm))[::-1], dims=('sequence',), coords={'sequence': dm})
    da.name = 'Test dataset'
    return da

def test_plot_double_mutations(general_sequence, simulated_data):
    # Without annotations
    plot_double_mutations(general_sequence, simulated_data)

    # With annotations, i.e. reporting the number
    plot_double_mutations(general_sequence, simulated_data, da_annotation=simulated_data)