import pytest
import xarray as xr
import tifffile
import numpy as np


@pytest.fixture
def experiment_output(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'BN_TIRF_output_test_file')

@pytest.fixture
def files(experiment_output):
    return experiment_output.files

def test_file_collection(files):
    files

def test_get_variable(files):
    assert isinstance(files.serial.get_variable('intensity'), xr.DataArray)

def test_show_histogram(files):
    files.show_histogram('intensity')
    files.show_histogram('FRET', bins=100, range=(0,1))

#TODO: Make new test dataset that does not contain background which has illumination as first dimension.
# def test_merge_datasets(files):
#     files.merge_datasets()
