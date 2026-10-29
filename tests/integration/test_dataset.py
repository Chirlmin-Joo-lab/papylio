import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from tests.integration.temporary_data import temporary_data

# @pytest.fixture
# def dataset(shared_datadir):
#     from papylio.dataset import SingleMoleculeDataset
#     return SingleMoleculeDataset(shared_datadir / 'test.nc')

# def create_full_experiment(src: Path, files) -> Path:
#     shutil.copytree(src, dst, dirs_exist_ok=True)

@pytest.fixture
def dataset():
    relative_path = Path(r'Papylio example dataset - analyzed/ssHJ1/ssHJ1 TIRF 561 0001.nc')

    with temporary_data(relative_path) as path:
        from papylio.dataset import SingleMoleculeDataset
        dataset = SingleMoleculeDataset(path)
        # dataset = xr.load_dataset(path)
        yield dataset

def test_get_dataset(dataset):
    ds = dataset.dataset
    assert ds.equals(xr.load_dataset(dataset.path))

def test_set_dataset(dataset):
    ds = dataset.dataset
    dataset.dataset = ds
    assert dataset.dataset.equals(ds)

def test_variables(dataset):
    dataset.variables

def test_set_coordinate(dataset):
    dataset.set_coordinate(channel=['green','red'])

def test_set_variables(dataset):
    dataset.set_variable(xr.DataArray(np.arange(10), dims=('frame'), name='test'))

### Attributes ###
def test_set_attribute(dataset):
    dataset.set_attribute(rotation=1)
    assert dataset.attributes['rotation'] == 1

def test_set_attributes(dataset):
    attributes = dict(rotation=1, channel_arrangement=[[[0]]])
    dataset.attributes = attributes
    assert dataset.attributes == attributes
    assert dataset.rotation == 1
    assert dataset.channel_arrangement == np.array([[[0]]])