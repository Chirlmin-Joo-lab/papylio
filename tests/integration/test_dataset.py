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

@pytest.fixture
def dataset_empty():
    with temporary_data() as path:
        from papylio.dataset import SingleMoleculeDataset
        dataset = SingleMoleculeDataset(path / 'test.nc')
        yield dataset

def test_get_dataset(dataset):
    ds = dataset.ds
    assert ds.equals(xr.load_dataset(dataset.path))

def test_set_dataset(dataset):
    ds = dataset.ds
    dataset.ds = ds
    assert dataset.ds.equals(ds)

def test_variable_names(dataset):
    dataset.variable_names()

def test_set_coordinate(dataset_empty):
    dataset_empty.set_coordinate(channel=['green','red'])

def test_set_variable(dataset_empty):
    dataset_empty.set_variable(test=xr.DataArray(np.arange(10), dims=('frame')))

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

def test_groups(dataset_empty):
    dataset_empty.set_variable(test=xr.DataArray(np.arange(10), dims=('t')))
    dataset_empty.set_variable(test2=xr.DataArray(np.arange(10), dims=('t')), group='test_group')
    dataset_empty.structure

def test_clear_groups(dataset_empty):
    dt = xr.DataTree(xr.Dataset(dict(test=xr.DataArray(np.arange(10), dims=('t')))))
    dataset_empty.dt = dt
    dataset_empty.set_variable(test2=xr.DataArray(np.arange(10), dims=('t')), group='test_group')
    dataset_empty.clear_group('test_group')
    assert dataset_empty.dt.identical(dt)