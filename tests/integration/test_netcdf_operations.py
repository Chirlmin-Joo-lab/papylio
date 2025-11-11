import pytest
import numpy as np
from papylio.netcdf_operations import merge_datasets, reorder_datasets_using_sequence_subset
import netCDF4

@pytest.fixture
def netcdf_filepaths(shared_datadir):
    return list((shared_datadir / 'netcdf_operations').glob('*.nc'))


def test_merge_datasets(shared_datadir, netcdf_filepaths):
    merge_datasets(netcdf_filepaths, shared_datadir / 'file_out.nc', 'molecule', init_file=None, with_sequence_only=False)

    ds_out = netCDF4.Dataset(shared_datadir / 'file_out.nc')
    assert ds_out.dimensions['molecule'].size == np.sum([netCDF4.Dataset(p).dimensions['molecule'].size for p in netcdf_filepaths])
    assert (ds_out['intensity'][1707:3331] == netCDF4.Dataset(netcdf_filepaths[1])['intensity'][:]).all()


def test_merge_datasets_sequence_only(shared_datadir, netcdf_filepaths):
    merge_datasets(netcdf_filepaths, shared_datadir / 'file_out.nc', 'molecule', init_file=None, with_sequence_only=True)

    ds_out = netCDF4.Dataset(shared_datadir / 'file_out.nc')
    assert ds_out.dimensions['molecule'].size == np.sum([(netCDF4.Dataset(p)['sequence'][:]!=b'-').all(axis=1).sum() for p in netcdf_filepaths])
    ds_in1 = netCDF4.Dataset(netcdf_filepaths[1])
    assert (ds_out['intensity'][319:319+318] == ds_in1['intensity'][(ds_in1['sequence'][:]!=b'-').all(axis=1)]).all()

def test_reorder_datasets_using_sequence_subset(shared_datadir, netcdf_filepaths):
    reorder_datasets_using_sequence_subset(netcdf_filepaths, shared_datadir, 'molecule')




# def test_merge_datasets():
#     from pathlib2 import Path
#     shared_datadir = Path(r'H:\My Documents\Python\traceanalysis\tests\integration\data\netcdf_operations')
#     netcdf_filepaths = list(shared_datadir.glob('*.nc'))
#     merge_datasets(netcdf_filepaths, shared_datadir / 'file_out.nc', 'molecule', init_file=None, with_sequence_only=False)
#     assert 1 == 1
