import pytest
from papylio.plugins.sequencing.sequencing_data import parse_sam, fastq_generator, add_sequence_data_to_dataset, SequencingData


@pytest.fixture
def read1_fastq_filepath(shared_datadir):
    return shared_datadir / 'sequencing' / 'read1.fastq'


@pytest.fixture
def index1_fastq_filepath(shared_datadir):
    return shared_datadir / 'sequencing' / 'index1.fastq'


@pytest.fixture
def sam_filepath(shared_datadir):
    return shared_datadir / 'sequencing' / 'aligned.sam'


@pytest.fixture
def nc_filepath(sam_filepath):
    test_parse_sam(sam_filepath)
    return sam_filepath.with_suffix('.nc')


@pytest.fixture
def sequencing_data(nc_filepath):
    return SequencingData(file_path=nc_filepath)


def test_parse_sam(sam_filepath):
    extract_sequence_subset = [30, 31, 56, 57, 82, 83, 108, 109]
    parse_sam(sam_filepath, remove_duplicates=True, add_aligned_sequence=True, extract_sequence_subset=extract_sequence_subset,
              chunksize=10, write_csv=False, write_nc=True, write_filepath=None)


def test_fastq_generator(read1_fastq_filepath):
    fqg = fastq_generator(read1_fastq_filepath, 100)
    next(fqg)


def test_add_sequence_data_to_dataset(nc_filepath, index1_fastq_filepath):
    add_sequence_data_to_dataset(nc_filepath, index1_fastq_filepath, 'index1')


def test_sequencing_data_coordinates(sequencing_data):
    sequencing_data.coordinates


# def test_unlim_vs_lim_dim_netcdf4():
#     import netCDF4
#     import numpy as np
#
#     ds_unlim = netCDF4.Dataset(r'D:\Test\ds_unlim.nc', 'w')
#     ds_unlim.createDimension('test_dim', None)
#     ds_unlim.createVariable('test_var', np.uint32, ('test_dim',), chunksizes=(1,))
#     ds_unlim['test_var'][:] = np.arange(10 ** 6)
#     ds_unlim.close()
#
#     ds_lim = netCDF4.Dataset(r'D:\Test\ds_lim.nc', 'w')
#     ds_lim.createDimension('test_dim', 10**6)
#     ds_lim.createVariable('test_var', np.uint32, ('test_dim',))
#     ds_lim['test_var'][:] = np.arange(10 ** 6)
#     ds_lim.close()
#
#
#     import xarray as xr
#     import time
#     ds_unlim = xr.open_dataset(r'D:\Test\ds_unlim.nc')
#     start = time.time()
#     ds_unlim['test_var'].load()
#     print(time.time()-start)
#     ds_unlim.close()
#
#     ds_lim = xr.open_dataset(r'D:\Test\ds_lim.nc')
#     start = time.time()
#     ds_lim['test_var'].load()
#     print(time.time()-start)
#     ds_lim.close()