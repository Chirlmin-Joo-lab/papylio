import tqdm
import numpy as np
import netCDF4
from joblib import Parallel, delayed
from pathlib2 import Path


def get_dimension_size(filepath, dimension, with_selected_only=False, with_sequence_only=False):
    with netCDF4.Dataset(filepath) as ds:
        if with_selected_only:
            return ds['selected'][:].sum()
        elif with_sequence_only:
            # return (~(ds['sequence_aligned'][:] == b'-').all(axis=1)).sum()
            return (ds['sequence_tile'][:] > 0).sum()
        else:
            return ds.dimensions[dimension].size


def get_dimension_sizes(filepaths, dimension, with_selected_only=False, with_sequence_only=False):
    return Parallel(prefer="threads")(delayed(get_dimension_size)(filepath, dimension, with_selected_only, with_sequence_only)
                                      for filepath in filepaths)
    # return [get_dimension_size(filepath, 'molecule', with_sequence_only) for filepath in filepaths]


def merge_datasets(files_in, file_out, concat_dim, init_file=None, with_selected_only=False, with_sequence_only=False):
    # TODO: remove sequencing part, or move to the sequencing plugin
    if init_file is None:
        init_file = files_in[0]

    concat_dim_size = np.sum(get_dimension_sizes(files_in, concat_dim, with_selected_only, with_sequence_only))

    with netCDF4.Dataset(file_out, mode='w') as ds_out:
        with netCDF4.Dataset(init_file) as ds_in:
            init_dataset_like(ds_in, ds_out, concat_dim, concat_dim_size=concat_dim_size)

        start_index_out = 0
        for file_in in tqdm.tqdm(files_in):
            with netCDF4.Dataset(file_in) as ds_in:
                if with_selected_only:
                    _, start_index_out = append_to_dataset(ds_in, ds_out, concat_dim, selection_in=ds_in['selected'][:].astype(bool),
                                                           start_index_out=start_index_out)
                elif with_sequence_only:
                    # has_sequence = (~(ds_in['sequence_aligned'][:] == b'-').all(axis=1))
                    has_sequence = ds_in['sequence_tile'][:] > 0
                    _, start_index_out = append_to_dataset(ds_in, ds_out, concat_dim, selection_in=has_sequence, start_index_out=start_index_out)
                else:
                    _, start_index_out = append_to_dataset(ds_in, ds_out, concat_dim, selection_in=None, start_index_out=start_index_out)


def reorder_datasets_using_sequence_subset(files_in, folder_out, concat_dim):
    folder_out = Path(folder_out)
    folder_out.mkdir(exist_ok=False)
    for file_in in tqdm.tqdm(files_in):
        with netCDF4.Dataset(file_in) as ds_in:
            # selection = np.squeeze(ds['sequence_subset'][:].view('S8') != b'--------')
            sequence_subsets = ds_in['sequence_subset'][:]
            sequence_subsets.set_fill_value(b'-')
            sequence_subsets = sequence_subsets.filled()
            selection = (sequence_subsets != b'-').all(axis=1)
            indices = np.where(selection)[0]

            sequence_subsets = sequence_subsets.view(f'S{sequence_subsets.shape[1]}').astype('U').squeeze()

            for i in indices:
                sequence_subset = sequence_subsets[i]
                with netCDF4.Dataset(folder_out / (sequence_subset + '.nc'), mode='a') as ds_out:
                    if concat_dim not in ds_out.dimensions.keys():
                        init_dataset_like(ds_in, ds_out, concat_dim)
                    append_index_to_dataset(ds_in, ds_out, concat_dim, i)


def init_dataset_like(ds_in, ds_out, concat_dim, concat_dim_size=None):
    ds_out.createDimension(concat_dim, concat_dim_size)
    for name, dimension in ds_in.dimensions.items():
        if name != concat_dim:
            ds_out.createDimension(dimension.name, dimension.size)

    for name, variable in ds_in.variables.items():
        if '_FillValue' in variable.ncattrs():
            fill_value = variable.getncattr('_FillValue')
        else:
            fill_value = None

        ds_out.createVariable(variable.name, variable.datatype, dimensions=variable.dimensions,
                               fill_value=fill_value)

        for attr in variable.ncattrs():
            if attr != '_FillValue':
                ds_out[name].setncattr(attr, variable.getncattr(attr))

        if concat_dim not in variable.dimensions:
            ds_out[name][:] = ds_in[name][:]


def append_to_dataset(ds_in, ds_out, concat_dim, selection_in=None, start_index_out=None):
    if selection_in is None:
        selection_in = np.ones(ds_in.dimensions[concat_dim].size).astype(bool)
    if start_index_out is None:
        if ds_out.dimensions[concat_dim].isunlimited():
            start_index_out = ds_out.dimensions[concat_dim].size
        else:
            start_index_out = 0
    # end = start + ds.dimensions[concat_dim].size
    end_index_out = start_index_out + selection_in.sum()

    for name, variable in ds_in.variables.items():
        if concat_dim in variable.dimensions:
            if name == 'file':
                min_len = min(ds_in[name].shape[-1], ds_out[name].shape[-1])
                ds_out[name][start_index_out:end_index_out, :min_len] = ds_in[name][selection_in, :min_len]
            else:
                ds_out[name][start_index_out:end_index_out] = ds_in[name][selection_in]

    return start_index_out, end_index_out


def append_index_to_dataset(ds_in, ds_out, concat_dim, index_in):
    index_to = ds_out.dimensions[concat_dim].size
    for name, variable in ds_in.variables.items():
        if concat_dim in variable.dimensions:
            if name == 'file':
                min_len = min(ds_in[name].shape[-1], ds_out[name].shape[-1])
                ds_out[name][index_to, :min_len] = ds_in[name][index_in, :min_len]
            else:
                ds_out[name][index_to] = ds_in[name][index_in]
