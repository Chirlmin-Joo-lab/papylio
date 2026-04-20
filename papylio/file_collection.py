"""Collection container for managing multiple data files.

Provides FileCollection class which extends ObjectList to handle batch operations
on multiple File objects, including method delegation and automatic result concatenation.
"""
from objectlist import ObjectList

from papylio.file import File
from papylio.netcdf_operations import merge_datasets, reorder_datasets_using_sequence_subset

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# TODO: Make sure that the collection can only contain File objects
# TODO: perhaps add file info method that generates a table with info of the file, for example number of molecules, etc.
class FileCollection(ObjectList):
    def __getattr__(self, item):
        """Delegate attribute access to individual files and concatenate xarray results if applicable."""
        # TODO: Replace this method by map?
        attrs = super(FileCollection, self).__getattr__(item)
        if callable(attrs):  # Test if attrs is a function, if this is the case add another function to concatenate possible xarray objects.
            def f2(*args, **kwargs):
                output = attrs(*args, **kwargs)
                if output is not None and isinstance(output[0], xr.DataArray):
                    output = xr.concat(output, dim=output[0].dims[0])
                return output
            return f2

        elif isinstance(attrs[0], xr.DataArray):
            attrs = xr.concat(attrs, dim=attrs[0].dims[0])
        elif isinstance(attrs[0], xr.Dataset):
            if 'molecule' in attrs[0].dims:
                attrs = xr.concat(attrs, dim='molecule')
            elif item == 'dwells':
                #TODO is this still necessary or is this done by the xr.DataArray elif?
                attrs = xr.concat(attrs, dim='dwell')
        return attrs

    @property
    def experiment(self):
        """Return the experiment object from the first file."""
        return self[0].experiment

    def select(self, search_string, variable='relative_filepath'):
        """Select files matching the search string on the specified variable using regex."""
        # TODO: Make this accept keyword arguments where the key is the variable and the item is the search_string.
        # TODO: Make this accept mulitple keyword arguments.
        return self[getattr(self, variable).str.regex(search_string)]

    def show_histogram(self, *args, **kwargs):
        """Display histogram for all files combined."""
        figure, axis = File.show_histogram(self.serial, *args, **kwargs)
        axis.set_title('')
        return figure, axis

    def histogram_FRET_intensity_total(self, **kwargs):
        """Display FRET vs intensity histogram for all files."""
        File.histogram_FRET_intensity_total(self, **kwargs)

    @property
    def cycle_time(self):
        """Return the cycle time from the first file."""
        return self[0].cycle_time

    # def analyze_dwells_combined(self, *args, **kwargs):
    #     return File.analyze_dwells(self.serial, *args, **kwargs)

    def print(self):
        """Print a formatted list of all files in the collection."""
        for i, file in enumerate(self):
            print(f"{i:3d}.  {file.relative_filepath}")

    def merge_datasets(self, filepath_out=None, init_file_index=0, with_selected_only=False, with_sequence_only=False):
        """Merge datasets from the collection into a single netCDF file."""
        #TODO: remove sequencing part, or move to the sequencing plugin
        if filepath_out is None:
            filepath_out = self[0].absolute_filepath.parent / 'merged_dataset.nc'
        filepaths_in = self.serial.absolute_filepath.with_suffix('.nc')
        merge_datasets(filepaths_in, filepath_out, concat_dim='molecule', init_file=filepaths_in[init_file_index],
                       with_selected_only=with_selected_only, with_sequence_only=with_sequence_only)

    def reorder_datasets_using_sequence_subset(self, folderpath_out):
        """Reorder datasets using sequence subset information."""
        files = self[self.has_sequencing_match]
        filepaths_in = files.serial.absolute_filepath.with_suffix('.nc')
        reorder_datasets_using_sequence_subset(filepaths_in, folderpath_out, concat_dim='molecule')