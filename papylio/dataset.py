import numpy as np
import xarray as xr
import sys
import netCDF4
from functools import cached_property
import json

class Dataset:
    def __init__(self, path, root_path=None):
        self.path = path
        self._variables = None
        self._attributes = None
        self._exists = False

    # def create(self):
    #     xr.Dataset().to_netcdf(self.filepath, engine='netcdf4')

    def __repr__(self) -> str:
        """Return a string representation of the File object."""
        with xr.open_dataset(self.path) as ds:
            return repr(ds)

    def __contains__(self, name):
        if self.path.exists():
            return name in self.variable_names or name in self.attributes
        else:
            return False
    #

    # def __enter__(self):
    #     """Context manager entry point. Opens file for reading."""
    #     if self._with_counter == 0:
    #         self.ds = xr.open_dataset(self.path)
    #     self._with_counter += 1
    #
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """Context manager exit point. Closes file when exiting context."""
    #     self._with_counter -= 1
    #     if self._with_counter == 0:
    #         self.ds.close()

    def __getattr__(self, name):
        if not name.startswith('_') and name != 'path' and self.exists:
            if name in self.variable_names:
                with xr.open_dataset(self.path) as ds:
                    return ds[name].load()
            if name in self.attributes:
                return self.attributes[name]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}",
            name=name, obj=self,
        )

    @property
    def exists(self):
        if not self._exists:
            self._exists = self.path.exists()
        return self._exists

    @property
    def ds(self):
        return xr.load_dataset(self.path)

    @ds.setter
    def ds(self, value):
        value.to_netcdf(self.path, mode='a')
        self.refresh_variable_info()
        self._attributes = None

    @cached_property
    def variable_info(self):
        """Return a dictionary with information about all variables in the dataset."""
        info = {}
        with netCDF4.Dataset(self.path) as ds:
            for name in ds.variables:
                info[name] = {
                    'dimensions': ds[name].dimensions,
                    'shape': ds[name].shape,
                    'dtype': str(ds[name].dtype),
                    # 'attrs': dict(ds[name].ncattrs()),
                }
        return info

    def refresh_variable_info(self):
        self.__dict__.pop('variable_info', None)

    @property
    def variable_names(self):
        return list(self.variable_info.keys())
        # with netCDF4.Dataset(self.path) as ds:
        #     return tuple(ds.variables.keys())
        # with xr.open_dataset(self.path) as ds:
        #     return tuple(ds.variables)

    @property
    def coordinate_names(self):
        if self.exists:
            with netCDF4.Dataset(self.path) as ds:
                return set(ds.getncattr('coordinates').split(' ') if 'coordinates' in ds.ncattrs() else ())
        else:
            return set()

    @coordinate_names.setter
    def coordinate_names(self, value):
        for name in value:
            if not name in self.variable_names:
                raise ValueError(f"Coordinate {name} does not exist in dataset")
        with netCDF4.Dataset(self.path, 'a') as ds:
            ds.setncattr('coordinates', ' '.join(value))

    @property
    def attributes(self):
        if self._attributes is None:
            with netCDF4.Dataset(self.path) as ds:
                self._attributes = {name: ds.getncattr(name) for name in ds.ncattrs()}
            self._attributes = _decode_attributes(self._attributes)
        return self._attributes

    @attributes.setter
    def attributes(self, value):
        value = _encode_attributes(value)
        with netCDF4.Dataset(self.path, 'a') as ds:
            for name in ds.ncattrs():
                ds.delncattr(name)
            ds.setncatts(dict(value))
        self._attributes = None  # reload on next access

    def set_attribute(self, **attributes):
        attributes = _encode_attributes(attributes)
        with netCDF4.Dataset(self.path, 'a') as ds:
            ds.setncatts(attributes)
            self._attributes = None

    def set_coordinate(self, **kwargs):
        coordinate_names = self.coordinate_names
        xr.Dataset(coords=kwargs).to_netcdf(self.path, mode='a')
        self.refresh_variable_info()
        self.coordinate_names = coordinate_names.union(set(kwargs.keys()))

    # def set_variable(self, data, **kwargs):
    #     """
    #     Save data as a variable in the netCDF dataset.
    #
    #     Parameters:
    #         data (numpy.ndarray or xarray.DataArray): The data to save.
    #         **kwargs: Additional arguments for xarray.DataArray.
    #     """
    #     da = xr.DataArray(data, **kwargs)
    #     da.to_dataset().to_netcdf(self.path, mode='a')
    #     self.refresh_variable_names()

    def set_variable(self, overwrite=False, **kwargs):
        if not overwrite and any(name in self.variable_names for name in kwargs):
            raise ValueError("One or more variable names already exist. Use overwrite=True to overwrite existing variables.")
        xr.Dataset(kwargs).to_netcdf(self.path, mode='a')
        self.refresh_variable_info()

    def search_variable_names(self, search_string):
        return [name for name in self.variable_names if search_string in name]

    def get_variables(self, variable_names):
        return xr.Dataset({name: getattr(self, name) for name in variable_names})

    def search_variables(self, search_string):
        variable_names = self.search_variable_names(search_string)
        return self.get_variables(variable_names)

    def clear_variables(self, search_string):
        ds = self.ds
        dropped_variable_names = [name for name in self.variable_names if search_string in name]
        ds = ds.drop_vars(dropped_variable_names)
        ds.to_netcdf(self.path, mode='w')
        self.refresh_variable_info()

    def clear(self):
        self.path.unlink(missing_ok=True)
        self._variables = None
        self._attributes = None
        self._exists = False
        self.refresh_variable_info()



def _encode_attributes(attributes):
    encoded = {}
    for name, value in attributes.items():
        if isinstance(value, (np.ndarray, list)) and np.ndim(value) > 1:
            value = json.dumps(np.asarray(value).tolist())
        encoded[name] = value
    return encoded

def _decode_attributes(attributes):
    decoded = {}
    for name, value in attributes.items():
        if isinstance(value, str) and value.startswith('[['):
            try:
                value = np.array(json.loads(value))
            except (json.JSONDecodeError, ValueError):
                pass  # ordinary string, or ragged list: leave as is
        elif isinstance(value, np.generic):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        decoded[name] = value
    return decoded


class SingleMoleculeDataset(Dataset):
    def __init__(self, path, root_path=None):
        super().__init__(path, root_path)

    @property
    def trace_names(self):
        """Return names of data variables that are traces (have 'frame' as the last dimension)."""
        return [
            name for name, info in self.variable_info
            if info.dimensions[-1] == "frame"
        ]

    def get_traces(self, selected=False):
        """
        Get all data variables that have a 'frame' dimension (traces).

        Parameters:
            selected (bool, optional): Whether to return traces only for selected molecules. Default is False.

        Returns:
            xarray.Dataset: The traces dataset.
        """
        dataset = self.dataset

        traces = self.get_variables(self.trace_names)

        if selected:
            traces = traces.sel(molecule=dataset.selected)

        return traces
