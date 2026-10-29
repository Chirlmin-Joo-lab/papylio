import numpy as np
import xarray as xr
import sys
import netCDF4
from functools import cached_property
import json

class SingleMoleculeDataset:
    def __init__(self, path, root_path=None):
        self.path = path
        self._variables = None
        self._attributes = None

    # def create(self):
    #     xr.Dataset().to_netcdf(self.filepath, engine='netcdf4')

    def __repr__(self) -> str:
        """Return a string representation of the File object."""
        return xr.open_dataset(self.path).__repr__()

    def __contains__(self, name):
        return name in self.variable_names or name in self.attributes
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
        if not name.startswith('_') and name != 'path':
            if name in self.variable_names:
                with xr.open_dataset(self.path) as ds:
                    return ds[name].load()
            elif name in self.attributes:
                return self.attributes[name]
        else:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}",
                name=name, obj=self,
            )

    @property
    def ds(self):
        return xr.load_dataset(self.path)

    @ds.setter
    def ds(self, value):
        value.to_netcdf(self.path, mode='a')
        self.refresh_variable_names()
        self._attributes = None

    @cached_property
    def variable_names(self):
        with netCDF4.Dataset(self.path) as ds:
            return tuple(ds.variables.keys())
        # with xr.open_dataset(self.path) as ds:
        #     return tuple(ds.variables)

    def refresh_variable_names(self):
        self.__dict__.pop('variables', None)

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
        xr.Dataset(coords=kwargs).to_netcdf(self.path, mode='a')
        self.refresh_variable_names()

    def set_variable(self, data, **kwargs):
        """
        Save data as a variable in the netCDF dataset.

        Parameters:
            data (numpy.ndarray or xarray.DataArray): The data to save.
            **kwargs: Additional arguments for xarray.DataArray.
        """
        da = xr.DataArray(data, **kwargs)
        da.to_dataset().to_netcdf(self.path, mode='a')
        self.refresh_variable_names()


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
        decoded[name] = value
    return decoded