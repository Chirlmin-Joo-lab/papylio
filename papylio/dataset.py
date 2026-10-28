import xarray as xr
import sys

class SingleMoleculeDataset:
    def __init__(self, filepath):
        self.filepath = filepath.with_suffix('.nc')

    # def create(self):
    #     xr.Dataset().to_netcdf(self.filepath, engine='netcdf4')

    def __repr__(self) -> str:
        """Return a string representation of the File object."""
        return xr.load_dataset(self.filepath, engine='netcdf4').__repr__()

    def open_folder(self) -> None:
        """Open the config directory in the system file manager."""
        import subprocess, os

        if sys.platform == "win32":
            os.startfile(self.filepath.parent)
        elif sys.platform == "darwin":
            subprocess.run(["open", self.filepath.parent])
        else:
            subprocess.run(["xdg-open", self.filepath.parent])

    def __getattr__(self, item):
        """
        Dynamically retrieve attributes, specifically looking into the netCDF dataset.

        Parameters:
            item (str): The name of the attribute to retrieve.

        Returns:
            The attribute value.

        Raises:
            AttributeError: If the attribute is not found.
        """
        if item == 'dataset_variables':
            return
        if item in self.dataset_variables or item.startswith('selection') or item.startswith('classification') or item.startswith('intensity'):
            with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
                try:
                    return dataset[item].load()
                except KeyError:
                    # It is desirable to raise an AttributeError instead of a KeyError,
                    # as this is used by hasattr for example. Hence the try except.
                    pass
        else:
            return super().__getattribute__(item)

    def set_global_attributes(self, **kwargs):
        pass

    def set_coordinate(self, **kwargs):
        xr.Dataset(coords=kwargs).to_netcdf(
            self.filepath, engine='netcdf4', mode='a'
        )

    def set_variable(self, data):
        """
        Save data as a variable in the netCDF dataset.

        Parameters:
            data (numpy.ndarray or xarray.DataArray): The data to save.
            **kwargs: Additional arguments for xarray.DataArray.
        """
        da = xr.DataArray(data, **kwargs)
        da.to_dataset().to_netcdf(self.filepath, engine='netcdf4', mode='a')
