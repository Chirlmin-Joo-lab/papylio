import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import re
import gc
import copy
from tabulate import tabulate
from pathlib import Path
import pandas as pd
import logomaker
import matplotlib.path as pth
import xarray as xr

from trace_analysis.plugins.sequencing.plotting import plot_cluster_locations_per_tile


class SequencingData:

    reagent_kit_info = {'v2':       {'number_of_tiles': 14, 'number_of_surfaces': 2},
                        'v2_micro': {'number_of_tiles':  4, 'number_of_surfaces': 2},
                        'v2_nano':  {'number_of_tiles':  2, 'number_of_surfaces': 1},
                        'v3':       {'number_of_tiles': 19, 'number_of_surfaces': 2}}

    # @classmethod
    # def load(cls):
    #     cls()

    def __init__(self, file_path=None, dataset=None, name='', reagent_kit='v3'):
        if file_path is not None:
            file_path = Path(file_path)
            if file_path.suffix == '.nc':
                with xr.open_dataset(file_path.with_suffix('.nc'), engine='h5netcdf') as dataset:
                    self.dataset = dataset.load().set_index({'sequence': ('tile','x','y')})
            else:
                data = pd.read_csv(file_path, delimiter='\t')
                data.columns = data.columns.str.lower()
                data = data.set_index(['tile', 'x', 'y'])
                data = data.drop_duplicates() # TODO: It would be better to do this when converting the sam file.
                data.index.name = 'sequence' # Do this after drop_duplicates!
                self.dataset = xr.Dataset(data) #.reset_index('sequence', drop=True)

                if name == '':
                    name = Path(file_path).name

        elif dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError('Either file_path or data should be given')

        self.name = name

        self.reagent_kit = reagent_kit
        self.reagent_kit_info = SequencingData.reagent_kit_info[reagent_kit]

        self._tiles = None


    def __getattr__(self, item):
        #try:
        print('seqdata=' + item)
        if 'dataset' in self.__dict__.keys() and hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        #except AttributeError:
        #    super().__getattribute__(item)
        else:
            raise AttributeError
            # super(SequencingData, self).__getattribute__(item)

        try:
            return getattr(self.dataset, item)
        except AttributeError:
            super().__getattribute__(item)

    def __getitem__(self, item):
        return SequencingData(dataset=self.dataset.sel(sequence=item))

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

    @property
    def coordinates(self):
        return self.dataset[['x','y']].to_array(dim='dimension', name='coordinates').transpose('sequence',...)
            # xr.DataArray(self.data[['x','y']], dims=['sequence', 'dimension'], name='coordinates')\
            # .reset_index('sequence', drop=True)

    @property
    def tile_numbers(self):
        return np.unique(self.dataset.tile)

    @property
    def tiles(self):
        if not self._tiles:
            # Perhaps more elegant if this returns SequencingData objects [25-10-2021 IS]
            self._tiles = [Tile(tile, tile_coordinates) for tile, tile_coordinates in self.coordinates.groupby('tile')]
        return self._tiles

    def sel(self, *args, **kwargs):
        return SequencingData(dataset=self.dataset.sel(*args, **kwargs))

    def plot_cluster_locations_per_tile(self, save_filepath=None):
        plot_cluster_locations_per_tile(self.dataset[['x','y']], **self.reagent_kit_info, save_filepath=save_filepath)

    def save(self, filepath):
        self.dataset.reset_index('sequence').to_netcdf(filepath, engine='h5netcdf', mode='w')

class Tile:
    def __init__(self, number, coordinates):
        self.name = str(number)
        self.number = number
        self.coordinates = coordinates

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')


# # For sam file import
# samfile = pd.read_csv(r'O:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\Alignment.sam', delimiter='\t', usecols=[0,])
# samfile = samfile.iloc[3:]
# test = samfile['@SQ'].str.split(':', expand=True)
# test2 = test[[5,6]].astype(int)
#
# test3 = test[test[4]=='1101'][[5,6]].astype(int).to_numpy()
# tttt = test[test.duplicated()]


if __name__ == '__main__':
    file_path = r'J:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\sequencing_data_MapSeq.csv'
    seqdata = SequencingData(file_path=file_path)
    seqdata.plot_cluster_locations_per_tile(save_filepath=r'J:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\Mapping_seqquences_per_tile.png')

    file_path = r'J:\Ivo\20211011 - Sequencer (MiSeq)\Analysis\sequencing_data_HJ_general.csv'
    seqdata_HJ = SequencingData(file_path=file_path)