# import numpy as np #scientific computing with Python
# import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework
# import pandas as pd
# import xarray as xr
# from pathlib import Path
# from papylio.analysis.autoThreshold import stepfinder
# # from papylio.plugin_manager import PluginManager
# # from papylio.plugin_manager import PluginMetaClass
# from papylio.plugin_manager import plugins
# from papylio.trace_extraction import make_gaussian
# import copy
#
# class Molecules:
#     def load(filepath):
#         return Molecules().load(filepath)
#
#     def sum(list_of_molecules):
#         new = Molecules()
#         new.dataset = xr.concat([m.dataset for m in list_of_molecules], dim='molecule')
#         return new
#
#     def __init__(self, dataset=None):#, file, number_of_molecules):
#         # for value in molecules:
#         #     if not isinstance(value, Molecule):
#         #         raise TypeError('MoleculeList can only contain Molecule objects')
#         # self.molecules = molecules
#         # self.name = name
#
#         if dataset is None:
#             self.dataset = xr.Dataset()
#             self.init_dataset(reset=True)
#
#
#     # @property
#     # def dataset(self):
#     #     return self._dataset
#     #
#     # @dataset.setter
#     # def dataset(self, dataset):
#     #     if not self._dataset:
#     #         self._dataset = dataset
#     #         if 'selected' not in self._dataset.keys():
#     #             self.dataset['selected'] = xr.DataArray(False, coords=[dataset.molecule])
#     #         elif 'selected' not in self._dataset.keys():
#     #             self.dataset['x'] = xr.DataArray(False, coords=[dataset.molecule])
#     #             self.dataset['y'] = xr.DataArray(False, coords=[dataset.molecule])
#     #     else:
#     #         self._dataset = dataset
#
#     def init_dataset(self, molecule_multiindex=None, reset=False):
#         if not reset and len(self.dataset.molecule) > 0:
#             return
#         if molecule_multiindex is None:
#             molecule_multiindex = pd.MultiIndex.from_tuples([], names=['molecule_in_file', 'file'])
#         # self.dataset = xr.Dataset(
#         #     {
#         #         'selected':     ('molecule', xr.DataArray(False, coords=(molecule_multiindex,)),
#         #         'x':            ('molecule', xr.DataArray(np.nan, coords=[molecule_multiindex])),
#         #         'y':            ('molecule', xr.DataArray(np.nan, coords=[molecule_multiindex])),
#         #         'background':   ('molecule', xr.DataArray(np.nan, coords=[molecule_multiindex]))
#         #         'traces':       (('molecule'))
#         #     },
#         #     coords=
#         #     {
#         #         'molecule': ('molecule', molecule_multiindex), # pd.MultiIndex.from_tuples([], names=['molecule_in_file', 'file'])),
#         #         'frame': ('frame', np.array([], dtype=int)),
#         #         'channel': ('channel', np.array([], dtype=int))
#         #     }
#         # )
#
#         self.dataset = xr.Dataset(
#             {
#                 'selected':     ('molecule', xr.DataArray(False, coords=[molecule_multiindex])),
#                 'x':            (('molecule', 'channel'), xr.DataArray(np.nan, coords=[molecule_multiindex,[]])),
#                 'y':            (('molecule', 'channel'), xr.DataArray(np.nan, coords=[molecule_multiindex, []])),
#                 'background':   (('molecule','channel'), xr.DataArray(np.nan, coords=[molecule_multiindex, []])),
#                 'traces':       (('molecule', 'channel', 'frame'), xr.DataArray(np.nan, coords=[molecule_multiindex,[],[]]))
#             },
#             coords=
#             {
#                 'molecule': ('molecule', molecule_multiindex), # pd.MultiIndex.from_tuples([], names=['molecule_in_file', 'file'])),
#                 'frame': ('frame', np.array([], dtype=int)),
#                 'channel': ('channel', np.array([], dtype=int))
#             }
#         )
#
#         # self.empty_parameter_index = pd.MultiIndex.from_arrays([['is_selected'],[]], names=['Parameter','Channel'])
#         # self.empty_traces_index = pd.MultiIndex.from_arrays([[],[]], names=['Frame','Channel'])
#         # self._traces = None
#         # self._parameters = None
#         # self.channels = [0,1]
#
#     # @property
#     # def traces(self):
#     #     return self._traces
#     #
#     # @traces.setter
#     # def traces(self, traces):
#     #     if self._traces is not None and (len(traces.columns) != len(self.molecule_index)):
#     #         raise ValueError('Number of molecules does not match current number of molecules, to proceed reset the object first ')
#     #     self._traces = traces
#     #     self.molecule_index = traces.columns
#     #
#     # @property
#     # def parameters(self):
#     #     return self._parameters
#     #
#     # @parameters.setter
#     # def parameters(self, parameters):
#     #     if self._parameters is not None and (len(parameters.columns) != len(self.molecule_index)):
#     #         raise ValueError('Number of molecules does not match current number of molecules, to proceed reset the object first ')
#     #     self._parameters = parameters
#     #     self.molecule_index = parameters.columns
#     #
#     # @property
#     # def molecule_index(self):
#     #     if self._traces is not None:
#     #         return self.traces.columns
#     #
#     # @molecule_index.setter
#     # def molecule_index(self, molecule_index):
#     #     if self._traces is None:
#     #         self._traces = pd.DataFrame(index=self.empty_traces_index, columns=molecule_index)
#     #     else:
#     #         self._traces.columns = molecule_index
#     #     if self._parameters is None:
#     #         self._parameters = pd.DataFrame(index=self.empty_parameter_index, columns=molecule_index)
#     #     else:
#     #         self._parameters.columns = self.molecule_index
#     #
#     # def reset(self):
#     #     self._traces = None
#     #     self._parameters = None
#
#     def __len__(self):
#         return len(self.dataset.molecule)
#
#     def __getitem__(self, item):
#         new = Molecules()
#         new.dataset = self.dataset.isel(molecule=item)
#         return new
#
#     def __add__(self, other):
#         new = Molecules()
#         new.dataset = xr.concat([self.dataset, other.dataset], dim='molecule')
#         return new
#
#     def __radd__(self, other):
#         if other == 0:
#             return self
#         else:
#             return self.__add__(other)
#
#     def __getattr__(self, name):
#         try:
#             return self.dataset[name]
#         except KeyError:
#             super().__getattribute__(name)
#
#     # def __setattr__(self, key, value):
#     #     if key in self.__dict__.keys():
#     #         super().__setattr__(key, value)
#     #     else:
#     #         self._parameters.loc[key,:] = value
#
#     # def add_parameters(self, added_parameters):
#     #     parameters = self.parameters.append(added_parameters)
#     #     self.parameters = parameters[~parameters.index.duplicated(keep='last')]
#     #
#     # def add_parameter_from_list(self, name, parameter_list, channel=''):
#     #     self.parameters.loc[(name, channel), :] = parameter_list
#
#     @property
#     def coordinates(self):
#         return self.dataset[['x', 'y']].to_array('dimension')
#
#     @coordinates.setter
#     def coordinates(self, coordinates):
#         self.dataset = self.dataset.merge(coordinates.to_dataset('dimension'))
#
#     # def select_with_coords(self, kwargs):
#     #     self.dataset[] == 'MapSeq'
#
#     def import_file(self, filepath):
#         filepath = Path(filepath)
#         if filepath.suffix == '.traces':
#             self.import_traces_file(filepath)
#         elif filepath.suffix == '.pks':
#             self.import_pks_file(filepath)
#         else:
#             raise FileNotFoundError(filepath)
#
#     def save(self, filepath):
#         pass
#         # filepath = Path(filepath)
#         # self.dataset.reset_index('molecule').to_netcdf(filepath.with_suffix('.nc'))
#
#     def load(self, filepath):
#         pass
#         # filepath = Path(filepath)
#         # loaded_dataset = xr.open_dataset(filepath.with_suffix('.nc')).set_index({'molecule': ('molecule_in_file','file')})
#         # self.init_dataset(loaded_dataset.molecule)
#         # self.dataset = loaded_dataset.combine_first(self.dataset)
#
#
#
#
#     # column_index = pd.MultiIndex.from_product([[traces_filepath.with_suffix('').name],
#     #                                     np.arange(number_of_molecules),
#     #                                     np.arange(number_of_channels)],
#     #                                    names=['File', 'Molecule', 'Channel'])
#     # index = pd.Index(data=np.arange(number_of_frames), name='Frame')
#     # return pd.DataFrame(traces.T, index=index, columns=column_index).stack('Channel')
#
#
#
# # test.reset_index('trace').set_index(trace=['file','molecule','channel'])
# # test3.reset_index('trace').reset_coords('file', drop=True).assign_coords({'file': ('trace',[1]*278)}).set_index(trace=['file','molecule','channel'])
# # number_of_channels = 2
# # number_of_molecules = a.shape[a.dims=='trace']
# # traces = traces.unstack('Channel')
#
#
#
#
# @plugins
# class Molecule:
#     pass
# #     slots = ('file', 'index', '_coordinates', 'intensity', '_background', 'is_selected', 'steps', 'kon_boolean')
# #
# #     def __init__(self, file):
# #         self.file = file
# #         self.index = None
# #         self._coordinates = None
# #         self.intensity = None
# #         self._background = None
# #
# #         self.is_selected = False
# #
# #         self.steps = None  #Defined in other classes as: pd.DataFrame(columns=['frame', 'trace', 'state', 'method','thres'])
# #         self.kon_boolean = None  # 3x3 matrix that is indicates whether the kon will be calculated from the beginning, in-between molecules or for the end only
# #         #self.bg_scale=np.sum(make_gaussian(self.file.experiment.configuration['find_coordinates']['coordinate_optimization']['coordinates_after_gaussian_fit']['gaussian_width']))
# #
# #     @property
# #     def coordinates(self):
# #         return self._coordinates
# #
# #     @property
# #     def background(self):
# #         return self._background
# #
# #     @coordinates.setter
# #     def coordinates(self, coordinates):
# #         self._coordinates = np.atleast_2d(coordinates)
# #
# #     def background(self, background):
# #         self.background=background # should be dependent on emission channel as well
# #
# #     @property  # this is just for the stepfinder to be called through Molecule. Maybe not needed
# #     def find_steps(self):
# #         return stepfinder
# #
# #     def I(self, emission, Ioff=0):
# #         return self.intensity[emission, :] - Ioff # - self.background[emission] * self.bg_scale #this number comes from sum(make_gaussian) in trace_extraction
# #
# #     def E(self, Imin=0, Iroff=0, Igoff=0, alpha=0):
# #         red = np.copy(self.I(1, Ioff=Iroff))
# #         green = self.I(0, Ioff=Igoff)
# #         np.putmask(green, green < 0, 0) # green < 0 is taken as 0
# #         np.putmask(red, red < Imin, 0)  # the mask makes all elements of acceptor that are below the Imin zero, for E caclulation
# #         E =  (red - alpha*green) / (green + red - alpha*green)
# #         E = np.nan_to_num(E)  # correct for divide with zero = None values
# #         return E
# #
# #     def plot(self, ylim=(0, 500), xlim=(), Ioff=[],  save=False, **fretkwargs):
# #         plt.style.use('seaborn-dark')
# #         plt.style.use('seaborn-colorblind')
# #         figure = plt.figure(f'{self.file.name}_mol_{self.index}', figsize=(7,4))
# #         if len(self.file.experiment.pairs) > 0:
# #             axis_I = figure.add_subplot(211)
# #         else:
# #             axis_I = figure.gca()
# #
# #         axis_I.set_ylabel('Intensity (a.u.)')
# #         axis_I.set_ylim(ylim[0], ylim[1])
# #         if xlim == ():
# #             axis_I.set_xlim(0, self.file.time.max()+1)
# #         else:
# #             axis_I.set_xlim(xlim[0], xlim[1])
# #
# #         axis_I.set_title(f'Molecule {self.index} /{len(self.file.molecules)}')
# #         if Ioff == []:
# #             Ioff = [0]*self.file.number_of_channels
# #         for i, channel in enumerate(self.file.experiment.channels):
# #             axis_I.plot(self.file.time, self.I(i, Ioff=Ioff[i]), channel)
# #
# #         if len(self.file.experiment.pairs) > 0:
# #             axis_E = figure.add_subplot(212, sharex=axis_I)
# #             axis_E.set_xlabel('Time (s)')
# #             axis_E.set_ylabel('FRET')
# #             axis_E.set_ylim(0,1.1)
# #             for i, pair in enumerate(self.file.experiment.pairs):
# #                 axis_E.plot(self.file.time, self.E(**fretkwargs), 'b')
# #
# #         plt.tight_layout()
# #         if save:
# #             plt.savefig(f'{self.file.relativeFilePath}_mol_{self.index}.eps', transparent=True)
# #             plt.savefig(f'{self.file.relativeFilePath}_mol_{self.index}.png', facecolor='white', dpi=300, transparent=True)
# #
#
#
#
#
#
# if __name__ == '__main__':
#     filepath = r'D:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy\test'
#     # traces_filepath = r'P:\SURFdrive\Promotie\Code\Python\papylio\twoColourExampleData\20141017 - Holliday junction - Copy\test'
#     filepath = Path(filepath)
#
#     test = Molecules()
#     test.import_traces_file(filepath.with_suffix('.traces'))
#     test.import_pks_file(filepath.with_suffix('.pks'))
#
#     # pks_filepath = Path(traces_filepath).with_suffix('.pks')
#     # test.import_pks_file(pks_filepath, 2)
#     # test.export_pks_file(pks_filepath.with_name('test2.pks'))
#     #
#     #
#     # test[5:10]
#     #
#     # test[5]+test[10]
#
#
#     test.export_traces_file(filepath.with_name('test2.traces'))
#     test.export_pks_file(filepath.with_name('test2.pks'))