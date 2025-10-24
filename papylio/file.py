if __name__ == '__main__':
    import sys
    from pathlib import Path
    p = Path(__file__).parents[1]
    sys.path.insert(0, str(p))

from pathlib import Path # For efficient path manipulation
import numpy as np #scientific computing with Python
import pandas as pd
import matplotlib.pyplot as plt #Provides a MATLAB-like plotting framework
import skimage.io as io
import xarray as xr
import skimage as ski
import warnings
import sys
import re
import logging
import inspect
import tifffile
import netCDF4
import json
import papylio
# from papylio.molecule import Molecule
from papylio.movie.movie import Movie
from papylio.movie.tif import TifMovie
from papylio.plotting import histogram
import matchpoint as mp
from papylio.peak_finding import find_peaks
from papylio.coordinate_optimization import  coordinates_within_margin, \
                                                    coordinates_after_gaussian_fit, \
                                                    coordinates_without_intensity_at_radius, \
                                                    merge_nearby_coordinates, \
                                                    set_of_tuples_from_array, array_from_set_of_tuples, \
                                                    coordinates_within_margin_selection
from papylio.trace_extraction import extract_traces
from papylio.log_functions import add_configuration_to_dataarray
# from matchpoint.coordinate_transformations import translate, transform # MD: we don't want to use this anymore I think, it is only linear
                                                                           # IS: We do! But we just need to make them usable with the nonlinear mapping
from papylio.background_subtraction import extract_background
from papylio.analysis.hidden_markov_modelling import hmm_traces, classify_hmm
# from papylio.plugin_manager import PluginManager
# from papylio.plugin_manager import PluginMetaClass
from papylio.plugin_manager import plugins
# from papylio.trace_plot import TraceAnalysisFrame
from papylio.analysis.dwell_time_extraction import dwell_times_from_classification
from papylio.analysis.dwell_time_analysis import analyze_dwells, plot_dwell_time_histogram, plot_dwell_analysis
from papylio.decorators import return_none_when_executed_by_pycharm

@plugins
class File:
    # plugins = []
    # _plugin_mixin_class = None
    #
    # @classmethod
    # def add_plugin(cls, plugin_class):
    #     cls.plugins.append(plugin_class)
    #     cls._plugin_mixin_class = type(cls.__name__, (cls,) + tuple(cls.plugins), {})
    #
    # def __new__(cls, *args, **kwargs):
    #     if not cls._plugin_mixin_class:
    #         return super().__new__(cls)
    #     else:
    #         return super().__new__(cls._plugin_mixin_class)

    def __init__(self, relativeFilePath, extensions=None, experiment=None):
        self.dataset_variables = ['molecule', 'frame', 'time', 'coordinates', 'background', 'intensity', 'FRET', 'selected',
                                  'molecule_in_file', 'illumination_correction', 'number_of_states', 'transition_rate', 'state_mean', 'classification']

        relativeFilePath = Path(relativeFilePath)
        self.experiment = experiment

        self.relativePath = relativeFilePath.parent
        self.name = relativeFilePath.name
        self.extensions = set()

        # self.molecules = Molecules()

        self.exposure_time = None  # Found from log file or should be inputted

        self.log_details = None  # a string with the contents of the log file
        self.number_of_frames = None

        self.isSelected = False
        self.is_mapping_file = False

        self.movie = None
        self.mapping = None


        # I think it will be easier if we have import functions for specific data instead of specific files.
        # For example. the sifx, pma and tif files can better be handled in the Movie class. Here we then just have a method import_movie.
        # [IS 10-08-2020]
        # TODO: Make an import_movie method and move the specific file type handling to the movie class (probably this should also include the log file)
        # TODO: Make an import_mapping method and move the specific mapping type handling (.map, .coeff) to the mapping class.

        self.importFunctions = {'.sifx': self.import_movie,
                                '.pma': self.import_movie,
                                '.nd2': self.import_movie,
                                '.tif': self.import_movie,
                                '.tiff': self.import_movie,
                                '.TIF': self.import_movie,
                                '.TIFF': self.import_movie,
                                '.bin': self.import_movie,
                                '.coeff': self.import_coeff_file,
                                '.map': self.import_map_file,
                                '.mapping': self.import_mapping_file,
                                '.pks': self.import_pks_file,
                                '.traces': self.import_traces_file,
                                '_steps_data.xlsx': self.import_excel_file,
                                '.nc': self.noneFunction
                                }

        # print(self)

        if extensions is None:
            extensions = self.find_extensions()
        self.add_extensions(extensions, load=self.experiment.import_all)

        self.logger = self._create_logger()
        self.logger.info(f"Initialized {self} with Papylio v{papylio.__version__}")

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.relativePath.joinpath(self.name)})')

    def _create_logger(self):
        """Create a dedicated logger per File instance."""
        logger_name = f"FileLogger.{self.relativeFilePath}"
        logger = logging.getLogger(logger_name)

        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_file = self.absoluteFilePath.with_suffix(".log")
            handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @property
    @return_none_when_executed_by_pycharm
    def relativeFilePath(self):
        return self.relativePath.joinpath(self.name)

    @property
    @return_none_when_executed_by_pycharm
    def absoluteFilePath(self):
        return self.experiment.main_path.joinpath(self.relativeFilePath)

    @property
    @return_none_when_executed_by_pycharm
    def number_of_molecules(self):
        # return len(self.dataset.molecule)
        # if self.absoluteFilePath.with_suffix('.nc').exists():
        try:
            # with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
            #     return len(dataset.molecule)
            with netCDF4.Dataset(self.absoluteFilePath.with_suffix('.nc')) as dataset:
                return dataset.dimensions['molecule'].size
        except FileNotFoundError:
            return 0

    @property
    @return_none_when_executed_by_pycharm
    def configuration(self):
        return self.experiment.configuration

    # @property
    # def molecule(self):
    #     with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
    #         return dataset['molecule'].load()
    # @number_of_molecules.setter
    # def number_of_molecules(self, number_of_molecules):
    #     if not self.molecules:
    #         for molecule in range(0, number_of_molecules):
    #             self.addMolecule()
    #     elif number_of_molecules != self.number_of_molecules:
    #         raise ValueError(f'Requested number of molecules ({number_of_molecules}) differs from existing number of '
    #                          f'molecules ({self.number_of_molecules}) in {self}. \n'
    #                          f'If you are sure you want to proceed, empty the molecules list file.molecules = [], or '
    #                          f'possibly delete old pks or traces files')

    @property
    @return_none_when_executed_by_pycharm
    def number_of_channels(self):
        return self.experiment.number_of_channels

    @property
    @return_none_when_executed_by_pycharm
    def selected_molecules(self):
        return self.molecule[self.selected]

    @property
    @return_none_when_executed_by_pycharm
    def number_of_selected_molecules(self):
        return len(self.selected_molecules)

    # def get_projection_image(self, configuration):

    # @property
    def projection_image(self):
        return self.get_projection_image()

    # @property
    def average_image(self):
        return self.get_projection_image(projection_type='average')

    # @property
    def maximum_projection_image(self):
        return self.get_projection_image(projection_type='maximum')

    def get_projection_image(self, load=True, **kwargs):
        configuration = self.experiment.configuration['projection_image'].copy()
        configuration.update(kwargs)
        # TODO: Make this independent of Movie, probably we want to copy all Movie metadata to the nc file.
        if configuration['frame_range'][1] is None:
            configuration['frame_range'] = (configuration['frame_range'][0], self.movie.number_of_frames)
        elif configuration['frame_range'][1] > self.movie.number_of_frames:
            configuration['frame_range'] = (configuration['frame_range'][0], self.movie.number_of_frames)
            warnings.warn(f'Frame range exceeds available frames, used frame range {configuration["frame_range"]} instead')
        image_filename = Movie.image_info_to_filename(self.name, **configuration)
        image_file_path = self.absoluteFilePath.with_name(image_filename).with_suffix('.tif')

        if load and image_file_path.is_file():
            # TODO: Make independent of movie, so that we can also load this without movie present
            # Perhaps make it part of Movie
            # Perhaps make a get_projection_image a class method of Movie
            # return self.movie.separate_channels(tifffile.imread(image_file_path))
            return tifffile.imread(image_file_path)
        else:
            return self.movie.make_projection_image(**configuration, write=True, flatten_channels=True)

    # @property
    # def coordinates(self):
    #     with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
    #         #.set_index({'molecule': ('molecule_in_file','file')})
    #         return dataset['coordinates'].load()
    #
    # @coordinates.setter

    @property
    @return_none_when_executed_by_pycharm
    def coordinates_metric(self):
        return self.coordinates * self.movie.pixel_size

    @property
    @return_none_when_executed_by_pycharm
    def coordinates_stage(self):
        coordinates = self.coordinates.sel(channel=0)
        # coordinates = coordinates.stack(temp=('molecule', 'channel')).T
        coordinates_stage = self.movie.pixel_to_stage_coordinates_transformation(coordinates)
        return xr.DataArray(coordinates_stage, coords=coordinates.coords)

    def set_coordinates_of_channel(self, coordinates, channel):
        # TODO: make this usable for more than two channels
        # TODO: Make this work for xarray DataArrays
        channel_index = self.movie.get_channel_number(channel)  # Or possibly make it self.channels
        if channel_index == 0:
            coordinates_in_main_channel = coordinates
        elif channel_index == 1:
            coordinates_in_main_channel = self.mapping.transform_coordinates(coordinates, inverse=True)
        else:
            raise NotImplementedError('File.set_coordinates_of_channel not implemented for more than two channels')

        # for channel in self.channels or for mapping in self.mappings
        coordinates_list = [coordinates_in_main_channel]
        for i in range(1, self.number_of_channels):
            if self.number_of_channels > 2:
                raise NotImplementedError('File.set_coordinates_of_channel not implemented for more than two channels')
            coordinates_in_other_channel = self.mapping.transform_coordinates(coordinates_in_main_channel, direction='Donor2Acceptor')
            coordinates_list.append(coordinates_in_other_channel)
        coordinates = np.hstack(coordinates_list).reshape((-1, 2))

        self.coordinates = coordinates

    def coordinates_from_channel(self, channel):
        # if not self._pks_file:
        #     _pks_file = PksFile(self.absoluteFilePath.with_suffix('.pks'))

        #return np.concatenate([[molecule.coordinates[0, :] for molecule in self.molecules]])
        if type(channel) is str:
            channel = {'d': 0, 'a': 1, 'g':0, 'r':1}[channel]

        return self.coordinates.sel(channel=channel)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, dict):
        self.__dict__.update(dict)

    def __getattr__(self, item):
        if item == 'dataset_variables':
            return
        if item in self.dataset_variables or item.startswith('selection') or item.startswith('classification') or item.startswith('intensity'):
            with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
                try:
                    return dataset[item].load()
                except KeyError:
                    # It is desirable to raise an AttributeError instead of a KeyError,
                    # as this is used by hasattr for example. Hence the try except.
                    pass
        # else:
        #     super().__getattribute__(item)
        raise AttributeError(f'Attribute {item} not found')

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # Skip logger itself
        if name != "logger" and hasattr(self, "logger"):
            # Check if the assignment comes from outside this instance
            stack = inspect.stack()
            external = all(frame.frame.f_locals.get("self") is not self for frame in stack[1:])
            if external:
                self.logger.info(f"Set attribute {name} = {value!r}")


    def get_data(self, key):
        with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return dataset[key].load()

    @property
    @return_none_when_executed_by_pycharm
    def dataset(self):
        if self.absoluteFilePath.with_suffix('.nc').exists():
            with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
                return dataset.load()
        else:
            return None

    @property
    @return_none_when_executed_by_pycharm
    def dataset_selected(self):
        dataset = self.dataset
        return dataset.sel(molecule=dataset.selected)

    @property
    @return_none_when_executed_by_pycharm
    def data_vars(self):
        with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return dataset.data_vars

    @property
    @return_none_when_executed_by_pycharm
    def dataset_attributes(self):
        if self.absoluteFilePath.with_suffix('.nc').exists():
            with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
                return dataset.attrs
        else:
            return {}

    # def get_coordinates(self, selected=False):
    #     if selected:
    #         molecules = self.selectedMolecules
    #     else:
    #         molecules = self.molecules
    #
    #     return np.vstack([molecule.coordinates for molecule in molecules])

    #in analogy with coordinates, also background:
    # @background.setter
    # def background(self, background, number_of_channels = None):
    #     if number_of_channels is None:
    #         number_of_channels = self.number_of_channels
    #     self.number_of_molecules = np.shape(background)[0]//number_of_channels
    #
    #     for i, molecule in enumerate(self.molecules):
    #         molecule.background = background[(i * number_of_channels):((i + 1) * number_of_channels)]
    #
    # def background_from_channel(self, channel):
    #     if type(channel) is str:
    #         channel = {'d': 0, 'a': 1, 'g':0, 'r':1}[channel]
    #
    #     return np.vstack([molecule.background[channel] for molecule in self.molecules])
    #

    def _init_dataset(self, number_of_molecules):
        selected = xr.DataArray(False, dims=('molecule',), coords={'molecule': range(number_of_molecules)}, name='selected')
        add_configuration_to_dataarray(selected)
        selected.attrs['configuration'] = json.dumps([])
        selected.attrs['selection_configurations'] = json.dumps({})

        # dataset = selected.reset_index('molecule').rename(_molecule='molecule_in_file').to_dataset()
        dataset = selected.to_dataset().assign_coords(molecule_in_file=('molecule', selected.molecule.values))
        dataset = dataset.reset_index('molecule', drop=True)
        dataset = dataset.assign_coords({'file': ('molecule', [str(self.relativeFilePath).encode()] * number_of_molecules)})
        encoding = {'file': {'dtype': '|S'}, 'selected': {'dtype': bool}}
        dataset.attrs['channel_arrangement'] = json.dumps(self.movie.channel_arrangement.tolist())
        dataset.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='w', encoding=encoding)
        self.extensions.add('.nc')

        # # pd.MultiIndex.from_tuples([], names=['molecule_in_file', 'file'])),

    def find_extensions(self):
        file_names = [file.name for file in self.experiment.main_path.joinpath(self.relativePath).glob(self.name + '*')]
        extensions = [file_name[len(self.name):] for file_name in file_names]
        # For the special case of a sifx file, which is located inside a folder
        if '' in extensions:
            extensions[extensions.index('')] = '.sifx'
        # elif 'fov' in self.name:
        #     # TODO: Check whether this works
        #     token_position = self.name.find('_fov')
        #     file_keyword = self.name[:token_position]
        #     if self.absoluteFilePath.with_name(file_keyword).with_suffix('.nd2').is_file():
        #         extensions.append('.nd2')
        return extensions

    def find_and_add_extensions(self):
        self.add_extensions(self.find_extensions())

    def add_extensions(self, extensions, load=True):
        if isinstance(extensions, str):
            extensions = [extensions]
        for extension in set(extensions)-self.extensions:
            if load:
                self.importFunctions.get(extension, self.noneFunction)(extension)
            if extension in self.importFunctions.keys():
                self.extensions.add(extension)
        # or self.extensions = self.extensions | extensions

    def noneFunction(self, *args, **kwargs):
        return

    def import_movie(self, extension):
        if extension == '.sifx':
            filepath = self.absoluteFilePath.joinpath('Spooled files.sifx')
        # elif extension == '.nd2' and '_fov' in self.name:
        #     # TODO: Make this working
        #     token_position = self.name.find('_fov')
        #     movie_name = self.name[:token_position]
        #     filepath = self.absoluteFilePath.with_name(movie_name).with_suffix(extension)
            # self.movie = ND2Movie(imageFilePath, fov_info=self.nd2_fov_info)
        else:
            filepath = self.absoluteFilePath.with_suffix(extension)

        rot90 = self.configuration['movie']['rot90']
        self.movie = Movie(filepath, rot90)
        if 'channel_arrangement' in self.dataset_attributes.keys():
            self.movie.channel_arrangement = self.dataset_attributes['channel_arrangement']
        # self.number_of_frames = self.movie.number_of_frames

    def import_coeff_file(self, extension):
        from skimage.transform import AffineTransform
        if self.mapping is None: # the following only works for 'linear'transformation_type
            file_content=np.genfromtxt(str(self.absoluteFilePath) + '.coeff')
            if len(file_content)==12:
                [coefficients, coefficients_inverse] = np.split(file_content,2)
            elif len(file_content)==6:
                coefficients = file_content
            else:
                raise TypeError('Error in importing coeff file, wrong number of lines')

            self.mapping = mp.MatchPoint(transformation_type='linear')

            transformation = np.zeros((3,3))
            transformation[2,2] = 1
            transformation[[0,0,0,1,1,1],[2,0,1,2,0,1]] = coefficients
            self.mapping.transformation = AffineTransform(matrix=transformation)

            if len(file_content)==6:
                self.mapping.transformation_inverse=\
                    AffineTransform(matrix=np.linalg.inv(self.mapping.transformation.params))
            else:
                transformation_inverse = np.zeros((3,3))
                transformation_inverse[2,2] = 1
                transformation_inverse[[0,0,0,1,1,1],[2,0,1,2,0,1]] = coefficients_inverse
                self.mapping.transformation_inverse = AffineTransform(matrix=transformation_inverse)

            self.mapping.file = self
            self.mapping.source_name = 'Donor'
            self.mapping.destination_name = 'Acceptor'

    def export_coeff_file(self):
        warnings.warn('The export_coeff_file method will be depricated, use export_mapping instead')
        self.export_mapping(filetype='classic')

    def export_mapping(self, filetype='yml'):
        self.mapping.save(self.absoluteFilePath, filetype)

    def import_map_file(self, extension):
        # TODO: Move this to the MatchPoint class
        #coefficients = np.genfromtxt(self.absoluteFilePath.with_suffix('.map'))
        file_content=np.genfromtxt(self.absoluteFilePath.with_suffix('.map'))
        if len(file_content) == 64:
            [coefficients, coefficients_inverse] = np.split(file_content, 2)
        elif len(file_content) == 32:
            coefficients = file_content
        else:
            raise TypeError('Error in import map file, incorrect number of lines')

        degree = int(np.sqrt(len(coefficients) // 2) - 1)
        P = coefficients[:len(coefficients) // 2].reshape((degree + 1, degree + 1))
        Q = coefficients[len(coefficients) // 2 : len(coefficients)].reshape((degree + 1, degree + 1))

        self.mapping = mp.MatchPoint(transformation_type='nonlinear')
        self.mapping.transformation = mp.polywarp.PolywarpTransform(params=(P,Q)) #{'P': P, 'Q': Q}
        #self.mapping.file = self

        if len(file_content)==64:
            degree = int(np.sqrt(len(coefficients_inverse) // 2) - 1)
            Pi = coefficients_inverse[:len(coefficients_inverse) // 2].reshape((degree + 1, degree + 1))
            Qi = coefficients_inverse[len(coefficients_inverse) // 2 : len(coefficients_inverse)].reshape((degree + 1, degree + 1))
        else:
            grid_range = 500 # in principle the actual image size doesn't matter
            # image_height = self._average_image.shape[0]

            # Can't we make this independent of the image?
            grid_coordinates = np.array([(a,b) for a in np.arange(0, grid_range//2, 5) for b in np.arange(0, grid_range, 5)])
            transformed_grid_coordinates = mp.polywarp.polywarp_apply(P, Q, grid_coordinates)
            # plt.scatter(grid_coordinates[:, 0], grid_coordinates[:, 1], marker='.')
            # plt.scatter(transformed_grid_coordinates[:,0], transformed_grid_coordinates[:,1], marker='.')
            Pi, Qi = mp.polywarp.polywarp(grid_coordinates, transformed_grid_coordinates)
            # transformed_grid_coordinates2 = polywarp_apply(Pi, Qi, transformed_grid_coordinates)
            # plt.scatter(transformed_grid_coordinates2[:, 0], transformed_grid_coordinates2[:, 1], marker='.')
            # plt.scatter(grid_coordinates[:, 0], grid_coordinates[:, 1], marker='.', facecolors='none', edgecolors='r')
       # self.mapping = mp.MatchPoint(transformation_type='nonlinear')
        self.mapping.transformation_inverse = mp.polywarp.PolywarpTransform(params=(Pi,Qi)) # {'P': Pi, 'Q': Qi}
        self.mapping.file = self
        self.mapping.source_name = 'Donor'
        self.mapping.destination_name = 'Acceptor'

    def export_map_file(self):
        warnings.warn('The export_map_file method will be depricated, use export_mapping instead')
        self.export_mapping(filetype='classic')

    def import_mapping_file(self, extension):
        self.mapping = mp.MatchPoint.load(self.absoluteFilePath.with_suffix(extension))

    def use_for_darkfield_correction(self):
        image = self.get_projection_image(projection_type='average', frame_range=(0, None), apply_corrections=False)
        tifffile.imwrite(self.experiment.main_path / 'darkfield.tif', image, imagej=True)
        self.experiment.load_darkfield_correction()

    def find_coordinates(self, **configuration):
        '''
        This function finds and sets the locations of all molecules within the movie's images.

        Main part of the work is done within 'find_molecules_utils.py'(imported as findMols)

        findMols.find_unique_molecules(), will loop through frames of the movie:
        1. creates an average image or maximum projection (for every N frames)
        2. finds the peaks in the new image (wraps around built-in method)
        3. keeps only the unique positions/pixels of peak locations ()
             3B allows to select only 'FRET pairs': intensity peak is seen in both donor/acceptor
        4. Finally, correct for an uncertainty in exact location due to photon shot noise etc. to remove 'functional duplicates'
        5. generate the coordinates array that is compatible with the Molecule() properties:
            [ [array1: donor coordinates],[array2: acceptor coordinates]  ]
            First array has all the coordinates (either found directly or inferred) of the donors:
            donor_coordinates = [[x1,y1],[x2,y2], etc.]
            Similar for acceptor coordinates



        configurations to be set by user:
        (within the find_coordinates section of the configuration file)
        --------------------------------
        channel : choose 'd'/'donor' or 'a'/'acceptor' for using only one of the two channels. alternatively
                  choose 'total_intensity' to use the sum of donor and acceptor signals
                  choose 'FRET pair' to only keep peak intensities found in both channels
                  choose 'both channels' to keep all peaks found in the entire image

        method:  choose 'average_image' or 'maximum_projection_image' to
                         set type of image used to find peak intensities.

        uncertainty_pixels: set number of pixels within which two peak intensities
                            should be considered to correspond to the same molecule

        img_per_N_pixels: use the image type of 'method' for every N frames of the movie (sliding window).
                         If no sliding window is used, the first N frames are used (a 'single window')

        use_sliding_window: type 'True' or 'False' to activate sliding window


        Additional configurations to be set
        (within 'peak_finding' section of configuration file)
        ------------------------------------
        ADD DESCRIPTIONS HERE!!!
        '''

        # TODO: Add configuration to nc file
        # TODO: Split method into multiple functions

        # --- Refresh configuration ----
        # if not configuration:
        configuration_from_config_file = self.experiment.configuration['find_coordinates']

        configuration_from_config_file.update(configuration)
        configuration = configuration_from_config_file

        # --- Get settings from configuration file ----
        channels = configuration['channels']
        method = configuration['method']
        peak_finding_configuration = configuration['peak_finding']
        projection_type = configuration['projection_image']['projection_type']
        frame_range = configuration['projection_image']['frame_range']
        illumination = configuration['projection_image']['illumination']

        sliding_window = configuration['sliding_window']
        use_sliding_window = configuration['sliding_window']['use_sliding_window']
        minimal_point_separation = sliding_window['minimal_point_separation']

        # --- set illumination configuration
        #  An integer number for choosing one of the laser lines (the order of it first appeared)
        #  ex. Two laser lines (532 and 640) in Alex mode starting with 532 excitation: 0 for green and 1 for red
        #  None for simple average of the frames regardless of the order of illumination profile.

        # --- make the windows
        # (if no sliding windows, just a single window is made to make it compatible with next bit of code) ----
        frame_ranges = [frame_range]
        if use_sliding_window:
            start_frames = (frame_ranges[0][0], self.movie.number_of_frames, sliding_window['frame_increment'])
            window_size = frame_ranges[0][1] - frame_ranges[0][0]
            frame_ranges = frame_ranges + [window_size, window_size, 0] * np.arange(start_frames)[:, None]

        # coordinates = set()
        if method == 'by_channel':
            coordinate_sets = [set() for channel in channels]
        elif method in ('average_channels', 'sum_channels'):
            # SHK: following two lines are not needed for average/sum channels
            # if len(channels) < 2:
            #     raise ValueError('No channels to overlay')
            # END SHK
            coordinate_sets = [set()]

        # coordinates_sets = dict([(channel, set()) for channel in channels])
        # coordinate_sets = [set() for channel in channels]

        # --- Loop over all frames and find unique set of molecules ----
        for frame_range in frame_ranges:

            # --- allowed to apply sliding window to either the max projection OR the averages ----
            image = self.get_projection_image(projection_type=projection_type, frame_range=frame_range,
                                              illumination=illumination)

            # image = self.average_image()
            self.movie.read_header()

            # Do we need a separate image?
            # # --- we output the "sum of these images" ----
            # find_coords_img += image

            if method == 'by_channel':
                # coordinates_per_channel = dict([(channel, set()) for channel in channels])
                channel_images = [self.movie.get_channel(image=image, channel=channel) for channel in channels]

            elif method in ('average_channels', 'sum_channels'):
                # Possibly we can move making the overlayed image to the Movie class.
                # TODO: make this usable for any number of channels
                donor_image = self.movie.get_channel(image=image, channel='d')
                # acceptor_image = self.movie.get_channel(image=image, channel='a')
                image_transformed = self.mapping.transform_image(image, direction='Acceptor2Donor')
                acceptor_image_transformed = self.movie.get_channel(image=image_transformed, channel='d')

                if method == 'average_channels':
                    channel_images = [(donor_image + acceptor_image_transformed) / 2]
                elif method == 'sum_channels':
                    channel_images = [(donor_image + acceptor_image_transformed)]
                channels = ['d'] # When number of channels can be > 2 this should probably be the channel with the lowest number

                # TODO: Make this a separate plotting function, possibly in Movie
                # plt.imshow(np.stack([donor_image.astype('uint8'),
                #                      acceptor_image_transformed.astype('uint8'),
                #                      np.zeros((self.movie.height,
                #                                self.movie.width // 2)).astype('uint8')],
                #                     axis=-1))
            else:
                raise ValueError(f'"{method}" is not a valid method.')

            print(f' Finding molecules in {self}')
            for i, channel_image in enumerate(channel_images):
                channel_coordinates = find_peaks(image=channel_image, **peak_finding_configuration)  # .astype(int)))

                # ---- optimize / fine-tune the coordinate positions ----
                coordinate_optimization_functions = \
                    {'coordinates_within_margin': coordinates_within_margin,
                     'coordinates_after_gaussian_fit': coordinates_after_gaussian_fit,
                     'coordinates_without_intensity_at_radius': coordinates_without_intensity_at_radius}
                for f, kwargs in configuration['coordinate_optimization'].items():
                    if len(channel_coordinates) == 0:
                        break
                    channel_coordinates = coordinate_optimization_functions[f](channel_coordinates, channel_image, **kwargs)


                channel_coordinates = set_of_tuples_from_array(channel_coordinates)

                coordinate_sets[i].update(channel_coordinates)

        # Check whether points are found
        for coordinate_set in coordinate_sets:
            if len(coordinate_set) == 0:
                # Reset current .nc file
                self._init_dataset(0) # SHK: Creating a dummy dataset tp avoid errors in the downstream analysis
                # This actually creates an empty dataset.
                coordinates = xr.DataArray(np.empty((0, 2, 2)), dims=('molecule', 'channel', 'dimension'),
                                    coords={'channel': [0, 1], 'dimension': [b'x', b'y']}, name='coordinates')
                coordinates.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')
                print('no peaks found')
                return

        # --- correct for photon shot noise / stage drift ---
        # Not sure whether to put this in front of combine_coordinate_sets/detect_FRET_pairs or behind [IS: 12-08-2020]
        # I think before, as you would do it either for each window, or for the combined windows.
        # Transforming the coordinate sets for each window will be time consuming and changes the distance_threshold.
        # And you would like to combine the channel sets on the merged coordinates.
        for i in range(len(coordinate_sets)):
            # --- turn into array ---
            coordinate_sets[i] = array_from_set_of_tuples(coordinate_sets[i])

            if use_sliding_window: # Do we actually need to put this if statement here [IS: 31-08-2020]
                                   # If not, in default configuration take minimal_point_separation outside sliding_window
                coordinate_sets[i] = merge_nearby_coordinates(coordinate_sets[i], distance_threshold=minimal_point_separation)

            # Map coordinates to main channel in movie
            # TODO: make this usable for any number of channels

            coordinate_sets[i] = coordinate_sets[i]+self.movie.get_channel_from_name(channels[i]).boundaries[0]
            if channels[i] in ['a', 'acceptor']:
            # if i > 0: #i.e. if channel is not main channel # this didn't work when selecting only the acceptor channel
                # Maybe we can do this earlier, right after point detection, then we need only a single coordinate_set
                coordinate_sets[i] = self.mapping.transform_coordinates(coordinate_sets[i],
                                                                        direction='Acceptor2Donor')

        # TODO: make this usable for any number of channels
        if len(coordinate_sets) == 1:
            coordinates = coordinate_sets[0]
        elif len(coordinate_sets) > 1:
            raise NotImplementedError('Assessing found coordinates in multiple channels does not work properly yet')
            # TODO: Make this function.
            #  This can easily be done by creating a cKDtree for each coordinate set and
            #  by finding the points close to each other
            coordinates = combine_coordinate_sets(coordinate_sets, method='and')  # the old detect_FRET_pairs

        # TODO: make this usable for more than two channels
        # TODO: Use set_coordinates_of_channel
        coordinates_in_main_channel = coordinates
        coordinates_list = [coordinates]
        for i in range(self.movie.number_of_channels)[1:]: # This for loop will only be useful once we make this usable for more than two channels
            if self.movie.number_of_channels > 2:
                raise NotImplementedError()
            coordinates_in_other_channel = self.mapping.transform_coordinates(coordinates_in_main_channel, direction='Donor2Acceptor')
            coordinates_list.append(coordinates_in_other_channel)

        coordinates = np.hstack(coordinates_list)

        coordinates_selections = [coordinates_within_margin_selection(coordinates, bounds=self.movie.channels[i].boundaries,
                                                                      **configuration['coordinate_optimization']['coordinates_within_margin'])
                                  for i, coordinates in enumerate(coordinates_list)]
        selection = np.vstack(coordinates_selections).all(axis=0)
        coordinates = coordinates[selection]

        coordinates = coordinates.reshape((-1, 2))

        # should also have incorporated check coordinatesDA_within_margin from MD_check_boundaries
        # --- finally, we set the coordinates of the molecules ---
        # self.coordinates = coordinates

        peaks = xr.DataArray(coordinates, dims=("peak", 'dimension'),
                     coords={'peak': range(len(coordinates)), 'dimension': [b'x', b'y']}, name='coordinates')

        coordinates = split_dimension(peaks, 'peak', ('molecule', 'channel'), (-1, self.movie.number_of_channels)).reset_index('molecule', drop=True)
        # file = str(self.relativeFilePath)
        # #coordinates = split_dimension(coordinates, 'molecule', ('molecule_in_file', 'file'), (-1, 1), (-1, [file]), to='multiindex')
        # coordinates = coordinates.reset_index('molecule').rename(molecule_='molecule_in_file')
        # self.experiment.dataset.drop_sel(file=str(self.relativeFilePath), errors='ignore')

        sys.stdout.write('\r')
        print(f'   {coordinates.molecule.size} molecules found')

        # Because split_dimension doesn't keep the channels in case of an empty array.
        if len(coordinates) == 0:
            coordinates = xr.DataArray(np.empty((0, 2, 2)), dims=('molecule', 'channel', 'dimension'),
                                       coords={'channel': [0, 1], 'dimension': [b'x', b'y']}, name='coordinates')

        # if len(coordinates) !=0:

        add_configuration_to_dataarray(coordinates, File.find_coordinates, locals(), units='pixel')

        for item_name in ['pixel_size', 'pixel_size_unit', 'stage_coordinates']:
            if hasattr(self.movie, item_name):
                if item_name == 'stage_coordinates':
                    item = getattr(self.movie, item_name)[0]
                else:
                    item = getattr(self.movie, item_name)
                coordinates.attrs[item_name] = item

        self.coordinates = coordinates

        # self.molecules.export_pks_file(self.relativeFilePath.with_suffix('.pks'))

    def determine_psf_size(self, method='gaussian_fit', projection_type='average', frame_range=(0,20), channel_index=0, illumination_index=0,
                           peak_finding_kwargs={'minimum_intensity_difference': 150}, maximum_radius=5):
        image = self.get_projection_image(projection_type=projection_type, frame_range=frame_range,
                                          illumination=illumination_index)
        image = self.movie.get_channel(image, channel=channel_index)

        coordinates = find_peaks(image=image, **peak_finding_kwargs)  # .astype(int)))
        coordinates_fit, parameters = coordinates_after_gaussian_fit(coordinates, image, gaussian_width=15, return_fit_parameters=True)
        # offset, amplitude, x0, y0, sigma_x, sigma_y
        sigmas = parameters[:, 4]
        selection = (0 < sigmas) & (sigmas < maximum_radius)
        sigmas = sigmas[selection]

        fig, ax = plt.subplots(layout='constrained')
        ax.imshow(image)
        # ax.scatter(*coordinates_fit.T, s=0.5, c=parameters[:,0])
        for c, s in zip(coordinates_fit[selection], sigmas):
            circle = plt.Circle(c, 2*s, ec='r', fc='None')
            ax.add_patch(circle)
        ax.set_xlabel('x (pixel)')
        ax.set_ylabel('y (pixel)')
        ax.set_title('Circles at $2\sigma$')

        psf_size_path = self.experiment.analysis_path.joinpath('PSF_size')
        psf_size_path.mkdir(parents=True, exist_ok=True)
        filename = Movie.image_info_to_filename('fits_in_image', projection_type=projection_type, frame_range=frame_range,
                                                illumination=illumination_index) + f'_c{channel_index}.png'
        fig.savefig(psf_size_path / filename, bbox_inches='tight')

        bins = 100
        fig, ax = plt.subplots(layout='constrained')
        counts, bin_edges, _ = ax.hist(sigmas, bins=bins, range=(0, maximum_radius))
        ax.set_xlabel('σ (pixel)')
        ax.set_ylabel('Count')
        bin_centers = (bin_edges[1:] + bin_edges[:-1])/2

        if method == 'median':
            psf_size = np.median(sigmas)

        elif method == 'gaussian_fit':
            from scipy.optimize import curve_fit
            def oneD_gaussian(x, offset, amplitude, x0, sigma):
                return offset + amplitude * np.exp(- (x - x0)**2 / (2 * sigma**2))

            p0 = [np.min(counts), np.max(counts) - np.min(counts), np.median(sigmas), np.std(sigmas)]
            popt, pcov = curve_fit(oneD_gaussian, bin_centers, counts, p0)
            x = np.linspace(0, maximum_radius, 1000)
            ax.plot(x, oneD_gaussian(x, *popt), c='r')
            psf_size = popt[2]

        y_range = ax.get_ylim()
        ax.vlines(psf_size, *y_range, color='r')
        ax.set_ylim(y_range)
        ax.set_title(f'psf_size = {psf_size}')
        filename = Movie.image_info_to_filename('sigma_plot', projection_type=projection_type, frame_range=frame_range,
                                                illumination=illumination_index) + f'_c{channel_index}.png'
        fig.savefig(psf_size_path / filename, bbox_inches='tight')

        return psf_size

    @property
    @return_none_when_executed_by_pycharm
    def coordinates(self):
        if self.absoluteFilePath.with_suffix('.nc').exists():
            with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
                if hasattr(dataset, 'coordinates'):
                    return dataset['coordinates'].load()
                else:
                    return None
        else:
            return None

    @coordinates.setter
    def coordinates(self, coordinates):
        # Reset current .nc file
        self._init_dataset(len(coordinates.molecule))

        coordinates.drop('file', errors='ignore').to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')
        # self.extract_background()

        # self.molecules.export_pks_file(self.absoluteFilePath.with_suffix('.pks'))

    def extract_background(self, configuration=None):
        #TODO: probably remove this
        sys.stdout.write(f' Calculating background in {self}')
        # --- Refresh configuration ----
        if not configuration:
            configuration = self.experiment.configuration['background']

        # --- get settings from configuration file ----
        if 'frames_for_background' in configuration.keys():
            frames_for_background = configuration['frames_for_background']
            if frames_for_background['last_frame'] == 'last':
                frames_for_background['last_frame'] = self.movie.number_of_frames-1
            frames_for_background = [int(frames_for_background['first_frame']),
                                     int(frames_for_background['last_frame']) - frames_for_background['first_frame'] + 1]
        else:
            print('configurations for the background subtraction is not found. All the frames will be used.')
            frames_for_background = [0, self.movie.number_of_frames-1]

        # --- get the averaged images for background extraction per illumination profile
        # todo: In the following code, let's use 'illumination' instead of 'illuminations', similar to channel.
        if self.movie.illumination_arrangement is not None:
            image_for_background = [None] * len(self.movie.illumination_arrangement)
            illuminations_to_use = self.movie.illumination_arrangement
        else:
            image_for_background = [None]
            illuminations_to_use = [None]
        for illumination_id, illumination in enumerate(illuminations_to_use):
            image_for_background[illumination_id] = \
                        self.movie.make_projection_image(projection_type='average',
                                                         start_frame=frames_for_background[0],
                                                         number_of_frames=frames_for_background[1],
                                                         illumination=illumination)

        # START of original code
        # background_list = []
        # for i, channel in enumerate(self.movie.channels):
        #     channel_image = self.movie.get_channel(image_for_background[illumination_id], i)
        #     channel_coordinates = self.coordinates_from_channel(i).values-self.movie.channels[i].vertices[0]
        #     background_list.append(extract_background(channel_image, channel_coordinates, method=configuration['method']))
        #
        # background = xr.DataArray(np.vstack(background_list).T, dims=['molecule','channel'], name='background')


        # END of original, START modified
        background_list = []
        for illumination_id, illumination in enumerate(illuminations_to_use):
            tmp_background_list = []
            for i, channel in enumerate(self.movie.channels):
                channel_image = self.movie.get_channel(image_for_background[illumination_id], i)
                channel_coordinates = self.coordinates_from_channel(i).values-self.movie.channels[i].vertices[0]
                tmp_background_list.append(extract_background(channel_image, channel_coordinates, method=configuration['method']))
            background_list.append(np.vstack(tmp_background_list).T)
        background = xr.DataArray(background_list, dims=['illumination', 'molecule', 'channel'], name='background')
        # END modified

        background.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')
        sys.stdout.write(f'\r   background calculated {self}\n')

    def import_excel_file(self, filename=None):
        if filename is None:
            filename = f'{self.absoluteFilePath}_steps_data.xlsx'
        try:
            steps_data = pd.read_excel(filename, index_col=[0,1],
                                       dtype={'kon':np.str})       # reads from the 1st excel sheet of the file

            print(f'imported steps data from excel file for {self.name}')
        except FileNotFoundError:
            print(f'No saved analysis for {self.name} as {filename}')
            return
        molecules = steps_data.index.unique(0)
        indices = [int(m.split()[-1]) for m in molecules]
        for mol in self.molecules:
            if mol.index + 1 not in indices:
                continue
            mol.steps = steps_data.loc[f'mol {mol.index + 1}']
            # if saved steps are found for molecule it is assumed selected
            # mol.isSelected = True
            if 'kon' in mol.steps.columns:
                k = [int(i) for i in mol.steps.kon[0]]
                mol.kon_boolean = np.array(k).astype(bool).reshape((4,3))
        return steps_data

    def extract_traces(self, mask_size=None, neighbourhood_size=None, background_correction=None, alpha_correction=None,
                       gamma_correction=None):
        # TODO: Add configuration to nc file
        if self.number_of_molecules == 0:
            print('   no traces available!!')
            return

        if self.movie is None: raise FileNotFoundError('No movie file was found')

        print(f'  Extracting traces in {self}')

        configuration = self.configuration['trace_extraction']
        # channel = configuration['channel']  # Default was 'all'
        if mask_size is None:
            mask_size = configuration['mask_size']  # Default was 11
        if neighbourhood_size is None:
            neighbourhood_size = configuration['neighbourhood_size']  # Default was 11
        # subtract_background = configuration['subtract_background']
        # correct_illumination = configuration['correct_illumination']

        if mask_size == 'TIR-T' or mask_size == 'TIR-V':
            mask_size = 1.291
        elif mask_size == 'TIR-S 1.5x 2x2':
            mask_size = 0.8
        elif mask_size == 'TIR-S 1x 2x2':
            mask_size = 0.55
        elif mask_size == 'BN-TIRF':
            mask_size = 1.01

        # if subtract_background:
        #     background = self.background
        # else:
        #     background = None

        # if correct_illumination:
        #     if not hasattr(self.dataset, 'illumination_correction'):
        #         self.determine_illumination_correction(frames=frames)
        #     # frames = frames.astype(float)
        #     # frames.loc[{'channel': channel.index}] *= self.illumination_correction[frame_number, channel.index]
        #     frames = frames * self.illumination_correction

        # channel_offsets = xr.DataArray(np.vstack([channel.origin for channel in self.movie.channels]),
        #                                dims=('channel', 'dimension'),
        #                                coords={'channel': [channel.index for channel in self.movie.channels],
        #                                        'dimension': ['x', 'y']}) # TODO: Move to Movie
        # coordinates = self.coordinates - channel_offsets

        intensity = extract_traces(self.movie, self.coordinates, background=None, mask_size=mask_size,
                                   neighbourhood_size=neighbourhood_size, correct_illumination=False)

        add_configuration_to_dataarray(intensity, units='a.u.') # TODO: Link to units in movie metadata?
        intensity.attrs['configuration'] = json.dumps(dict(mask_size=mask_size, neighbourhood_size=neighbourhood_size))
        intensity.attrs['movie_configuration'] = json.dumps(self.movie.configuration)

        if self.movie.time is not None: # hasattr(self.movie, 'time')
            intensity = intensity.assign_coords(time=self.movie.time)

        # if self.movie.illumination is not None:
        intensity = intensity.assign_coords(illumination=self.movie.illumination_index_per_frame)

        intensity.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if 'intensity_raw' in self.data_vars:
            intensity_raw = self.intensity
            intensity_raw.name = 'intensity_raw'
            intensity_raw.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if background_correction is not None or alpha_correction is not None or gamma_correction is not None:
            self.apply_trace_corrections(background_correction, alpha_correction, gamma_correction)

        if self.movie.number_of_channels > 1:
            self.calculate_FRET()

    def apply_trace_corrections(self, background_correction=None, alpha_correction=None,
                       gamma_correction=None):
        from papylio.trace_correction import trace_correction

        if 'intensity_raw' in self.data_vars:
            intensity_raw = self.intensity_raw
        else:
            intensity_raw = self.intensity
            intensity_raw.name = 'intensity_raw'
            intensity_raw.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

        intensity = trace_correction(intensity_raw, background_correction, alpha_correction, gamma_correction)
        intensity.name = 'intensity'
        initial_configuration = intensity.attrs['configuration']
        add_configuration_to_dataarray(intensity, File.apply_trace_corrections, locals(), units='a.u.') # TODO: Link to units in movie metadata?
        intensity.attrs['configuration'] = initial_configuration[:-1] + ', ' + intensity.attrs['configuration'][1:]

        intensity.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if 'FRET' in self.data_vars:
            self.calculate_FRET()

    def calculate_FRET(self):
        intensity = self.intensity
        FRET = calculate_FRET(intensity)
        FRET.attrs = intensity.attrs
        FRET.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def get_traces(self, selected=False):
        dataset = self.dataset

        included_data_var_names = []
        for name in list(dataset.data_vars.keys()):
            if 'frame' in dataset[name].dims:
                included_data_var_names.append(name)

        traces = dataset[included_data_var_names]

        if selected:
            traces = traces.sel(molecule=dataset.selected)

        return traces

    def plot_hmm_rates(self, name=None):
        if name is None:
            name = self.name
        dataset = self.dataset
        title = name + '_hmm_rates'
        figure, axis = plt.subplots(figsize=(4, 2.5), layout='constrained')
        # axis = axes[i]
        ds_2_state = dataset.sel(molecule=(dataset.number_of_states == 2) & dataset.selected)
        save_path = self.experiment.analysis_path / 'hmm rate histograms'
        save_path.mkdir(exist_ok=True)
        file_path = save_path / (title + '.png')
        if len(ds_2_state.molecule) == 0:
            if file_path.exists():
                file_path.unlink()
            return None
        unit_string = '$s^{-1}$'
        rate_1_to_0 = ds_2_state.transition_rate.sel(from_state=1, to_state=0)
        label_1_to_0 = '$\\overline{k_{1\\rightarrow0}}$ = ' +\
                       f'{rate_1_to_0.mean().item():.1f}±{rate_1_to_0.std().item():.1f} {unit_string}'
        rate_1_to_0.plot.hist(bins=50, label=label_1_to_0, ax=axis, range=(0, 15), alpha=0.5)
        rate_0_to_1 = ds_2_state.transition_rate.sel(from_state=0, to_state=1)
        label_0_to_1 = '$\\overline{k_{0\\rightarrow1}}$ = ' +\
                       f'{rate_0_to_1.mean().item():.1f}±{rate_0_to_1.std().item():.1f} {unit_string}'
        rate_0_to_1.plot.hist(bins=50, label=label_0_to_1, ax=axis, range=(0, 15), alpha=0.5)
        axis.legend()
        axis.set_xlabel('Transition rate ($s^{-1}$)')
        axis.set_ylabel('Count')

        axis.set_title(title)
        #
        # unit_string = '$s^{-1}$'
        # rate_string = f'Mean rates:\n' \
        #               '$\\overline{k_{0\\rightarrow1}}$ = ' + f'{rate_0_to_1.mean().item():.1f}±{rate_0_to_1.std().item():.1f} {unit_string}\n' \
        #               '$\\overline{k_{1\\rightarrow0}}$ = ' + f'{rate_1_to_0.mean().item():.1f}±{rate_1_to_0.std().item():.1f} {unit_string}'
        # axis.text(0.95,0.05, rate_string, ha='right', transform=axis.transAxes)
        figure.savefig(file_path)

    def save_dataset_selected(self):
        encoding = {'file': {'dtype': '|S'}, 'selected': {'dtype': bool}}
        self.dataset_selected.to_netcdf(self.absoluteFilePath.parent / (self.name + '_selected.nc'), engine='netcdf4', mode='w', encoding=encoding)

    def import_pks_file(self, extension):
        peaks = import_pks_file(self.absoluteFilePath.with_suffix('.pks'))
        peaks = split_dimension(peaks, 'peak', ('molecule', 'channel'), (-1, 2)).reset_index('molecule', drop=True)
        # peaks = split_dimension(peaks, 'molecule', ('molecule_in_file', 'file'), (-1, 1), (-1, [file]), to='multiindex')

        if not self.absoluteFilePath.with_suffix('.nc').is_file():
            self._init_dataset(len(peaks.molecule))

        coordinates = peaks.sel(parameter=['x', 'y']).rename(parameter='dimension')
        background = peaks.sel(parameter='background', drop=True)

        xr.Dataset({'coordinates': coordinates, 'background': background})\
            .to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def export_pks_file(self):
        peaks = xr.merge([self.coordinates.to_dataset('dimension'), self.background.to_dataset()])\
            .stack(peaks=('molecule', 'channel')).to_array(dim='parameter').T
        export_pks_file(peaks, self.absoluteFilePath.with_suffix('.pks'))
        self.extensions.add('.pks')

    def import_traces_file(self, extension):
        traces = import_traces_file(self.absoluteFilePath.with_suffix('.traces'))
        intensity = split_dimension(traces, 'trace', ('molecule', 'channel'), (-1, 2))\
            .reset_index(['molecule','frame'], drop=True)

        if not self.absoluteFilePath.with_suffix('.nc').is_file():
            self._init_dataset(len(intensity.molecule))

        xr.Dataset({'intensity': intensity}).to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def export_traces_file(self):
        traces = self.intensity.stack(trace=('molecule', 'channel')).T
        export_traces_file(traces, self.absoluteFilePath.with_suffix('.traces'))
        self.extensions.add('.traces')


    # def addMolecule(self):
    #     index = len(self.molecules) # this is the molecule number
    #     self.molecules.append(Molecule(self))
    #     self.molecules[-1].index = index

    def savetoExcel(self, filename=None, save=True):
        if filename is None:
            filename = f'{self.absoluteFilePath}_steps_data.xlsx'

        # Find the molecules for which steps were selected
        molecules_with_data = [mol for mol in self.molecules if mol.steps is not None]


        # Concatenate all steps dataframes that are not None
        mol_data = [mol.steps for mol in molecules_with_data]
        if not mol_data:
            print(f'no data to save for {self.name}')
            return
        keys = [f'mol {mol.index + 1}' for mol in molecules_with_data]

        steps_data = pd.concat(mol_data, keys=keys, sort=False)
        # drop duplicate columns
        steps_data = steps_data.loc[:,~steps_data.columns.duplicated()]
        if save:
            print("data saved in: " + filename)
            writer = pd.ExcelWriter(filename)
            steps_data.to_excel(writer, self.name)
            writer.save()
        return steps_data

    def autoThreshold(self, trace_name, threshold=100, max_steps=20,
                      only_selected=False, kon_str='000000000'):
        nam = trace_name
        for mol in self.molecules:

            trace = mol.I(0)*int((nam == 'green')) + \
                    mol.I(1)*int((nam == 'red')) +\
                     mol.E()*int((nam == 'E'))  # Here no offset corrections are applied yet

            d = mol.find_steps(trace)
            frames = d['frames']
            times = frames*self.exposure_time
            times = np.sort(times)
            mol.steps = pd.DataFrame({'time': times, 'trace': nam,
                                  'state': 1, 'method': 'thres',
                                'thres': threshold, 'kon': kon_str})
        filename = self.name+'_steps_data.xlsx'
        data = self.savetoExcel(filename)
        return data

    def perform_mapping(self, **configuration):
        image = self.average_image()
        if not configuration:
            configuration = self.experiment.configuration['mapping']

        transformation_type = configuration['transformation_type']
        print(transformation_type)
        method = configuration['method']

        donor_image = self.movie.get_channel(image=image, channel='d')
        acceptor_image = self.movie.get_channel(image=image, channel='a')
        donor_coordinates = find_peaks(image=donor_image,
                                       **configuration['peak_finding']['donor'])
        if donor_coordinates.size == 0: #should throw a error message to warm no acceptor molecules found
            print('No donor molecules found')
        acceptor_coordinates = find_peaks(image=acceptor_image,
                                          **configuration['peak_finding']['acceptor'])
        if acceptor_coordinates.size == 0: #should throw a error message to warm no acceptor molecules found
            print('No acceptor molecules found')
        acceptor_coordinates = mp.coordinate_transformations.transform(acceptor_coordinates, translation=[image.shape[1]//2, 0])
        # print(acceptor_coordinates.shape, donor_coordinates.shape)
        print(f'Donor: {donor_coordinates.shape[0]}, Acceptor: {acceptor_coordinates.shape[0]}')
        coordinates = np.append(donor_coordinates, acceptor_coordinates, axis=0)

        # coordinate_optimization_functions = \
        #     {'coordinates_within_margin': coordinates_within_margin,
        #      'coordinates_after_gaussian_fit': coordinates_after_gaussian_fit,
        #      'coordinates_without_intensity_at_radius': coordinates_without_intensity_at_radius}
        #
        # for f, kwargs in configuration['coordinate_optimization'].items():
        #     coordinates = coordinate_optimization_functions[f](coordinates, image, **kwargs)

        if 'coordinates_after_gaussian_fit' in configuration['coordinate_optimization']:
            gaussian_width = configuration['coordinate_optimization']['coordinates_after_gaussian_fit']['gaussian_width']
            coordinates = coordinates_after_gaussian_fit(coordinates, image, gaussian_width)

        if 'coordinates_without_intensity_at_radius' in configuration['coordinate_optimization']:
            coordinates = coordinates_without_intensity_at_radius(coordinates, image,
                                                                  **configuration['coordinate_optimization']['coordinates_without_intensity_at_radius'])
                                                                  # radius=4,
                                                                  # cutoff=np.median(image),
                                                                  # fraction_of_peak_max=0.35) # was 0.25 in IDL code

        if 'coordinates_within_margin' in configuration['coordinate_optimization']:
            margin = configuration['coordinate_optimization']['coordinates_within_margin']['margin']
        else:
            margin = 0

        donor_coordinates = coordinates_within_margin(coordinates,
                                                      bounds=self.movie.get_channel_from_name('d').boundaries, margin=margin)
        acceptor_coordinates = coordinates_within_margin(coordinates,
                                                         bounds=self.movie.get_channel_from_name('a').boundaries, margin=margin)

        # TODO: put overlapping coordinates in file.coordinates for mapping file
        # Possibly do this with mapping.nearest_neighbour match
        # self.coordinates = np.hstack([donor_coordinates, acceptor_coordinates]).reshape((-1, 2))

        if ('initial_translation' in configuration) and (configuration['initial_translation'] == 'width/2'):
            # initial_transformation = {'translation': [image.shape[0] // 2, 0]}
            initial_transformation = {'translation': [image.shape[1] // 2, 0]}
        else:
            if configuration['initial_translation'][0] == '[':  # remove brackets
                arr = [float(x) for x in configuration['initial_translation'][1:-1].split(' ')]
                initial_transformation = {'translation': arr}
            else:
                arr = [float(x) for x in configuration['initial_translation'].split(' ')]
                initial_transformation = {'translation': configuration['initial_translation']}

        # Obtain specific mapping parameters from configuration file
        additional_mapping_parameters = {key: configuration[key]
                                         for key in (configuration.keys() and {'distance_threshold'})}

        self.mapping = mp.MatchPoint(source_name='Donor',
                                     source=donor_coordinates,
                                     destination_name='Acceptor',
                                     destination=acceptor_coordinates,
                                     method=method,
                                     transformation_type=transformation_type,
                                     initial_transformation=initial_transformation)
        self.mapping.perform_mapping(**additional_mapping_parameters)
        self.mapping.file = self

        # self.export_mapping(filetype='classic')
        self.export_mapping()

        self.show_mapping_in_image()

        self.use_mapping_for_all_files()

    def show_mapping_in_image(self, axis=None, save=True):
        if not hasattr(self, 'mapping') or self.mapping is None:
            raise RuntimeError('File does not contain a mapping.')
        if axis is None:
            figure, axis = plt.subplots()
        else:
            figure = axis.figure
        self.show_average_image(figure=figure)
        self.mapping.show_mapping_transformation(axis=axis, show_source=True)

        if save:
            axis.axis('off')
            axis.set_title('')
            figure.set_size_inches(8, 8)
            figure.savefig(self.relativePath / (self.name + '_mapping.png'), bbox_inches="tight", pad_inches=0, dpi=300)

        return figure, axis

    def copy_coordinates_to_selected_files(self):
        for file in self.experiment.selectedFiles:
            if file is not self:
                file._init_dataset(len(self.molecule))
                self.coordinates.to_netcdf(file.absoluteFilePath.with_suffix('.nc'), engine='netcdf4')

    def use_mapping_for_all_files(self):
        print(f"\n{self} used as mapping")
        self.is_mapping_file = True
        #mapping = self.movie.use_for_mapping()
        for file in self.experiment.files:
            if file is not self:
                file.mapping = self.mapping
                file.is_mapping_file = False

    # def get_variable(self, variable, selected=False, frame_range=None, average=False):
    #     if variable in ['intensity','FRET','intensity_total']:
    #         da = getattr(self, 'get_'+variable)
    #     else:
    #         with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
    #             da = dataset[variable].load()
    #
    def get_variable(self, variable, selected=False, frame_range=None, average=False, return_none_if_nonexistent=False):
        # TODO: make it possible to also select the channel (or perform other selections), e.g. by passing 'intensity_c0'.
        """
        Get a variable.

        Parameters:
            variable (str): The name of the variable to retrieve.
            selected (bool, optional): Whether to return only selected molecules. Default is False.
            frame_range (tuple, optional): In case the returned variable has dimension 'frame', frame_range can be used
                to select the desired frames. Default is None.
            average (bool or str, optional): Whether to calculate the average of the variable over a specific dimension.
                If a string is provided, it represents the dimension to average over. Default is False.
            return_none_if_nonexistent (bool, optional): Whether to return None if the variable does not exist in the object.
                Default is False.

        Returns:
            xarray.DataArray: The requested variable.

        """
        if return_none_if_nonexistent and not hasattr(self, variable):
            return None

        da = getattr(self, variable)

        if selected is not False:
            if selected is True:
                selected = self.selected
            da = da.sel(molecule=selected)

        if frame_range is not None:
            da = da.sel(frame=slice(*frame_range))

        if average:
            da = da.mean(dim=average)
            if average == 'molecule':
                da = da.expand_dims({'name': [self.name]}, 0)

        return da

    def set_variable(self, data, **kwargs):
        da = xr.DataArray(data, **kwargs)
        da.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

    @property
    @return_none_when_executed_by_pycharm
    def intensity_total(self):
        return calculate_intensity_total(self.intensity)

    @property
    @return_none_when_executed_by_pycharm
    def selections(self):
        with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return xr.Dataset({value.name: value for key, value in dataset.data_vars.items()
                               if key.startswith('selection_')}).load() # .to_array(dim='selection')
        # return xr.concat([value for key, value in self.dataset.data_vars.items() if key.startswith('filter')], dim='filter')

    def create_selection(self, variable, channel, aggregator, operator, threshold, name=None):
        data_array = getattr(self, variable)

        if 'channel' in data_array.dims:
            channel_index = self.movie.get_channel_from_name(channel).index
            data_array = data_array.sel(channel=channel_index, drop=True)
            channel_str = 'c' + str(channel_index)
        else:
            channel_str = ''

        data_array = getattr(data_array, aggregator)('frame')

        if operator == '<':
            selection = data_array < threshold
        elif operator == '>':
            selection = data_array > threshold
        else:
            raise ValueError('Unknown operator')

        # selection.attrs = {'variable': variable, 'channel': channel, 'aggregator': aggregator,
        #                     'operator': operator, 'threshold': threshold}

        threshold_str = str(threshold).replace('.','p')

        add_configuration_to_dataarray(selection, File.create_selection, locals())

        if name is None:
            name = f'selection_{variable}_{channel_str}_{aggregator}_{operator}_{threshold_str}'
        if not name.startswith('selection_'):
            name = 'selection_' + name

        self.set_variable(selection, name=name)

    def copy_selections_to_selected_files(self):
        selection_configurations = self.selection_configurations()
        applied_selection = json.loads(self.selected.attrs['configuration'])

        for file in self.experiment.selectedFiles:
            if file is not self:
                for name, configuration in selection_configurations.items():
                    if configuration is None:
                        raise ValueError(f'Selection {name} is a custom selection that cannot be copied')
                    file.create_selection(**configuration)
                file.apply_selections(*applied_selection)

    @property
    @return_none_when_executed_by_pycharm
    def selection_names(self):
        return list(self.selections.data_vars.keys())

    @property
    @return_none_when_executed_by_pycharm
    def selection_names_active(self):
        return json.loads(self.selected.attrs['configuration'])

    def clear_selections(self):
        dataset = self.dataset
        dataset = dataset.drop_vars([name for name in dataset.data_vars.keys() if name.startswith('selection_')])
        # for name, da in dataset.data_vars.items():
        #     da.encoding['dtype'] = da.dtype
        encoding = {
            var: {"dtype": 'bool'} for var in dataset.data_vars if dataset[var].dtype == bool
        }
        dataset.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='w', encoding=encoding)

    def selection_configurations(self, *selection_names):
        selection_names = list(selection_names)
        if not selection_names:
            selection_names = self.selection_names

        selection_configurations = {}
        for name, selection in self.selections.items():
            if name in selection_names:
                if 'configuration' in selection.attrs:
                    selection_configurations[name] = json.loads(selection.attrs['configuration'])
                else:
                    selection_configurations[name] = None
        return selection_configurations

    def apply_selections(self, *selection_names, add_to_current=False):
        selection_names = list(selection_names)
        all_selection_names = self.selection_names
        if not selection_names:
            selection_names = all_selection_names

        if add_to_current:
            selection_names = json.loads(self.selected.attrs['configuration']) + selection_names
            selection_names = list(set(selection_names)) # Remove double names

        if not selection_names or selection_names[0] in [None, 'none', 'None']:
            selection_names = []
            selected = self.selected
            selected[:] = False
        else:
            for selection_name in selection_names:
                if selection_name not in all_selection_names:
                    raise ValueError(f'Selection {selection_name} does not exist')

            invert = np.zeros(len(selection_names), bool)
            for i, selection_name in enumerate(selection_names):
                if selection_name.startswith('~'):
                    invert[i] = True
                    selection_names[i] = selection_name[1:]

            #     selections = self.selections
            # else:
            selections = self.selections[selection_names].to_array(dim='selection')

            selections[invert] = ~selections[invert]
            selected = selections.all(dim='selection')

        add_configuration_to_dataarray(selected)
        selected.attrs['configuration'] = json.dumps(selection_names)
        selected.attrs['selection_configurations'] = json.dumps(self.selection_configurations(*selection_names))
        self.set_variable(selected, name='selected')

    @property
    @return_none_when_executed_by_pycharm
    def classification(self):
        # Or add a standard classification datavar in the dataset?
        if not 'classification' in self.data_vars:
            self.apply_classifications()
        classification = self.__getattr__('classification')
        # if 'classification_configurations' not in classification.attrs:
        #     classification.attrs['classification_configurations'] = None
        return classification

    @property
    @return_none_when_executed_by_pycharm
    def classifications(self):
        with xr.open_dataset(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return xr.Dataset({value.name: value for key, value in dataset.data_vars.items()
                               if key.startswith('classification_')}).load()  # .to_array(dim='selection')
        # return xr.concat([value for key, value in self.dataset.data_vars.items() if key.startswith('filter')], dim='filter')

    @property
    @return_none_when_executed_by_pycharm
    def classification_names(self):
        return list(self.classifications.data_vars.keys())

    def classification_configurations(self, classification_names='all'):
        if classification_names == 'all':
            classification_names = self.classification_names

        classification_configurations = {}
        for name, classification in self.classifications.items():
            if name in classification_names:
                if 'configuration' in classification.attrs:
                    classification_configurations[name] = json.loads(classification.attrs['configuration'])
                else:
                    classification_configurations[name] = None
        return classification_configurations

    def create_classification(self, classification_type, variable, select=None, name=None, classification_kwargs=None, apply=None):
        if classification_kwargs is None:
            classification_kwargs = {}
        if isinstance(variable, str):
            traces = getattr(self, variable)
        else:
            traces = variable
            variable = traces.name

        if select is not None:
            traces = traces.sel(**select)

        if classification_type == 'threshold':
            from papylio.analysis.classification_simple import classify_threshold
            ds = classify_threshold(traces, **classification_kwargs).to_dataset()
            # TODO: perhaps replace the following line with some function that actually spits out the classification kwargs.
            add_configuration_to_dataarray(ds.classification, classify_threshold, classification_kwargs)

        elif classification_type in ['hmm', 'hidden_markov_model']:
            # ds = hmm_traces(self.FRET, n_components=2, covariance_type="full", n_iter=100) # Old
            classification = self.classification
            selected = self.selected
            ds = classify_hmm(traces, classification, selected, **classification_kwargs)
            if 'configuration' in selected.attrs:
                ds.classification.attrs['applied_selections'] = selected.attrs['configuration']
            if 'configuration' in classification.attrs:
                ds.classification.attrs['applied_classifications'] = classification.attrs['configuration']
            #TODO: perhaps replace the following line with some function that actually spits out the classification kwargs.
            add_configuration_to_dataarray(ds.classification, classify_hmm, classification_kwargs)
        # TODO: create classification to deactivate certain frames of the trace
        else:
            raise ValueError('Unknown classification type')

        classification_kwargs = json.loads(ds.classification.attrs['configuration'])
        add_configuration_to_dataarray(ds.classification, File.create_classification, locals())

        if name is None:
            name = classification_type
        if not name.startswith('classification_'):
            name = 'classification_' + name
        ds = ds.rename({'classification': name})

        ds.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if apply is not None:
            self.apply_classifications(add_to_current=True, **{name: apply})

    def classify_hmm(self, variable, seed=0, n_states=2, threshold_state_mean=None,
                     level='molecule'):  # , use_selection=True, use_classification=True):
        self.create_classification(name='hmm', classification_type='hmm', variable=variable,
                                   classification_kwargs=dict(seed=seed, n_states=n_states,
                                threshold_state_mean=threshold_state_mean, level=level))

    def apply_classifications(self, add_to_current=False, **classification_assignment):
        if add_to_current:
            classification_assignment_old = json.loads(self.classification.attrs['configuration'])
            for key in classification_assignment.keys():
                classification_assignment_old.pop(key, None)
            classification_assignment = classification_assignment_old | classification_assignment

        all_classification_names = self.classification_names
        for classification_name in classification_assignment.keys():
            if classification_name not in all_classification_names:
                raise ValueError(f'Classification {classification_name} does not exist')

        classification_combined = np.zeros((len(self.molecule), len(self.frame)), 'int8')

        for classification_name, state_indices in classification_assignment.items():
            if not classification_name.startswith('classification'):
                raise ValueError('Only insert classifications')

            classification = getattr(self, classification_name)

            # TODO: The .values after classification can likely be removed after a certain update of xarray.
            if classification.dtype == 'bool':
                if type(state_indices) == list:
                    classification_combined[~classification.values] = state_indices[0]
                    classification_combined[classification.values] = state_indices[1]
                elif type(state_indices) == int:
                    if state_indices < 0:
                        classification_combined[~classification.values] = state_indices
                    else:
                        classification_combined[classification.values] = state_indices
                else:
                    raise TypeError('Wrong classification datatype')
            else: #if classification.dtype == int:
                for i, c in enumerate(np.unique(classification)):
                    if state_indices[i] is not None:
                        classification_combined[(classification == c).values] = state_indices[i]

        classification_combined = xr.DataArray(classification_combined)
        add_configuration_to_dataarray(classification_combined)
        classification_combined.attrs['configuration'] = json.dumps(classification_assignment)
        classification_combined.attrs['classification_configurations'] = json.dumps(self.classification_configurations(list(classification_assignment.keys())))

        self.set_variable(classification_combined, name='classification', dims=('molecule','frame'))

    @property
    @return_none_when_executed_by_pycharm
    def cycle_time(self):
        return self.time.diff('frame').mean().item()

    @property
    @return_none_when_executed_by_pycharm
    def sampling_interval(self):
        return self.time.diff('frame').mean().item()

    @property
    @return_none_when_executed_by_pycharm
    def frame_rate(self):
        return 1 / self.cycle_time

    def determine_dwells_from_classification(self, variable='FRET', selected=False, inactivate_start_and_end_states=True):
        # TODO: Make it possible to pass multiple traces.
        classification = self.classification
        classification = classification.assign_coords(molecule=self.molecule)
        traces = getattr(self, variable)

        if selected:
            classification = classification.sel(molecule=self.selected)
            traces = traces.sel(molecule=self.selected)

        dwells = dwell_times_from_classification(classification, traces=traces, cycle_time=self.cycle_time,
                                                 inactivate_start_and_end_states=inactivate_start_and_end_states)

        dwells['number_of_states'] = self.number_of_states_from_classification.sel(molecule=dwells.molecule)\
            .reset_coords(drop=True)

        add_configuration_to_dataarray(dwells, File.determine_dwells_from_classification, locals())

        if not selected:
            dwells.attrs['applied_selections'] = json.dumps([])
        elif 'configuration' in self.selected.attrs:
            dwells.attrs['applied_selections'] = self.selected.attrs['configuration']
        if 'configuration' in self.classification.attrs:
            dwells.attrs['applied_classifications'] = self.classification.attrs['configuration']

        dwells.to_netcdf(self.absoluteFilePath.with_name(self.name + '_dwells').with_suffix('.nc'), engine='netcdf4', mode='w')

    def classification_binary(self, positive_states_only=False, selected=False):
        states_in_file = xr.DataArray(np.unique(self.classification), dims='state')
        if positive_states_only:
            states_in_file = states_in_file[states_in_file >= 0]
        classification_binary = (self.classification == states_in_file).assign_coords(state=states_in_file)
        if selected:
            classification_binary = classification_binary.sel(molecule=self.selected)
        return classification_binary.transpose(..., 'frame')

    @property
    @return_none_when_executed_by_pycharm
    def number_of_states_from_classification(self):
        molecule_has_state = self.classification_binary(positive_states_only=True).any(dim='frame')
        number_of_states = molecule_has_state.sum(dim='state')
        return number_of_states

    @property
    @return_none_when_executed_by_pycharm
    def dwells(self):
        return xr.load_dataset(self.absoluteFilePath.with_name(self.name + '_dwells').with_suffix('.nc'), engine='netcdf4')

    def analyze_dwells(self, method='maximum_likelihood_estimation', number_of_exponentials=[1,2], state_names=None,
                       truncation=None, P_bounds=(-1, 1), k_bounds=(1e-9, np.inf), plot=False,
                       fit_dwell_times_kwargs={}, plot_dwell_analysis_kwargs={}, save_file_path=None):

        dwells = self.dwells

        # At the moment single-state states are already set at -128 so they don't need to be separated.
        # For >2 states we will need to do this.
        # for n in np.arange(dwells.number_of_states.max().item())+1:
        #     dwells['state'][dict(dwell=(dwells['number_of_states'] == n) & dwells['state'] >= 0)] += n-1

        #TODO: Add sampling interval to File and refer to it here?
        dwell_analysis = analyze_dwells(dwells, method=method, number_of_exponentials=number_of_exponentials,
                                        state_names=state_names, P_bounds=P_bounds, k_bounds=k_bounds,
                                        sampling_interval=None, truncation=truncation, fit_dwell_times_kwargs=fit_dwell_times_kwargs)

        add_configuration_to_dataarray(dwell_analysis, File.analyze_dwells, locals())

        if 'applied_selections' in dwells.attrs:
            dwell_analysis.attrs['applied_selections'] = dwells.attrs['applied_selections']
        if 'applied_classifications' in dwells.attrs:
            dwell_analysis.attrs['applied_classifications'] = dwells.attrs['applied_classifications']
        if 'configuration' in dwells.attrs:
            dwell_analysis.attrs['dwells_configuration'] = dwells.attrs['configuration']

        if save_file_path is None:
            self.dwell_analysis = dwell_analysis
            if plot:
                self.plot_dwell_analysis(**plot_dwell_analysis_kwargs)
        else:
            dwell_analysis.to_netcdf(self.absoluteFilePath.with_name(save_file_path).with_suffix('.nc'),
                                     engine='netcdf4', mode='w')
            if plot:
                plot_dwell_analysis(dwell_analysis, dwells, **plot_dwell_analysis_kwargs)

        return dwell_analysis

    def plot_dwell_analysis(self, name=None, plot_type='pdf', plot_range=None, axes=None, bins='auto_discrete',
                            log=False, sharey=False, save_path=None):
        dwell_analysis = self.dwell_analysis
        dwells = self.dwells

        if name is None:
            name = self.name

        # axes[0].set_title(name)
        if save_path is None:
            save_path = self.experiment.analysis_path / 'Dwell time analysis'

        fig, axes = plot_dwell_analysis(dwell_analysis, dwells, plot_type=plot_type, plot_range=plot_range, axes=axes,
                                        bins=bins, log=log, sharey=sharey, name=name, save_path=save_path)

        return axes[0].figure, axes

    @property
    @return_none_when_executed_by_pycharm
    def dwell_analysis(self):
        # return pd.read_excel(self.absoluteFilePath.with_name(self.name + '_dwell_analysis').with_suffix('.xlsx'))
        return xr.load_dataset(self.absoluteFilePath.with_name(self.name + '_dwell_analysis').with_suffix('.nc'), engine='netcdf4')

    @dwell_analysis.setter
    def dwell_analysis(self, dwell_analysis):
        # dwell_analysis.to_excel(self.absoluteFilePath.with_name(self.name + '_dwell_analysis').with_suffix('.xlsx'))
        # dataset = xr.DataArray(dataset, dims=('exponential', ' variable'))
        dwell_analysis.to_netcdf(self.absoluteFilePath.with_name(self.name + '_dwell_analysis').with_suffix('.nc'),
                                 engine='netcdf4', mode='w')

    def state_count(self, selected=True, states=None):
        # if hasattr(self, 'number_of_states'):
        n, c = np.unique(self.get_variable('number_of_states', selected=selected), return_counts=True)
        if states is None:
            states = n
        # else:
        #     if states is None:
        #         states = np.array([0])
        #     n = states
        #     c = np.array([0] * len(states))

        state_count = xr.DataArray(0, dims=('name','number_of_states'), coords={'name': [self.name], 'number_of_states': states})
        state_count.loc[dict(number_of_states=n)] = c
        state_count.name = 'state_count'
        return state_count

    def state_fraction(self, **state_count_kwargs):
        state_count = self.state_count(**state_count_kwargs)
        state_fraction = state_count / state_count.sum('number_of_states')
        state_fraction.name = 'state_fraction'
        return state_fraction

    #
    # def get_FRET(self, **kwargs):
    #     return self.get_variable('FRET', **kwargs)
    #
    # def get_intensity(self, **kwargs):
    #     intensity = self.get_variable('intensity', **kwargs)
    #     if hasattr(self, 'background_correction') and self.background_correction is not None:
    #         intensity[dict(channel=0)] -= self.background_correction[0]
    #         intensity[dict(channel=1)] -= self.background_correction[1]
    #     if hasattr(self, 'alpha_correction') and self.alpha_correction is not None:
    #         intensity[dict(channel=0)] += self.alpha_correction * intensity[dict(channel=0)]
    #         intensity[dict(channel=1)] -= self.alpha_correction * intensity[dict(channel=0)]
    #     # if hasattr(self, 'delta_correction') and self.delta_correction is not None:
    #     #     intensity[dict(channel=0)] *= self.delta_correction
    #     if hasattr(self, 'gamma_correction') and self.gamma_correction is not None:
    #         intensity[dict(channel=0)] *= self.gamma_correction
    #     # if hasattr(self, 'beta_correction') and self.beta_correction is not None:
    #     #     intensity[dict(channel=0)] *= self.beta_correction
    #
    #     return intensity

    def determine_trace_correction(self):
        from papylio.trace_correction import TraceCorrectionWindow
        # TODO: Should work on intensity raw if intensity raw is there else on intensity.
        TraceCorrectionWindow(self.intensity)

    def show_histogram(self, variable, selected=False, frame_range=None, average=False, axis=None, **hist_kwargs):
        # TODO: add save
        da = self.get_variable(variable, selected=selected, frame_range=frame_range, average=average)
        figure, axis = histogram(da, axis=axis, **hist_kwargs)
        axis.set_title(str(self.relativeFilePath))
        return figure, axis

    def histogram_2D_FRET_intensity_total(self, selected=False, frame_range=None, average=False,
                                       **marginal_hist2d_kwargs):
        """
        Generates a 2D histogram plot of FRET vs. total intensity with optional marginal histograms.

        This function retrieves the 'FRET' and 'intensity_total' variables from the File object, then plots their
        relationship in a 2D histogram, with optional marginal histograms along the axes.

        Parameters:
        -----------
        selected : bool, optional (default=False)
            If True, only selected molecules will be used for plotting.
        frame_range : tuple of two ints, optional (default=None)
            The range of frames to use. If None, all frames are used.
        average : bool, optional (default=False)
            If True, the function averages the data over the specified frame range.
        axis : matplotlib.axes.Axes, optional (default=None)
            The axes object to plot on. If None, a new plot will be created.
        **marginal_hist2d_kwargs : dict, optional
            Additional keyword arguments passed to the `marginal_hist2d` function for customizing the plot.
            Default arguments are used for the 2D histogram's range.

        Returns:
        --------
        axes : list of matplotlib.axes.Axes
            A list of axes objects corresponding to the 2D histogram plot and optional marginal histograms.

        Notes:
        ------
        The function utilizes the `marginal_hist2d` function from the `papylio.plotting` module to create the plot.
        The default range for the FRET values is (-0.05, 1.05) for the x-axis and no limit for the y-axis.
        """

        FRET = self.get_variable('FRET', selected=selected, frame_range=frame_range, average=average)
        intensity_total = self.get_variable('intensity_total', selected=selected, frame_range=frame_range, average=average)

        marginal_hist2d_kwargs_default = dict(range=((-0.05, 1.05), None))
        marginal_hist2d_kwargs = {**marginal_hist2d_kwargs_default, **marginal_hist2d_kwargs}

        from papylio.plotting import marginal_hist2d
        figure, axes = marginal_hist2d(FRET, intensity_total, **marginal_hist2d_kwargs)

        return axes


    def histogram_2D_intensity_per_channel(self, selected=False, frame_range=None, average=False,
                                           channel_x=0, channel_y=1, **marginal_hist2d_kwargs):
        """
        Generates a 2D histogram plot of intensity between two specified channels, with optional marginal histograms.

        This function retrieves intensity data for the specified channels from the File object and generates a 2D histogram
        to visualize the relationship between intensities in the selected channels. Marginal histograms along the axes can
        optionally be included for additional insight.

        Parameters:
        -----------
        selected : bool, optional (default=False)
            If True, only selected molecules are used for the plot.
        frame_range : tuple of two ints, optional (default=None)
            Specifies the range of frames to use. If None, all frames are included.
        average : bool, optional (default=False)
            If True, averages the intensity data over the specified frame range.
        channel_x : int, optional (default=0)
            The index of the channel for the x-axis data.
        channel_y : int, optional (default=1)
            The index of the channel for the y-axis data.
        **marginal_hist2d_kwargs : dict, optional
            Additional keyword arguments passed to the `marginal_hist2d` function to customize the plot.
            Defaults include no specific range for the histogram axes.

        Returns:
        --------
        axes : list of matplotlib.axes.Axes
            A list of axes objects corresponding to the 2D histogram plot and optional marginal histograms.

        Notes:
        ------
        - The function uses the `marginal_hist2d` function from the `papylio.plotting` module for visualization.
        """

        intensity_x = self.get_variable('intensity', selected=selected, frame_range=frame_range, average=average).sel(channel=channel_x)
        intensity_x.name = intensity_x.name + f'_c{channel_x}'
        intensity_y = self.get_variable('intensity', selected=selected, frame_range=frame_range, average=average).sel(channel=channel_y)
        intensity_y.name = intensity_y.name + f'_c{channel_y}'

        marginal_hist2d_kwargs_default = dict(range=(None, None))
        marginal_hist2d_kwargs = {**marginal_hist2d_kwargs_default, **marginal_hist2d_kwargs}

        from papylio.plotting import marginal_hist2d
        figure, axes = marginal_hist2d(intensity_x, intensity_y, **marginal_hist2d_kwargs)

        return axes


    def show_image(self, projection_type='default', figure=None, unit='pixel', **kwargs):
        # TODO: Show two channels separately and connect axes
        # Refresh configuration
        if projection_type == 'default':
            projection_type = self.experiment.configuration['projection_image']['projection_type']

        if figure is None:
            figure = plt.figure()
        axis = figure.gca()

        # Choose method to plot
        if projection_type == 'average':
            image = self.average_image()
            axis.set_title('Average image')
        elif projection_type == 'maximum':
            image = self.maximum_projection_image()
            axis.set_title('Maximum projection')


        if unit == 'pixel':
            unit_string = ' (pixels)'
        elif unit == 'metric':
            kwargs['extent'] = self.movie.boundaries_metric.T.flatten()[[0,1,3,2]]
            unit_string = f' ({self.movie.pixel_size_unit})'

        axis.imshow(image, **kwargs)
        axis.set_title(self.relativeFilePath)
        axis.set_xlabel('x'+unit_string)
        axis.set_ylabel('y'+unit_string)

        # TODO: Remove following commented out code
        # as the vmin and vmax can now be set by passing the kwargs to imshow.
        #
        # image_handle = axis.imshow(image)
        #
        # # process keyword arguments
        # colorscale = list(image_handle.get_clim())
        # if 'vmin' in kwargs:
        #     colorscale[0] = kwargs['vmin']
        # if 'vmax' in kwargs:
        #     colorscale[1] = kwargs['vmax']
        # image_handle.set_clim(colorscale)

        return figure, axis

    def show_average_image(self, figure=None, **kwargs):
        self.show_image(projection_type='average', figure=figure, **kwargs)

    def show_coordinates(self, figure=None, annotate=None, unit='pixel', **kwargs):
        if not figure:
            figure = plt.figure()

        if annotate is None:
            annotate = self.experiment.configuration['show_movie']['annotate']

        if self.coordinates is not None:
            axis = figure.gca()
            if unit == 'pixel':
                coordinates = self.coordinates
            elif unit == 'metric':
                coordinates = self.coordinates_metric
            else:
                raise ValueError('Unit can be either "pixel" or "metric"')

            coordinates = coordinates.stack({'peak': ('molecule', 'channel')}).T.values
            sc_coordinates = axis.scatter(coordinates[:, 0], coordinates[:, 1], facecolors='none', edgecolors='red', **kwargs)
            # marker='o'

            selected_coordinates = self.coordinates.sel(molecule=self.selected.values).stack({'peak': ('molecule', 'channel')}).T.values
            axis.scatter(selected_coordinates[:, 0], selected_coordinates[:, 1], facecolors='none', edgecolors='green', **kwargs)

            if annotate:
                annotation = axis.annotate("", xy=(0, 1.03), xycoords=axis.transAxes) # x in data units, y in axes fraction
                annotation.set_visible(False)

                #molecule_indices = np.repeat(np.arange(0, self.number_of_molecules), self.number_of_channels)
                molecule_indices = np.repeat(self.molecule_in_file.values, self.number_of_channels)
                # sequence_indices = np.repeat(self.sequence_indices, self.number_of_channels)

                def update_annotation(ind):
                    # print(ind)

                    # text = "Molecule number: {} \nSequence: {}".format(" ".join([str(indices[ind["ind"][0]])]),
                    #                        " ".join([str(sequences[ind["ind"][0]].decode('UTF-8'))]))
                    plot_index = ind["ind"]
                    molecule_index = molecule_indices[plot_index]
                    text = f'Molecule number: {", ".join(map(str, molecule_index))}'

                    if hasattr(self, 'sequences'):
                        sequence_names = [str(self.sequence_name[index]) for index in molecule_index]
                        sequences = [str(self.sequence[index]) for index in molecule_index]
                        text += f'\nSequence name: {", ".join(sequence_names)}'
                        text += f'\nSequence: {", ".join(sequences)}'

                    annotation.set_text(text)

                def hover(event):
                    vis = annotation.get_visible()
                    if event.inaxes == axis:
                        cont, ind = sc_coordinates.contains(event)
                        if cont:
                            update_annotation(ind)
                            annotation.set_visible(True)
                            figure.canvas.draw_idle()
                        else:
                            if vis:
                                annotation.set_visible(False)
                                figure.canvas.draw_idle()

                figure.canvas.mpl_connect("motion_notify_event", hover)

            plt.show()

    def show_coordinates_in_image(self, figure=None, **kwargs):
        #TODO: change figure to axis
        if not figure:
            figure = plt.figure()

        self.show_image(figure=figure, **kwargs)
        self.show_coordinates(figure=figure)
        # plt.savefig(self.writepath.joinpath(self.name + '_ave_circles.png'), dpi=600)

    def show_traces(self, plot_variables=['intensity', 'FRET'], ylims=[(0, 35000), (0, 1)], colours=[('g', 'r'), ('b')],
                    selected=False, split_illuminations=True, **kwargs):
        # probably better to put selected and illumination options in TracePlotWindow
        dataset = xr.Dataset()
        for variable in plot_variables:
            dataset[variable] = getattr(self, variable)
        selected_original = self.selected
        dataset['selected'] = selected_original
        if selected:
            dataset = dataset.sel(molecule=dataset['selected'])

        if split_illuminations:
            illuminations_in_file = np.unique(dataset.illumination)
            if len(illuminations_in_file) > 1:
                for plot_variable in ['intensity', 'FRET']:
                    if plot_variable in plot_variables:
                        plot_variable_index = plot_variables.index(plot_variable)
                        for j, illumination_index in enumerate(illuminations_in_file):
                            name = f'{plot_variable}_i{illumination_index}'
                            dataset[name] = dataset[plot_variable].sel(frame=dataset.illumination == illumination_index)
                            plot_variables.insert(plot_variable_index + j + 1, name)
                            if j > 0:
                                ylims.insert(plot_variable_index + j, ylims[plot_variable_index])
                                colours.insert(plot_variable_index + j, colours[plot_variable_index])
                        plot_variables.pop(plot_variable_index)

        # dataset = self.dataset
        save_path = self.experiment.main_path.joinpath('Trace plots')
        if not save_path.is_dir():
            save_path.mkdir()

        from papylio.trace_plot import TracePlotWindow
        TracePlotWindow(dataset=dataset, plot_variables=plot_variables, ylims=ylims, colours=colours, save_path=save_path, **kwargs)
        if selected:
            selected_original[dict(molecule=selected_original)] = dataset.selected
        else:
            selected_original = dataset.selected

        # We could also save the whole dataset, but since currently only alterations are made to selected.
        selected_original.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='a')


def calculate_intensity_total(intensity):
    intensity_total = intensity.sum(dim='channel')
    intensity_total.name = 'intensity_total'
    intensity_total.attrs = intensity.attrs
    return intensity_total


def calculate_FRET(intensity):
    # TODO: Make suitable for mutliple colours
    donor = intensity.sel(channel=0, drop=True)
    acceptor = intensity.sel(channel=1, drop=True)
    FRET = acceptor / (donor + acceptor)
    FRET = FRET.where(FRET >= 0, 0)
    FRET.name = 'FRET'
    return FRET

def calculate_stoichiometry(intensity):
    intensity_total = calculate_intensity_total(intensity)
    intensity_total_i0 = intensity_total.sel(frame=intensity.illumination == 0)
    intensity_total_i1 = intensity_total.sel(frame=intensity.illumination == 1).values

    stoichiometry = intensity_total_i0 / (intensity_total_i0 + intensity_total_i1)
    stoichiometry.name = 'stoichiometry'
    return stoichiometry

def import_pks_file(pks_filepath):
    pks_filepath = Path(pks_filepath)
    data = np.genfromtxt(pks_filepath)
    if len(data) == 0:
        return xr.DataArray(np.empty((0,3)), dims=("peak",'parameter'), coords={'parameter': ['x', 'y', 'background']})
    data = np.atleast_2d(data)[:,1:]
    if data.shape[1] == 2:
        data = np.hstack([data, np.zeros((len(data),1))])
    return xr.DataArray(data, dims=("peak",'parameter'),
                        coords={'peak': range(len(data)), 'parameter': ['x', 'y', 'background']})


def export_pks_file(peaks, pks_filepath):
    pks_filepath = Path(pks_filepath)
    with pks_filepath.open('w') as pks_file:
        for i, (x, y, background) in enumerate(peaks.values):
            # outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format(i, coordinate[0], coordinate[1], 0, 0, width4=4, width6=6))
            # pks_file.write('{0:4.0f} {1:4.4f} {2:4.4f} \n'.format(i + 1, coordinate[0], coordinate[1]))
            pks_file.write('{0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f}\n'.format(i + 1, x, y, background))


def import_traces_file(traces_filepath):
    traces_filepath = Path(traces_filepath)
    with traces_filepath.open('r') as traces_file:
        number_of_frames = np.fromfile(traces_file, dtype=np.int32, count=1).item()
        number_of_traces = np.fromfile(traces_file, dtype=np.int16, count=1).item()
        # number_of_molecules = number_of_traces // number_of_channels
        raw_data = np.fromfile(traces_file, dtype=np.int16, count=number_of_frames * number_of_traces)
    # traces = np.reshape(rawData.ravel(),
    #                         (number_of_channels, number_of_molecules, number_of_frames),
    #                         order='F')  # 3d array of trace # 2d array of traces
    traces = np.reshape(raw_data.ravel(), (number_of_traces, number_of_frames), order='F') # 2d array of traces
    traces = xr.DataArray(traces, dims=("trace", "frame"), coords=(range(number_of_traces), range(number_of_frames)))
    return traces


def export_traces_file(traces, traces_filepath):
    traces_filepath = Path(traces_filepath)
    with traces_filepath.open('w') as traces_file:
        # Number of frames
        np.array([len(traces.frame)], dtype=np.int32).tofile(traces_file)
        # Number of traces
        np.array([len(traces.trace)], dtype=np.int16).tofile(traces_file)
        traces.values.T.astype(np.int16).tofile(traces_file)

def split_dimension(data_array, old_dim, new_dims, new_dims_shape=None, new_dims_coords=None, to='dimensions'):
    all_dims = list(data_array.dims)
    old_dim_index = all_dims.index(old_dim)
    all_dims[old_dim_index:old_dim_index + 1] = new_dims
    new_dims_shape = np.array(new_dims_shape)
    if sum(new_dims_shape == -1) == 1:
        fixed_dim_prod = np.prod(new_dims_shape[new_dims_shape!=-1])
        old_len = data_array.shape[data_array.dims==old_dim]
        if old_len % fixed_dim_prod != 0:
            raise ValueError('Incorrect dimension shape')
        new_dims_shape[new_dims_shape == -1] = old_len // fixed_dim_prod
    elif sum(new_dims_shape == -1) > 1:
        raise ValueError

    if new_dims_coords is None:
        new_dims_coords = [-1]*len(new_dims_shape)
    new_dims_coords = (range(new_dims_shape[i]) if new_dims_coord == -1 else new_dims_coord
                       for i, new_dims_coord in enumerate(new_dims_coords))

    new_dims_coords = [np.arange(new_dims_shape[i]) if new_dims_coord == -1 else new_dims_coord
                       for i, new_dims_coord in enumerate(new_dims_coords)]

    new_index = pd.MultiIndex.from_product(new_dims_coords, names=new_dims)
    data_array = data_array.assign_coords(**{old_dim: new_index})

    if to == 'dimensions':
        # Unstack does not work well for empty data_arrays, but in principle all necessary information is contained in the multiindex, i.e. range of all dimensions.
        return data_array.unstack(old_dim).transpose(*all_dims)
    elif to == 'multiindex':
        return data_array
    else:
        raise ValueError
