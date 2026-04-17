"""File and I/O utilities for Papylio.

This module defines the File class which represents all the data related to a single experimental movie file
and provides methods for reading movies, extracting coordinates, performing
trace extraction and analysis, and saving results.
"""

if __name__ == '__main__':
    import sys
    from pathlib import Path
    p = Path(__file__).parents[1]
    sys.path.insert(0, str(p))

from pathlib import Path # For efficient path manipulation
import numpy as np #scientific computing with Python
import pandas as pd
import matplotlib.pyplot as plt
import ast
import xarray as xr
import warnings
import sys
import re
from typing import Literal
import logging
import inspect
import tifffile
import netCDF4
import json
import papylio
import matchpoint as mp
from papylio.movie.movie import Movie
from papylio.plotting import histogram
from papylio.peak_finding import find_peaks
from papylio.coordinate_optimization import  coordinates_within_margin, \
                                                    coordinates_after_gaussian_fit, \
                                                    coordinates_without_intensity_at_radius, \
                                                    merge_nearby_coordinates, \
                                                    set_of_tuples_from_array, array_from_set_of_tuples, \
                                                    coordinates_within_margin_selection
from papylio.trace_extraction import extract_traces
from papylio.log_functions import add_configuration_to_dataarray
from papylio.background_subtraction import extract_background
from papylio.plugin_manager import plugins
from papylio.analysis.dwell_time_extraction import dwell_times_from_classification
from papylio.analysis.dwell_time_analysis import analyze_dwells, plot_dwell_time_histogram, plot_dwell_analysis
from papylio.decorators import return_none_when_executed_by_pycharm
from papylio.helper_functions import get_default_parameters
from papylio.analysis.classification_simple import classify_threshold
from papylio.analysis.hidden_markov_modelling import classify_hmm


@plugins
class File:
    """
    A class representing all the data related to a single-molecule data file, handling movie imports,
    coordinate finding, trace extraction, and analysis.
    """

    unit_mapping = mp.MatchPoint()

    def __init__(self, relative_filepath, extensions=None, experiment=None, perform_logging=True):

        """
        Initialize a File object.

        Parameters:
            relativeFilePath (str or Path): Path to the file relative to the experiment root.
            extensions (set, optional): Set of file extensions associated with this file.
            experiment (Experiment, optional): The experiment object this file belongs to.
            perform_logging (bool, optional): Whether to log activities for this file. Default is True.
        """
        self.perform_logging = False # It is set to False temporarily until the end of __init__.

        self.dataset_variables = ['molecule', 'frame', 'time', 'coordinates', 'background', 'intensity', 'FRET', 'selected',
                                  'molecule_in_file', 'illumination_correction', 'number_of_states', 'transition_rate', 'state_mean', 'classification']

        relative_filepath = Path(relative_filepath)
        self.experiment = experiment

        self.relativePath = relative_filepath.parent
        self.name = relative_filepath.name
        self.extensions = set()

        # self.exposure_time = None  # Found from log file or should be inputted

        self.number_of_frames = None

        self.is_selected = False
        # self.is_mapping_file = False

        self.movie = None
        # self.mapping = None

        self._rotation = 0

        self._mappings = None
        self.mappings = [self.unit_mapping, ] * (self.number_of_channels - 1)

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
                                # '.coeff': self.import_coeff_file,
                                # '.map': self.import_map_file,
                                # '.mapping': self.import_mapping_file,
                                '.pks': self.import_pks_file,
                                '.traces': self.import_traces_file,
                                '.nc': self.none_function
                                }

        if extensions is None:
            extensions = self.find_extensions()
        self.add_extensions(extensions, load=self.experiment.import_all)

        self.perform_logging = perform_logging
        self.__logger = None
        self._log('info', f"Initialized {self} with Papylio v{papylio.__version__}")

    def __repr__(self):
        """Return a string representation of the File object."""
        return (f'{self.__class__.__name__}({self.relativePath.joinpath(self.name)})')

    @property
    @return_none_when_executed_by_pycharm
    def _log_filepath(self):
        """Return the path to the log file."""
        return self.absolute_filepath.with_suffix(".log")

    @property
    def _logger(self):
        """Create a dedicated logger per File instance."""
        if self.__logger is None:
            logger_name = f"FileLogger.{self.relative_filepath}"
            self.__logger = logging.getLogger(logger_name)
            self.__logger.setLevel(logging.INFO)
        return self.__logger

    def _log(self, log_type, message):
        """
        Log a message to the file's log.

        Parameters:
            log_type (str): The logging level (e.g., 'info', 'warning', 'error').
            message (str): The message to log.
        """
        if self.perform_logging:
            handler = logging.FileHandler(self._log_filepath, mode="a", encoding="utf-8")
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)

            getattr(self._logger, log_type)(message)

            handler.close()
            self._logger.removeHandler(handler)

    @property
    @return_none_when_executed_by_pycharm
    def relative_filepath(self):
        """Return the path to the file relative to the experiment root."""
        return self.relativePath.joinpath(self.name)

    @property
    @return_none_when_executed_by_pycharm
    def absolute_filepath(self):
        """Return the absolute path to the file."""
        return self.relative_filepath.absolute()

    @property
    @return_none_when_executed_by_pycharm
    def number_of_molecules(self):
        """Return the number of molecules in the file's dataset."""
        try:
            with netCDF4.Dataset(self.absolute_filepath.with_suffix('.nc')) as dataset:
                return dataset.dimensions['molecule'].size
        except FileNotFoundError:
            return 0

    @property
    @return_none_when_executed_by_pycharm
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        self.movie.rot90 = rotation
        self._rotation = rotation

    @property
    @return_none_when_executed_by_pycharm
    def number_of_channels(self):
        """Return the number of channels in the experiment."""
        #TODO this needs to be independent from Experiment and should probably be set.
        return self.experiment.number_of_channels

    def projection_image(self, **kwargs):
        """Return the default projection image."""
        return self.get_projection_image(**kwargs)

    def average_image(self, **kwargs):
        """Return the average projection image."""
        return self.get_projection_image(projection_type='average', **kwargs)

    def maximum_projection_image(self, **kwargs):
        """Return the maximum projection image."""
        return self.get_projection_image(projection_type='maximum', **kwargs)

    def get_projection_image(self, load=True, **projection_image_configuration):
        """
        Get or generate a projection image.

        Parameters:
            load (bool, optional): Whether to try loading an existing image from disk. Default is True.
            **kwargs: Additional configuration parameters for image projection.

        Returns:
            numpy.ndarray: The projection image.
        """
        # TODO: Add option to flatten channels?
        if load:
            image = Movie.load_projection_image(self.absolute_filepath, **projection_image_configuration)
        else:
            image = None

        if image is None:
            image = self.movie.save_projection_image(**projection_image_configuration)

        return image

    @property
    @return_none_when_executed_by_pycharm
    def coordinates_metric(self):
        """Return the molecule coordinates in metric units (e.g., nanometers)."""
        return self.coordinates * self.movie.pixel_size

    @property
    @return_none_when_executed_by_pycharm
    def coordinates_stage(self):
        """Return the molecule coordinates in stage units."""
        coordinates = self.coordinates.sel(channel=0)
        coordinates_stage = self.movie.pixel_to_stage_coordinates_transformation(coordinates)
        return xr.DataArray(coordinates_stage, coords=coordinates.coords)

    def set_coordinates_of_channel(self, coordinates, channel):
        """
        Set coordinates for a specific channel and update other channels using mapping.

        Parameters:
            coordinates (numpy.ndarray or xarray.DataArray): The coordinates to set.
            channel (int or str): The channel index or name.
        """
        if not isinstance(coordinates, xr.DataArray):
            coordinates = xr.DataArray(coordinates, dims=('molecule', 'dimension'))
        channel_index = self.movie.get_channel_indices_from_names(channel)[0]
        if channel_index > 0:
            coordinates = self.mappings[channel_index-1].transform_coordinates(coordinates, inverse=True)

        coordinates = coordinates.expand_dims(channel=np.arange(self.number_of_channels), axis=1).copy()

        for i in range(self.number_of_channels)[1:]:
            coordinates[:,i,:] = self.mappings[i-1].transform_coordinates(coordinates[:,i,:], inverse=False)

        coordinates = coordinates_within_margin(coordinates, bounds=self.movie.channels[0].boundaries, margin=0)
        self.coordinates = coordinates

    def coordinates_from_channel(self, channel):
        """
        Get coordinates for a specific channel.

        Parameters:
            channel (int or str): The channel index or name.

        Returns:
            xarray.DataArray: The coordinates for the specified channel.
        """

        channel = self.movie.get_channel_from_name(channel).index
        return self.coordinates.sel(channel=channel)

    def __getstate__(self):
        """Return the object's state for pickling."""
        return self.__dict__.copy()

    def __setstate__(self, dict):
        """Set the object's state during unpickling."""
        self.__dict__.update(dict)

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

    def __setattr__(self, name, value):
        """
        Set an attribute and log the change if it's an external assignment.

        Parameters:
            name (str): The name of the attribute.
            value: The value to set.
        """
        super().__setattr__(name, value)
        # # Skip logger itself
        # if name != "_logger" and self.perform_logging:
        #     # Check if the assignment comes from outside this instance
        #     stack = inspect.stack()
        #     external = all(frame.frame.f_locals.get("self") is not self for frame in stack[1:])
        #     if external:
        #         self._log('info', f"Set attribute {name} = {value!r}")

        if name.startswith("_") or name == "perform_logging":
            return
        if self.__dict__.get("perform_logging"):  # avoids descriptor overhead
            caller = sys._getframe(1).f_locals.get("self")
            # if not isinstance(caller, File):
            if caller is not self:
                self._log('info', f"Set attribute {name} = {value!r}")

    def get_data(self, key):
        """
        Retrieve data from the netCDF dataset.

        Parameters:
            key (str): The name of the data variable to retrieve.

        Returns:
            xarray.DataArray: The retrieved data.
        """
        with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return dataset[key].load()

    @property
    @return_none_when_executed_by_pycharm
    def dataset(self):
        """Return the full xarray dataset for this file."""
        if self.absolute_filepath.with_suffix('.nc').exists():
            with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
                return dataset.load()
        else:
            return None

    @property
    @return_none_when_executed_by_pycharm
    def dataset_selected(self):
        """Return the xarray dataset containing only selected molecules."""
        dataset = self.dataset
        return dataset.sel(molecule=dataset.selected)

    @property
    @return_none_when_executed_by_pycharm
    def data_vars(self):
        """Return the data variables of the netCDF dataset."""

        with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return dataset.data_vars

    @property
    @return_none_when_executed_by_pycharm
    def dataset_attributes(self):
        """Return the global attributes of the netCDF dataset."""
        if self.absolute_filepath.with_suffix('.nc').exists():
            with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
                return dataset.attrs
        else:
            return {}

    def get_dataset_attribute(self, attribute_name):
        #TODO: Check whether this is necessary
        dataset_filepath = self.absoluteFilePath.with_suffix('.nc')
        if dataset_filepath.exists():
            with netCDF4.Dataset(dataset_filepath) as dataset:
                return dataset.getncattr(attribute_name) if attribute_name in dataset.ncattrs() else None
        return None

    def _init_dataset(self, number_of_molecules):
        """
        Initialize the netCDF dataset for this file.

        Parameters:
            number_of_molecules (int): The number of molecules to initialize the dataset with.
        """
        selected = xr.DataArray(False, dims=('molecule',), coords={'molecule': range(number_of_molecules)}, name='selected')
        add_configuration_to_dataarray(selected)
        selected.attrs['configuration'] = json.dumps([])
        selected.attrs['selection_configurations'] = json.dumps({})

        dataset = selected.to_dataset().assign_coords(molecule_in_file=('molecule', selected.molecule.values))
        dataset = dataset.reset_index('molecule', drop=True)
        dataset = dataset.assign_coords({'file': ('molecule', [str(self.relative_filepath).encode()] * number_of_molecules)})
        encoding = {'file': {'dtype': '|S'}, 'selected': {'dtype': bool}}
        dataset.attrs['channel_arrangement'] = json.dumps(self.movie.channel_arrangement.tolist())
        dataset.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='w', encoding=encoding)
        self.extensions.add('.nc')

    def find_extensions(self):
        """
        Scan the experiment directory for files matching this file's name and return their extensions.

        Returns:
            list: A list of found file extensions.
        """
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
        """Find associated file extensions and add them to this File object."""
        self.add_extensions(self.find_extensions())

    def add_extensions(self, extensions, load=True):
        """
        Add extensions to this file and optionally load the data.

        Parameters:
            extensions (str or list): One or more extensions to add.
            load (bool, optional): Whether to import the data associated with the extensions. Default is True.
        """
        if isinstance(extensions, str):
            extensions = [extensions]
        for extension in set(extensions)-self.extensions:
            if load:
                self.importFunctions.get(extension, self.none_function)(extension)
            self.extensions.add(extension) # Maybe not necessary?

    def none_function(self, *args, **kwargs):
        """A placeholder function that does nothing."""
        return

    def import_movie(self, extension):
        """
        Import movie data associated with the given extension.

        Parameters:
            extension (str): The file extension to import from.
        """
        if extension == '.sifx':
            filepath = self.absolute_filepath.joinpath('Spooled files.sifx')
        # elif extension == '.nd2' and '_fov' in self.name:
        #     # TODO: Make this working
        #     token_position = self.name.find('_fov')
        #     movie_name = self.name[:token_position]
        #     filepath = self.absoluteFilePath.with_name(movie_name).with_suffix(extension)
            # self.movie = ND2Movie(imageFilePath, fov_info=self.nd2_fov_info)
        else:
            filepath = self.absolute_filepath.with_suffix(extension)

        self.movie = Movie(filepath, self.rotation)
        if 'channel_arrangement' in self.dataset_attributes.keys():
            channel_arrangement_text_string=self.dataset_attributes['channel_arrangement']
            self.movie.channel_arrangement = ast.literal_eval(channel_arrangement_text_string)
        # self.number_of_frames = self.movie.number_of_frames

    def import_coeff_file(self, extension):
        """
        Import a coefficient file for linear coordinate mapping.

        Parameters:
            extension (str): The file extension (usually '.coeff').
        """
        # TODO: Move this to the MatchPoint class
        from skimage.transform import AffineTransform
        if self.mapping is None: # the following only works for 'linear'transformation_type
            file_content=np.genfromtxt(str(self.absolute_filepath) + '.coeff')
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

    def export_mapping(self, filetype='nc'):
        """
        Export the current coordinate mapping.

        Parameters:
            filetype (str, optional): The format to save the mapping in (e.g., 'yml', 'classic'). Default is 'yml'.
        """
        for i, mapping in enumerate(self.mappings):
            filename = f'channel_mapping_c{i+1}_' + mapping.source_name + '_to_' + mapping.destination_name
            mapping.save(self.experiment.main_path / filename, filetype)

    def import_map_file(self, extension):
        """
        Import a map file for nonlinear coordinate mapping.

        Parameters:
            extension (str): The file extension (usually '.map').
        """
        # TODO: Move this to the MatchPoint class
        #coefficients = np.genfromtxt(self.absoluteFilePath.with_suffix('.map'))
        file_content=np.genfromtxt(self.absolute_filepath.with_suffix('.map'))
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

    # def import_mapping_file(self, extension):
    #     self.mapping = mp.MatchPoint.load(self.absolute_filepath.with_suffix(extension))

    def use_for_darkfield_correction(self):
        """Use the average projection of this file as a darkfield correction image for the experiment."""
        image = self.get_projection_image(projection_type='average', frame_range=(0, None), apply_corrections=False)
        tifffile.imwrite(self.experiment.main_path / 'darkfield.tif', image, imagej=True)
        self.experiment.load_darkfield_correction()

    def find_coordinates(self, channels=('donor', 'acceptor'),
                         projection_image_configuration=None, sliding_window=None,
                         peak_finding_configuration=None, margin=10, fit_peaks=True, remove_peaks_with_close_neighbors=None):
        """
        Find and set the locations of all molecules within the movie's images.

        This function performs peak finding on projection images, handles multiple channels,
        and manages coordinate sets across different frames if sliding windows are used.

        For configuration options see the "find_coordinates" section in the default configuration file.
        """

        if projection_image_configuration is None:
            projection_image_configuration = dict()

        if peak_finding_configuration is None:
            peak_finding_configuration = dict()

        if len(channels) > 1:
            projection_image_configuration['overlay_channels'] = True

        # TODO: copy relevant info from movie into dataset
        self.movie.read_header()

        # TODO: Perhaps it is best to always return an image with a channel dimension (when overlay_channels this can be one)
        image = self.get_projection_image(**projection_image_configuration)
        channel_index = self.movie.get_channel_indices_from_names(channels)[0]
        if len(image.shape) == 3:
            image = image[channel_index]

        print(f' Finding molecules in {self}')

        coordinates = find_peaks(image=image, **peak_finding_configuration)  # .astype(int)))

        # TODO: Check that it goes well if there are no coordinates left
        if margin:
            coordinates = coordinates_within_margin(coordinates, image, margin=margin)

        # TODO: Make gaussian width configurable or derive from psf size
        if fit_peaks:
            coordinates = coordinates_after_gaussian_fit(coordinates, image, gaussian_width=3)

        # TODO: Make radius, cutoff and fraction_of_peak_max configurable or derive from psf size
        # TODO: Make sure this handles empty arrays well
        if remove_peaks_with_close_neighbors:
            coordinates = coordinates_without_intensity_at_radius(coordinates, image, radius=3,
                                                                  cutoff='image_median', fraction_of_peak_max=0.25)

        coordinates = xr.DataArray(coordinates, dims=('molecule', 'dimension'),
                                   coords={'dimension': [b'x', b'y']}, name='coordinates')

        add_configuration_to_dataarray(coordinates, File.find_coordinates, locals(), units='pixel')

        for item_name in ['pixel_size', 'pixel_size_unit', 'stage_coordinates']:
            if hasattr(self.movie, item_name):
                if item_name == 'stage_coordinates':
                    item = getattr(self.movie, item_name)[0]
                else:
                    item = getattr(self.movie, item_name)
                coordinates.attrs[item_name] = item

        self.set_coordinates_of_channel(coordinates, channel=channel_index)

    def determine_psf_size(self, method='gaussian_fit', projection_type='average', frame_range=(0,20), channel_index=0, illumination_index=0,
                           peak_finding_configuration={'minimum_intensity_difference': 150}, maximum_radius=5):
        """
        Determine the Point Spread Function (PSF) size by fitting Gaussians to detected peaks.

        Parameters:
            method (str, optional): Method to determine PSF size ('gaussian_fit' or 'median'). Default is 'gaussian_fit'.
            projection_type (str, optional): Type of image projection to use. Default is 'average'.
            frame_range (tuple, optional): Range of frames to use for projection. Default is (0, 20).
            channel_index (int, optional): Index of the channel to use. Default is 0.
            illumination_index (int, optional): Index of the illumination to use. Default is 0.
            peak_finding_configuration (dict, optional): Arguments for peak finding.
            maximum_radius (int, optional): Maximum radius for PSF size. Default is 5.

        Returns:
            float: The determined PSF size.
        """

        image = self.get_projection_image(projection_type=projection_type, frame_range=frame_range,
                                          illumination=illumination_index)
        image = self.movie.get_channel(image, channel=channel_index)

        coordinates = find_peaks(image=image, **peak_finding_configuration)  # .astype(int)))
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
        """Return the molecule coordinates from the netCDF dataset."""
        if self.absolute_filepath.with_suffix('.nc').exists():
            with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
                if hasattr(dataset, 'coordinates'):
                    return dataset['coordinates'].load()
                else:
                    return None
        else:
            return None

    @coordinates.setter
    def coordinates(self, coordinates):
        """
        Set molecule coordinates and update the netCDF dataset.

        Parameters:
            coordinates (xarray.DataArray): The coordinates to set.
        """
        # Reset current .nc file
        self._init_dataset(len(coordinates.molecule))
        coordinates.drop('file', errors='ignore').to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def extract_traces(self, mask_size, neighbourhood_size=11, background_correction=None, alpha_correction=None,
                       gamma_correction=None):
        """
        Extract intensity traces for each molecule from the movie.

        Parameters:
            mask_size (int): Size of the mask for intensity extraction.
            neighbourhood_size (int, optional): Size of the neighbourhood for background extraction.
            background_correction (optional): Background correction method.
            alpha_correction (optional): Alpha correction factor.
            gamma_correction (optional): Gamma correction factor.
        """

        if self.number_of_molecules == 0:
            print('No molecules found, perform `find_coordinates` first')
            return

        if self.movie is None:
            raise FileNotFoundError('No movie file was found')

        print(f'Extracting traces in {self}')

        if mask_size == 'TIR-T' or mask_size == 'TIR-V':
            mask_size = 1.291
        elif mask_size == 'TIR-S 1.5x 2x2':
            mask_size = 0.8
        elif mask_size == 'TIR-S 1x 2x2':
            mask_size = 0.55
        elif mask_size == 'BN-TIRF':
            mask_size = 1.01

        intensity = extract_traces(self.movie, self.coordinates, mask_size=mask_size,
                                   neighbourhood_size=neighbourhood_size)

        add_configuration_to_dataarray(intensity, units='a.u.') # TODO: Link to units in movie metadata?
        intensity.attrs['configuration'] = json.dumps(dict(mask_size=mask_size, neighbourhood_size=neighbourhood_size))
        intensity.attrs['movie_configuration'] = json.dumps(self.movie.configuration)

        if self.movie.time is not None: # hasattr(self.movie, 'time')
            intensity = intensity.assign_coords(time=self.movie.time)

        intensity = intensity.assign_coords(illumination=self.movie.illumination_index_per_frame)

        intensity.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if 'intensity_raw' in self.data_vars:
            intensity_raw = self.intensity
            intensity_raw.name = 'intensity_raw'
            intensity_raw.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if background_correction is not None or alpha_correction is not None or gamma_correction is not None:
            self.apply_trace_corrections(background_correction, alpha_correction, gamma_correction)

        if self.movie.number_of_channels > 1:
            self.calculate_FRET()

    def apply_trace_corrections(self, background_correction=None, alpha_correction=None,
                       gamma_correction=None):
        """
        Apply corrections (background, alpha, gamma) to existing intensity traces.

        Parameters:
            background_correction (optional): Background correction method.
            alpha_correction (optional): Alpha correction factor.
            gamma_correction (optional): Gamma correction factor.
        """
        from papylio.trace_correction import trace_correction

        if 'intensity_raw' in self.data_vars:
            intensity_raw = self.intensity_raw
        else:
            intensity_raw = self.intensity
            intensity_raw.name = 'intensity_raw'
            intensity_raw.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

        intensity = trace_correction(intensity_raw, background_correction, alpha_correction, gamma_correction)
        intensity.name = 'intensity'
        initial_configuration = intensity.attrs['configuration']
        add_configuration_to_dataarray(intensity, File.apply_trace_corrections, locals(), units='a.u.') # TODO: Link to units in movie metadata?
        intensity.attrs['configuration'] = initial_configuration[:-1] + ', ' + intensity.attrs['configuration'][1:]

        intensity.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if 'FRET' in self.data_vars:
            self.calculate_FRET()

    def calculate_FRET(self):
        """Calculate and save FRET values for the intensity traces."""
        intensity = self.intensity
        FRET = calculate_FRET(intensity)
        FRET.attrs = intensity.attrs
        FRET.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def get_traces(self, selected=False):
        """
        Get all data variables that have a 'frame' dimension (traces).

        Parameters:
            selected (bool, optional): Whether to return traces only for selected molecules. Default is False.

        Returns:
            xarray.Dataset: The traces dataset.
        """
        dataset = self.dataset

        included_data_var_names = []
        for name in list(dataset.data_vars.keys()):
            if 'frame' in dataset[name].dims:
                included_data_var_names.append(name)

        traces = dataset[included_data_var_names]

        if selected:
            traces = traces.sel(molecule=dataset.selected)

        return traces

    @property
    @return_none_when_executed_by_pycharm
    def traces_names(self):
        """Return names of data variables that are traces (have 'frame' as the last dimension)."""
        # return list(self.classifications.data_vars.keys())
        return [
            name for name, da in self.data_vars.items()
            if da.dims and da.dims[-1] == "frame"
        ]

    def plot_hmm_rates(self, name=None):
        # TODO: Check this out and what we want to do with it.
        """
        Plot histograms of HMM transition rates for molecules with 2 states.

        Parameters:
            name (str, optional): Name to use for the plot title and filename.
        """
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
        """Save the dataset containing only selected molecules to a new netCDF file."""
        encoding = {'file': {'dtype': '|S'}, 'selected': {'dtype': bool}}
        self.dataset_selected.to_netcdf(self.absolute_filepath.parent / (self.name + '_selected.nc'), engine='netcdf4', mode='w', encoding=encoding)

    def import_pks_file(self, extension):
        """
        Import molecule coordinates and background from a .pks file.

        Parameters:
            extension (str): The file extension.
        """
        peaks = import_pks_file(self.absolute_filepath.with_suffix('.pks'))
        peaks = split_dimension(peaks, 'peak', ('molecule', 'channel'), (-1, 2)).reset_index('molecule', drop=True)
        # peaks = split_dimension(peaks, 'molecule', ('molecule_in_file', 'file'), (-1, 1), (-1, [file]), to='multiindex')

        if not self.absolute_filepath.with_suffix('.nc').is_file():
            self._init_dataset(len(peaks.molecule))

        coordinates = peaks.sel(parameter=['x', 'y']).rename(parameter='dimension')
        background = peaks.sel(parameter='background', drop=True)

        xr.Dataset({'coordinates': coordinates, 'background': background})\
            .to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def export_pks_file(self):
        """Export current molecule coordinates and background to a .pks file."""
        peaks = xr.merge([self.coordinates.to_dataset('dimension'), self.background.to_dataset()])\
            .stack(peaks=('molecule', 'channel')).to_array(dim='parameter').T
        export_pks_file(peaks, self.absolute_filepath.with_suffix('.pks'))
        self.extensions.add('.pks')

    def import_traces_file(self, extension):
        """
        Import intensity traces from a .traces file.

        Parameters:
            extension (str): The file extension.
        """
        traces = import_traces_file(self.absolute_filepath.with_suffix('.traces'))
        intensity = split_dimension(traces, 'trace', ('molecule', 'channel'), (-1, 2))\
            .reset_index(['molecule','frame'], drop=True)

        if not self.absolute_filepath.with_suffix('.nc').is_file():
            self._init_dataset(len(intensity.molecule))

        xr.Dataset({'intensity': intensity}).to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

    def export_traces_file(self):
        """Export current intensity traces to a .traces file."""
        traces = self.intensity.stack(trace=('molecule', 'channel')).T
        export_traces_file(traces, self.absolute_filepath.with_suffix('.traces'))
        self.extensions.add('.traces')

    def perform_mapping(self, method='icp', distance_threshold=3, transformation_type='polynomial',
                        projection_image_configuration=None, peak_finding_configuration=None,
                        margin=10, fit_peaks=True, remove_peaks_with_close_neighbors=None):
        #TODO: Update this for the new image style and for multiple mappings

        if projection_image_configuration is None:
            projection_image_configuration = dict(frame_range=(0,20), projection_type='average')

        if peak_finding_configuration is None:
            #TODO: Also make it possible to input just a dict, instead of a list of dicts
            peak_finding_configuration =[{'method': 'local-maximum-auto', 'filter_neighbourhood_size_min': 10,
                                                      'filter_neighbourhood_size_max': 5}] * self.movie.number_of_channels

        image = self.get_projection_image(**projection_image_configuration)

        print(transformation_type)

        coordinates_per_channel = []
        for i, channel_image in enumerate(image):
            coordinates = find_peaks(image=channel_image, **peak_finding_configuration[i])

            # TODO: Check that it goes well if there are no coordinates left
            if margin:
                coordinates = coordinates_within_margin(coordinates, image[i], margin=margin)

            # TODO: Make gaussian width configurable or derive from psf size
            if fit_peaks:
                coordinates = coordinates_after_gaussian_fit(coordinates, image[i], gaussian_width=5)

            # TODO: Make radius, cutoff and fraction_of_peak_max configurable or derive from psf size
            # TODO: Make sure this handles empty arrays well
            if remove_peaks_with_close_neighbors:
                coordinates = coordinates_without_intensity_at_radius(coordinates, image[i], radius=3,
                                                                      cutoff='image_median', fraction_of_peak_max=0.25)

            if coordinates.size == 0:  # should throw a error message to warm no acceptor molecules found
                raise ValueError(f'No molecules in channel {i}')

            coordinates_per_channel.append(coordinates)
            print(f'Found {coordinates.size} molecules in channel {i}')

        # TODO: put overlapping coordinates in file.coordinates for mapping file
        # Possibly do this with mapping.nearest_neighbour match
        # self.coordinates = np.hstack([donor_coordinates, acceptor_coordinates]).reshape((-1, 2))

        mappings = []
        for i in range(1, len(coordinates_per_channel)):
            mapping = mp.MatchPoint(source_name=self.movie.channels[0].name,
                                    source=coordinates_per_channel[0],
                                    destination_name=self.movie.channels[1].name,
                                    destination=coordinates_per_channel[i],
                                    method=method,
                                    transformation_type=transformation_type,
                                    initial_transformation=None)
            mapping.perform_mapping(distance_threshold=distance_threshold)
            mapping.source_channel_index = 0
            mapping.destination_channel_index = i
            mapping.file = str(self.relative_filepath)
            mappings.append(mapping)

        self.mappings = mappings

        self.export_mapping()

        # self.show_mapping_in_image(projection_image_configuration=projection_image_configuration)

        self.experiment.load_mappings()

    @property
    def mappings(self):
        return self._mappings

    @mappings.setter
    def mappings(self, mappings):
        self._mappings = mappings
        if self.movie is not None:
            self.movie.channel_mappings = mappings

    def show_mapping_in_image(self, axes=None, save=True, unit='pixel', projection_image_configuration=None, imshow_configuration=None):
        """
        Visualize the coordinate mapping on a projection image.

        Parameters:
            axis (matplotlib.axes.Axes, optional): The axis to plot on.
            save (bool, optional): Whether to save the plot as an image. Default is True.
        """
        if not hasattr(self, 'mappings') or self.mappings is None:
            raise RuntimeError('File does not contain a mapping.')

        if projection_image_configuration is None:
            projection_image_configuration = dict(frame_range=(0, 20), projection_type='average')

        #TODO: Update this after updating show_image
        figure, axes = self.show_image(axes=axes, unit=unit, projection_image_configuration=projection_image_configuration)
        for i, axis in enumerate(axes):
            if i==0:
                self.mappings[0].show(axis=axes[0], show_source=True, show_destination=False, show_transformed_coordinates=False)
                axis.set_title('')
            else:
                self.mappings[i-1].show(axis=axis, show_source=False, show_destination=True)
                axis.set_ylabel('')
                axis.set_title(axis.get_title().replace('channel_mapping_',''))

        if save:
            # axis.axis('off')
            # axis.set_title('')
            figure.set_size_inches(4*len(axes), 8)
            figure.savefig(self.relativePath / (self.name + '_mapping.png'), bbox_inches="tight", pad_inches=0, dpi=300)

        return figure, axis

    def copy_coordinates_to_selected_files(self):
        """Copy the current coordinates to all selected files in the experiment."""
        for file in self.experiment.selected_files:
            if file is not self:
                file.coordinates = self.coordinates

    # def use_mapping_for_all_files(self, perform_logging=True):
    #     """Apply the current coordinate mapping to all files in the experiment."""
    #     print(f"\n{self} used as mapping")
    #     self.is_mapping_file = True
    #     #mapping = self.movie.use_for_mapping()
    #     for file in self.experiment.files:
    #         if file is not self:
    #             perform_logging_file = file.perform_logging
    #             file.perform_logging = perform_logging
    #             file.mapping = self.mapping
    #             file.is_mapping_file = False
    #             file.perform_logging = perform_logging_file

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
        """
        Save data as a variable in the netCDF dataset.

        Parameters:
            data (numpy.ndarray or xarray.DataArray): The data to save.
            **kwargs: Additional arguments for xarray.DataArray.
        """
        da = xr.DataArray(data, **kwargs)
        da.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

    @property
    @return_none_when_executed_by_pycharm
    def intensity_total(self):
        """Return the sum of intensities across all channels."""
        return calculate_intensity_total(self.intensity)

    @property
    @return_none_when_executed_by_pycharm
    def selections(self):
        """Return all selection variables from the dataset."""
        with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return xr.Dataset({value.name: value for key, value in dataset.data_vars.items()
                               if key.startswith('selection_')}).load()

    def create_selection(self, variable, channel, aggregator, operator, threshold, name=None):
        # TODO: Make this more general like classification
        """
        Create a new selection based on a threshold applied to a variable.

        Parameters:
            variable (str): The name of the variable to use.
            channel (int or str): The channel to use.
            aggregator (str): The aggregator to apply over frames (e.g., 'mean', 'max').
            operator (str): The operator for thresholding ('<' or '>').
            threshold (float): The threshold value.
            name (str, optional): The name of the selection.
        """
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
        """Copy current selections and active selection state to all selected files in the experiment."""
        selection_configurations = self.selection_configurations()
        applied_selection = json.loads(self.selected.attrs['configuration'])

        for file in self.experiment.selected_files:
            if file is not self:
                for name, configuration in selection_configurations.items():
                    if configuration is None:
                        raise ValueError(f'Selection {name} is a custom selection that cannot be copied')
                    file.create_selection(**configuration)
                file.apply_selections(*applied_selection)

    @property
    @return_none_when_executed_by_pycharm
    def selection_names(self):
        """Return the names of all available selections."""
        return list(self.selections.data_vars.keys())

    @property
    @return_none_when_executed_by_pycharm
    def selection_names_active(self):
        """Return the names of the currently active selections."""
        return json.loads(self.selected.attrs['configuration'])

    def clear_selections(self):
        """Clear all selections and reset the active selection state."""
        dataset = self.dataset
        dataset = dataset.drop_vars([name for name in dataset.data_vars.keys() if name.startswith('selection_')])
        encoding = {
            var: {"dtype": 'bool'} for var in dataset.data_vars if dataset[var].dtype == bool
        }
        dataset.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='w', encoding=encoding)

    def selection_configurations(self, *selection_names):
        """
        Get the configurations for the specified selections.

        Parameters:
            *selection_names: The names of the selections to get configurations for.

        Returns:
            dict: A dictionary of selection names and their configurations.
        """
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
        """
        Apply the specified selections to the dataset.

        Parameters:
            *selection_names: The names of the selections to apply.
            add_to_current (bool, optional): Whether to add to the currently active selections. Default is False.
        """
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
        """Get the default classification for molecules in this file."""
        # Or add a standard classification datavar in the dataset?
        if not 'classification' in self.data_vars:
            self.apply_classifications()
        classification = self.__getattr__('classification')
        return classification

    @property
    @return_none_when_executed_by_pycharm
    def classifications(self):
        """Return all classification variables from the dataset."""
        with xr.open_dataset(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4') as dataset:
            return xr.Dataset({value.name: value for key, value in dataset.data_vars.items()
                               if key.startswith('classification_')}).load()

    @property
    @return_none_when_executed_by_pycharm
    def classification_names(self):
        """Return the names of all available classifications."""
        return list(self.classifications.data_vars.keys())

    def classification_configurations(self, classification_names='all'):
        """
        Get the configurations for the specified classifications.

        Parameters:
            classification_names (str or list, optional): The names of the classifications to get configurations for.
                Default is 'all'.

        Returns:
            dict: A dictionary of classification names and their configurations.
        """
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

    def create_classification(self, classification_type: Literal["threshold", "hmm"], variable, select=None, name=None, classification_configuration=None, apply=None):
        """
        Create a new classification.

        Parameters:
            classification_type (str): The type of classification ('threshold' or 'hmm').
            variable (str): The name of the variable to classify.
            select (optional): A selection to apply before classification.
            name (str, optional): The name of the classification.
            classification_configuration (dict, optional): Additional arguments for the classification method.
            apply (bool, optional): Whether to apply the classification immediately.
        """
        if classification_configuration is None:
            classification_configuration = {}

        if isinstance(variable, str):
            traces = getattr(self, variable)
        else:
            traces = variable
            variable = traces.name

        if select is not None:
            traces = traces.sel(**select)

        if classification_type == 'threshold':
            ds = classify_threshold(traces, **classification_configuration).to_dataset()
            # TODO: perhaps replace the following line with some function that actually spits out the classification configuration.
            add_configuration_to_dataarray(ds.classification, classify_threshold, classification_configuration)
        elif classification_type in ['hmm', 'hidden_markov_model']:
            # ds = hmm_traces(self.FRET, n_components=2, covariance_type="full", n_iter=100) # Old
            classification = self.classification
            selected = self.selected
            ds = classify_hmm(traces, classification, selected, **classification_configuration)
            if 'configuration' in selected.attrs:
                ds.classification.attrs['applied_selections'] = selected.attrs['configuration']
            if 'configuration' in classification.attrs:
                ds.classification.attrs['applied_classifications'] = classification.attrs['configuration']
            #TODO: perhaps replace the following line with some function that actually spits out the classification configuration.
            add_configuration_to_dataarray(ds.classification, classify_hmm, classification_configuration)
        # TODO: create classification to deactivate certain frames of the trace
        else:
            raise ValueError('Unknown classification type')

        classification_configuration = json.loads(ds.classification.attrs['configuration'])
        add_configuration_to_dataarray(ds.classification, File.create_classification, locals())

        if name is None:
            name = classification_type
        if not name.startswith('classification_'):
            name = 'classification_' + name
        ds = ds.rename({'classification': name})

        ds.to_netcdf(self.absolute_filepath.with_suffix('.nc'), engine='netcdf4', mode='a')

        if apply is not None:
            self.apply_classifications(add_to_current=True, **{name: apply})

    def classify_hmm(self, variable, seed=0, n_states=2, threshold_state_mean=None,
                     level='molecule'):  # , use_selection=True, use_classification=True):
        #TODO: Depricate this
        """
        Create an HMM-based classification.

        Parameters:
            variable (str): The name of the variable to classify.
            seed (int, optional): Random seed for HMM. Default is 0.
            n_states (int, optional): Number of states for HMM. Default is 2.
            threshold_state_mean (float, optional): Threshold for state mean.
            level (str, optional): The level at which to perform HMM ('molecule' or 'frame'). Default is 'molecule'.
        """
        self.create_classification(name='hmm', classification_type='hmm', variable=variable,
                                   classification_configuration=dict(seed=seed, n_states=n_states,
                                threshold_state_mean=threshold_state_mean, level=level))

    def apply_classifications(self, add_to_current=False, **classification_assignment):
        """
        Apply the specified classifications to the dataset.

        Parameters:
            add_to_current (bool, optional): Whether to add to the currently active classification. Default is False.
            **classification_assignment: Mapping of classification names to state indices.
        """
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

    def clear_classifications(self):
        """Clear all classifications and reset the active classification state."""
        dataset = self.dataset
        dataset = dataset.drop_vars([name for name in dataset.data_vars.keys() if name.startswith('classification_')])
        # for name, da in dataset.data_vars.items():
        #     da.encoding['dtype'] = da.dtype
        encoding = {
            var: {"dtype": 'bool'} for var in dataset.data_vars if dataset[var].dtype == bool
        }
        dataset.to_netcdf(self.absoluteFilePath.with_suffix('.nc'), engine='netcdf4', mode='w', encoding=encoding)


    @property
    @return_none_when_executed_by_pycharm
    def cycle_time(self):
        """Return the mean cycle time (time between frames)."""
        return self.time.diff('frame').mean().item()

    @property
    @return_none_when_executed_by_pycharm
    def sampling_interval(self):
        """Return the mean sampling interval (time between frames)."""
        return self.time.diff('frame').mean().item()

    @property
    @return_none_when_executed_by_pycharm
    def frame_rate(self):
        """Return the average frame rate."""
        return 1 / self.cycle_time

    def determine_dwells_from_classification(self, variable='FRET', selected=False, inactivate_start_and_end_states=True):
        """
        Extract dwell times from the current classification.

        Parameters:
            variable (str, optional): The trace variable to use for dwell time extraction. Default is 'FRET'.
            selected (bool, optional): Whether to use only selected molecules. Default is False.
            inactivate_start_and_end_states (bool, optional): Whether to ignore the first and last dwells. Default is True.
        """
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

        dwells.to_netcdf(self.absolute_filepath.with_name(self.name + '_dwells').with_suffix('.nc'), engine='netcdf4', mode='w')

    def classification_binary(self, positive_states_only=False, selected=False):
        """
        Get a binary representation of the classification.

        Parameters:
            positive_states_only (bool, optional): Whether to include only positive states. Default is False.
            selected (bool, optional): Whether to use only selected molecules. Default is False.

        Returns:
            xarray.DataArray: The binary classification.
        """
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
        """Return the number of unique states detected for each molecule based on the classification."""
        molecule_has_state = self.classification_binary(positive_states_only=True).any(dim='frame')
        number_of_states = molecule_has_state.sum(dim='state')
        return number_of_states

    @property
    @return_none_when_executed_by_pycharm
    def dwells(self):
        """Load and return the dwell times dataset."""
        return xr.load_dataset(self.absolute_filepath.with_name(self.name + '_dwells').with_suffix('.nc'), engine='netcdf4')

    def analyze_dwells(self, method='maximum_likelihood_estimation', number_of_exponentials=[1,2], state_names=None,
                       truncation=None, P_bounds=(-1, 1), k_bounds=(1e-9, np.inf), plot=False,
                       fit_dwell_times_configuration={}, plot_dwell_analysis_configuration={}, save_file_path=None):
        """
        Analyze dwell times.

        Parameters:
            method (str, optional): The analysis method. Default is 'maximum_likelihood_estimation'.
            number_of_exponentials (list, optional): Number of exponentials to fit. Default is [1, 2].
            state_names (list, optional): Names of the states.
            truncation (float, optional): Truncation time for fitting.
            P_bounds (tuple, optional): Bounds for the amplitudes.
            k_bounds (tuple, optional): Bounds for the rates.
            plot (bool, optional): Whether to plot the results. Default is False.
            fit_dwell_times_configuration (dict, optional): Additional arguments for dwell time fitting.
            plot_dwell_analysis_configuration (dict, optional): Additional arguments for plotting.
            save_file_path (str, optional): Path to save the analysis results.

        Returns:
            xarray.Dataset: The dwell analysis results.
        """
        dwells = self.dwells

        # At the moment single-state states are already set at -128 so they don't need to be separated.
        # For >2 states we will need to do this.
        # for n in np.arange(dwells.number_of_states.max().item())+1:
        #     dwells['state'][dict(dwell=(dwells['number_of_states'] == n) & dwells['state'] >= 0)] += n-1

        #TODO: Add sampling interval to File and refer to it here?
        dwell_analysis = analyze_dwells(dwells, method=method, number_of_exponentials=number_of_exponentials,
                                        state_names=state_names, P_bounds=P_bounds, k_bounds=k_bounds,
                                        sampling_interval=None, truncation=truncation, fit_dwell_times_config=fit_dwell_times_configuration)

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
                self.plot_dwell_analysis(**plot_dwell_analysis_configuration)
        else:
            dwell_analysis.to_netcdf(self.absolute_filepath.with_name(save_file_path).with_suffix('.nc'),
                                     engine='netcdf4', mode='w')
            if plot:
                plot_dwell_analysis(dwell_analysis, dwells, **plot_dwell_analysis_configuration)

        return dwell_analysis

    def plot_dwell_analysis(self, name=None, plot_type='pdf', plot_range=None, axes=None, bins='auto_discrete',
                            log=False, sharey=False, save_path=None):
        """
        Plot the results of dwell time analysis.

        Parameters:
            name (str, optional): Name for the plot title.
            plot_type (str, optional): Type of plot ('pdf', 'cdf', etc.). Default is 'pdf'.
            plot_range (tuple, optional): Range for the x-axis.
            axes (optional): Matplotlib axes to plot on.
            bins (optional): Binning strategy. Default is 'auto_discrete'.
            log (bool, optional): Whether to use a log scale for the x-axis. Default is False.
            sharey (bool, optional): Whether to share the y-axis across plots. Default is False.
            save_path (str or Path, optional): Directory to save the plot.

        Returns:
            tuple: (figure, axes)
        """
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
        # TODO: Consider whether to save the dwell_analysis in an excel file instead of an nc file.
        """Load and return the dwell analysis results from the netCDF file."""
        # return pd.read_excel(self.absoluteFilePath.with_name(self.name + '_dwell_analysis').with_suffix('.xlsx'))
        return xr.load_dataset(self.absolute_filepath.with_name(self.name + '_dwell_analysis').with_suffix('.nc'), engine='netcdf4')

    @dwell_analysis.setter
    def dwell_analysis(self, dwell_analysis):
        """
        Set dwell analysis results and save to the netCDF file.

        Parameters:
            dwell_analysis (xarray.Dataset): The dwell analysis results to save.
        """
        # dwell_analysis.to_excel(self.absoluteFilePath.with_name(self.name + '_dwell_analysis').with_suffix('.xlsx'))
        # dataset = xr.DataArray(dataset, dims=('exponential', ' variable'))
        dwell_analysis.to_netcdf(self.absolute_filepath.with_name(self.name + '_dwell_analysis').with_suffix('.nc'),
                                 engine='netcdf4', mode='w')

    def state_count(self, selected=True, states=None):
        """
        Count the number of molecules in each state.

        Parameters:
            selected (bool, optional): Whether to use only selected molecules. Default is True.
            states (list, optional): The states to count.

        Returns:
            xarray.DataArray: The counts for each state.
        """
        # if hasattr(self, 'number_of_states'):
        n, c = np.unique(self.get_variable('number_of_states', selected=selected), return_counts=True)
        if states is None:
            states = n

        state_count = xr.DataArray(0, dims=('name','number_of_states'), coords={'name': [self.name], 'number_of_states': states})
        state_count.loc[dict(number_of_states=n)] = c
        state_count.name = 'state_count'
        return state_count

    def state_fraction(self, **state_count_kwargs):
        """
        Calculate the fraction of molecules in each state.

        Parameters:
            **state_count_kwargs: Arguments passed to state_count.

        Returns:
            xarray.DataArray: The fraction of molecules in each state.
        """
        state_count = self.state_count(**state_count_kwargs)
        state_fraction = state_count / state_count.sum('number_of_states')
        state_fraction.name = 'state_fraction'
        return state_fraction

    def determine_trace_correction(self):
        """Open a GUI window to determine trace correction parameters."""
        from papylio.trace_correction import TraceCorrectionWindow
        # TODO: Should work on intensity raw if intensity raw is there else on intensity.
        TraceCorrectionWindow(self.intensity)

    def show_histogram(self, variable, selected=False, frame_range=None, average=False, axis=None, **hist_configuration):
        """
        Show a histogram of a variable.

        Parameters:
            variable (str): The name of the variable.
            selected (bool, optional): Whether to use only selected molecules. Default is False.
            frame_range (tuple, optional): Range of frames to include.
            average (bool or str, optional): Whether to average the variable.
            axis (optional): Matplotlib axis to plot on.
            **hist_configuration: Additional arguments for the histogram plot.

        Returns:
            tuple: (figure, axis)
        """
        # TODO: add save
        da = self.get_variable(variable, selected=selected, frame_range=frame_range, average=average)
        figure, axis = histogram(da, axis=axis, **hist_configuration)
        axis.set_title(str(self.relative_filepath))
        return figure, axis

    def histogram_2D_FRET_intensity_total(self, selected=False, frame_range=None, average=False,
                                       **marginal_hist2d_configuration):
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
        **marginal_hist2d_configuration : dict, optional
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

        marginal_hist2d_configuration_default = dict(range=((-0.05, 1.05), None))
        marginal_hist2d_configuration = {**marginal_hist2d_configuration_default, **marginal_hist2d_configuration}

        from papylio.plotting import marginal_hist2d
        figure, axes = marginal_hist2d(FRET, intensity_total, **marginal_hist2d_configuration)

        return axes

    def histogram_2D_intensity_per_channel(self, selected=False, frame_range=None, average=False,
                                           channel_x=0, channel_y=1, **marginal_hist2d_configuration):
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
        **marginal_hist2d_configuration : dict, optional
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

        marginal_hist2d_configuration_default = dict(range=(None, None))
        marginal_hist2d_configuration = {**marginal_hist2d_configuration_default, **marginal_hist2d_configuration}

        from papylio.plotting import marginal_hist2d
        figure, axes = marginal_hist2d(intensity_x, intensity_y, **marginal_hist2d_configuration)

        return axes

    def show_image(self, axes=None, unit='pixel', projection_image_configuration=None, imshow_configuration=None):
        #TODO: Finish docstring
        """
        Show a projection image of the movie.

        Returns:
            tuple: (figure, axes)
        """
        # TODO: Show two channels separately and connect axes
        # Split configuration based on inspect??

        if projection_image_configuration is None:
            projection_image_configuration = {}

        if imshow_configuration is None:
            imshow_configuration = {}

        image = self.get_projection_image(**projection_image_configuration)

        if axes is None:
            figure, axes = plt.subplots(1, image.shape[0], sharex=True, sharey=True, layout='constrained')
        else:
            figure = axes[0].figure
        figure.subplots_adjust(wspace=0, hspace=0)

        projection_image_configuration_defaults = get_default_parameters(Movie.make_projection_image)
        projection_image_configuration = (projection_image_configuration_defaults | projection_image_configuration)
        filename = Movie.image_info_to_filename(self.name, **projection_image_configuration)

        if projection_image_configuration['projection_type'] == 'average':
            figure.suptitle('Average image\n' + str(self.relativePath / filename))
        elif projection_image_configuration['projection_type'] == 'maximum':
            figure.suptitle('Maximum projection\n' + str(self.relativePath / filename))

        if unit == 'pixel':
            unit_string = ' (pixels)'
        elif unit == 'metric':
            imshow_configuration['extent'] = self.movie.boundaries_metric.T.flatten()[[0,1,3,2]]
            unit_string = f' ({self.movie.pixel_size_unit})'
        else:
            raise ValueError('Wrong unit value')
        for i, (im, axis) in enumerate(zip(image, axes.flatten())):
            axis.imshow(im, **imshow_configuration)
            axis.set_title(f'Channel {i}')
            axis.set_xlabel('x'+unit_string)
            if i > 0:
                axis.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)

        axes.flatten()[0].set_ylabel('y'+unit_string)
        return figure, axes

    def show_coordinates(self, axes=None, annotate=False, unit='pixel', scatter_configuration=None):
        # TODO: Consider making this a QWidget
        """
        Show detected molecule coordinates on a plot.

        Parameters:
            axes (optional): Matplotlib axes to plot on.
            annotate (bool, optional): Whether to enable interactive annotations.
            unit (str, optional): Unit for coordinates ('pixel' or 'metric'). Default is 'pixel'.
            **scatter_configuration: Additional arguments for scatter plot.
        """
        if scatter_configuration is None:
            scatter_configuration = {}

        if self.coordinates is None:
            return

        if unit == 'pixel':
            coordinates = self.coordinates
        elif unit == 'metric':
            coordinates = self.coordinates_metric
        else:
            raise ValueError('Unit can be either "pixel" or "metric"')

        if axes is None:
            figure, axes = plt.subplots(1, len(coordinates.channel), sharex=True, sharey=True, layout='constrained')
        else:
            figure = axes[0].figure
        figure.subplots_adjust(wspace=0, hspace=0)

        PathCollections = []
        for ((i, c), axis) in zip(coordinates.groupby('channel'), axes.flatten()):
            pc = axis.scatter(*c.T, facecolors='none', edgecolors='red', **scatter_configuration)
            PathCollections.append(pc)

            selected_coordinates = c.sel(molecule=self.selected.values)
            axis.scatter(*selected_coordinates.T, facecolors='none', edgecolors='green', **scatter_configuration)

        if annotate:
            annotation = axes[0].annotate("", xy=(0, 1.03), xycoords=axes[0].transAxes) # x in data units, y in axes fraction
            annotation.set_visible(False)

            #molecule_indices = np.repeat(np.arange(0, self.number_of_molecules), self.number_of_channels)
            molecule_indices = self.molecule_in_file.values #np.repeat(self.molecule_in_file.values, self.number_of_channels)
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
                # if event.inaxes == axis:
                for pc in PathCollections:
                    cont, ind = pc.contains(event)
                    if cont:
                        update_annotation(ind)
                        annotation.set_visible(True)
                        figure.canvas.draw_idle()
                        return
                else:
                    if vis:
                        annotation.set_visible(False)
                        figure.canvas.draw_idle()

            figure.canvas.mpl_connect("motion_notify_event", hover)

    def show_coordinates_in_image(self, axes=None, unit='pixel', annotate=False, projection_image_configuration=None,
                                  imshow_configuration=None, scatter_configuration=None):
        # TODO: Finish docstring
        """
        Show projection image with overlaid molecule coordinates.
        """

        figure, axes = self.show_image(axes=axes, unit=unit, projection_image_configuration=projection_image_configuration,
                                       imshow_configuration=imshow_configuration)
        self.show_coordinates(axes=axes, annotate=annotate, unit=unit, scatter_configuration=scatter_configuration)
        # plt.savefig(self.writepath.joinpath(self.name + '_ave_circles.png'), dpi=600)

    def show_traces(self, split_illuminations=True, **kwargs):
        """
        Open a GUI window to visualize intensity traces.

        Parameters:
            split_illuminations (bool, optional): Whether to split traces by illumination. Default is True.
            **kwargs: Additional arguments for TracePlotWindow.
        """
        dataset = self.dataset

        save_path = self.experiment.main_path.joinpath('Trace plots')
        if not save_path.is_dir():
            save_path.mkdir()

        from papylio.trace_plot import TracePlotWindow
        TracePlotWindow(dataset=dataset, split_illuminations=split_illuminations,
                        dataset_path=self.absolute_filepath.with_suffix('.nc'), save_path=save_path, **kwargs)


def calculate_intensity_total(intensity):
    """
    Calculate the total intensity by summing across channels.

    Parameters:
        intensity (xarray.DataArray): The input intensity DataArray.

    Returns:
        xarray.DataArray: The total intensity.
    """
    intensity_total = intensity.sum(dim='channel')
    intensity_total.name = 'intensity_total'
    intensity_total.attrs = intensity.attrs
    return intensity_total

def calculate_FRET(intensity):
    """
    Calculate FRET efficiency from donor and acceptor intensities.

    Parameters:
        intensity (xarray.DataArray): The input intensity DataArray with at least two channels.

    Returns:
        xarray.DataArray: The FRET efficiency.
    """
    # TODO: Make suitable for mutliple colours
    donor = intensity.sel(channel=0, drop=True)
    acceptor = intensity.sel(channel=1, drop=True)
    FRET = acceptor / (donor + acceptor)
    FRET.name = 'FRET'
    return FRET

def calculate_stoichiometry(intensity):
    """
    Calculate stoichiometry from intensity data.

    Parameters:
        intensity (xarray.DataArray): The input intensity DataArray.

    Returns:
        xarray.DataArray: The stoichiometry values.
    """
    intensity_total = calculate_intensity_total(intensity)
    intensity_total_i0 = intensity_total.sel(frame=intensity.illumination == 0)
    intensity_total_i1 = intensity_total.sel(frame=intensity.illumination == 1).values

    stoichiometry = intensity_total_i0 / (intensity_total_i0 + intensity_total_i1)
    stoichiometry.name = 'stoichiometry'
    return stoichiometry

def import_pks_file(pks_filepath):
    """
    Import peaks and background data from a .pks file.

    Parameters:
        pks_filepath (str or Path): Path to the .pks file.

    Returns:
        xarray.DataArray: The imported peak data.
    """
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
    """
    Export peak data to a .pks file.

    Parameters:
        peaks (xarray.DataArray): The peak data to export.
        pks_filepath (str or Path): The destination file path.
    """
    pks_filepath = Path(pks_filepath)
    with pks_filepath.open('w') as pks_file:
        for i, (x, y, background) in enumerate(peaks.values):
            # outfile.write(' {0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f} {4:4.4f} \n'.format(i, coordinate[0], coordinate[1], 0, 0, width4=4, width6=6))
            # pks_file.write('{0:4.0f} {1:4.4f} {2:4.4f} \n'.format(i + 1, coordinate[0], coordinate[1]))
            pks_file.write('{0:4.0f} {1:4.4f} {2:4.4f} {3:4.4f}\n'.format(i + 1, x, y, background))


def import_traces_file(traces_filepath):
    """
    Import intensity traces from a .traces file.

    Parameters:
        traces_filepath (str or Path): Path to the .traces file.

    Returns:
        xarray.DataArray: The imported traces.
    """
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
    """
    Export intensity traces to a .traces file.

    Parameters:
        traces (xarray.DataArray): The traces to export.
        traces_filepath (str or Path): The destination file path.
    """
    traces_filepath = Path(traces_filepath)
    with traces_filepath.open('w') as traces_file:
        # Number of frames
        np.array([len(traces.frame)], dtype=np.int32).tofile(traces_file)
        # Number of traces
        np.array([len(traces.trace)], dtype=np.int16).tofile(traces_file)
        traces.values.T.astype(np.int16).tofile(traces_file)

def split_dimension(data_array, old_dim, new_dims, new_dims_shape=None, new_dims_coords=None, to='dimensions'):
    """
    Split a single dimension into multiple dimensions in an xarray DataArray.

    Parameters:
        data_array (xarray.DataArray): The input DataArray.
        old_dim (str): The dimension to split.
        new_dims (tuple of str): The names of the new dimensions.
        new_dims_shape (tuple of int, optional): The shapes of the new dimensions.
        new_dims_coords (tuple, optional): The coordinates for the new dimensions.
        to (str, optional): Target format ('dimensions' or 'multiindex'). Default is 'dimensions'.

    Returns:
        xarray.DataArray: The DataArray with split dimensions.
    """
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