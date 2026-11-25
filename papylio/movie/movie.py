"""Movie IO and processing classes.

Provides the Movie base class and format-specific subclasses for reading frames,
creating projection images, and applying corrections.
"""

import itertools
import warnings
import tqdm
import re
import json
from pathlib import Path
import tifffile
import numpy as np
# from matchpoint import MatchPoint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr
from skimage.transform import AffineTransform

import matchpoint as mp

from papylio.helper_functions import get_default_parameters
# from papylio.movie.background_correction import rollingball
from papylio.timer import Timer
from papylio.log_functions import add_configuration_to_dataarray

class Movie:
    """Base class for microscopy movie/image stack handling.

    Provides abstract interface for loading, processing, and accessing
    image data from various microscopy file formats.
    """

    unit_mapping = mp.MatchPoint()
    default_microscope = None

    @classmethod
    def type_dict(cls):
        """Get dictionary mapping file extensions to Movie subclasses.

        Returns
        -------
        dict
            Dictionary with file extensions as keys and Movie subclasses as values
        """
        # It is important to import all movie files to recognize them by subclasses.
        # Perhaps we can make this more elegant in some way.
        from papylio.movie.sifx import SifxMovie
        from papylio.movie.pma import PmaMovie
        from papylio.movie.tif import TifMovie
        from papylio.movie.nd2 import ND2Movie
        from papylio.movie.nsk import NskMovie
        from papylio.movie.binary import BinaryMovie
        return {extension: subclass for subclass in cls.__subclasses__() for extension in subclass.extensions}

    @classmethod
    def default_movie_classes(cls, extension=None):
        # It is important to import all movie files to recognize them by subclasses.
        # Perhaps we can make this more elegant in some way.
        from papylio.movie.sifx import SifxMovie
        from papylio.movie.pma import PmaMovie
        from papylio.movie.tif import TifMovie
        from papylio.movie.nd2 import ND2Movie
        from papylio.movie.nsk import NskMovie
        from papylio.movie.binary import BinaryMovie

        default_movie_classes = cls.__subclasses__()
        if extension is None:
            return default_movie_classes
        else:
            for default_movie_class in default_movie_classes:
                if extension in default_movie_class.extensions:
                    return [default_movie_class]

    @classmethod
    def custom_movie_classes(cls, extension=None):
        # from papylio.configuration import load_user_microscope_classes
        # load_user_microscope_classes()

        default_movie_classes = cls.default_movie_classes(extension)
        custom_movie_classes = []
        for default_movie_class in default_movie_classes:
            if extension is not None and extension not in default_movie_class.extensions:
                continue
            custom_movie_classes += default_movie_class.__subclasses__()

        return custom_movie_classes

    @classmethod
    def subclass_from_filepath(cls, filepath):
        filepath = Path(filepath)
        extension = filepath.suffix.lower()

        default_movie_class = cls.default_movie_classes(extension)[0]
        custom_movie_classes = cls.custom_movie_classes(extension)
        if len(custom_movie_classes) == 0:
            return default_movie_class
        elif len(custom_movie_classes) == 1:
            return custom_movie_classes[0]
        else:
            if cls.default_microscope is None:
                custom_movie_classes = [custom_movie_class for custom_movie_class in custom_movie_classes if getattr(custom_movie_class(filepath), 'correct_microscope', lambda: False)()]
                number_of_correct_microscopes = len(custom_movie_classes)
                if number_of_correct_microscopes == 1:
                    custom_movie_class = custom_movie_classes[0]
                    cls.default_microscope = custom_movie_class.microscope
                elif number_of_correct_microscopes > 1:
                    raise ValueError('Multiple microscopes found for this filetype, please specify a microscope when initializing or improve the `correct_microscope` method in custom movie classes.')
                else:
                    raise ValueError('No correct microscope found for this filetype, please specify a microscope when initializing or improve the `correct_microscope` method in custom movie classes.')


            for custom_movie_class in custom_movie_classes:
                if custom_movie_class.microscope == cls.default_microscope:
                    return custom_movie_class
            else:
                raise ValueError('Default microscope not found in custom movie classes')
                # Or: return default_movie_class, but not sure what is better.

    illuminations = ['green', 'red']
    default_illumination = 0

    @classmethod
    def subclass_from_microscope(cls, microscope):
        for custom_movie_class in cls.custom_movie_classes():
            if custom_movie_class.microscope == microscope:
                return custom_movie_class
        else:
            raise ValueError('Unknown microscope')

    @classmethod
    def get_illumination_indices_from_names(cls, illuminations):
        """Get list of illumination indices by illumination names.

        Parameters
        ----------
        illumination_names : str, list, or None
            Illumination name(s)

        Returns
        -------
        list
            List of illumination indices
        """
        # illuminations = cls.get_illuminations_from_names(illumination_names)
        if illuminations in [None, 'all']:
            illuminations = cls.illuminations

        if not isinstance(illuminations, list):
            illuminations = [illuminations]

        illumination_indices = []
        for illumination in illuminations:
            if illumination in cls.illuminations:
                illumination_indices.append(cls.illuminations.index(illumination))
            elif isinstance(illumination, int):
                illumination_indices.append(illumination)
            else:
                None

        return illumination_indices

    @classmethod
    def image_info_from_filename(cls, filename):
        """Extract image metadata from filename using regex patterns.

        Parses standardized filename format to extract FOV index, projection type,
        frame range, illumination index, and correction flags.

        Parameters
        ----------
        filename : str
            Image filename to parse

        Returns
        -------
        dict
            Dictionary containing extracted image metadata

        Notes
        -----
        Filename format patterns recognized:
        - _fov{N}: Field of view index
        - _ave/_max: Projection type
        - _f{start}-{end}-{interval}: Frame range
        - _i{N}: Illumination index
        - _raw: Indicates raw (uncorrected) image
        """
        image_info = {}

        fov_index_result = re.search(r'(?<=_fov)\d*(?=[_.])', filename)
        if fov_index_result is not None:
            image_info['fov_index'] = int(fov_index_result.group())

        if '_ave' in filename:
            image_info['projection_type'] = 'average'
        elif '_max' in filename:
            image_info['projection_type'] = 'maximum'

        frame_start = re.search(r'(?<=_f)\d*(?=[-])', filename)
        if frame_start is not None:
            frame_end = re.search(rf'(?<=_f{frame_start.group()}-)\d*(?=[-_.])', filename)
            frame_interval = re.search(rf'(?<=_f{frame_start.group()}-{frame_end.group()}-)\d*(?=[_.])', filename)
            if frame_end is not None:
                frame_range = (int(frame_start.group()), int(frame_end.group()))
            else:
                raise ValueError('Invalid filename')
            if frame_interval is not None:
                frame_range += (int(frame_interval.group()),)
            image_info['frame_range'] = frame_range

        illumination_result = re.search(r'(?<=_i)\d*(?=[_.])', filename)
        if illumination_result is None:
            image_info['illumination_index'] = None  # list(self.illumination_indices.values)
        else:
            image_info['illumination_index'] = int(illumination_result.group())

        # channel_result = re.search('(?<=_c)\d*(?=[_.])', filename)
        # if channel_result is None:
        #     image_info['channel_indices'] = list(self.channel_indices.values)
        # else:
        #     image_info['channel_indices'] = int(channel_result.group())

        # fov_index = re.search('(?<=_fov)\d*(?=[_.])', filename)
        # if fov_index is not None:
        #     fov_index = int(fov_index)
        #     image_info['fov_index'] = fov_index

        illumination_result = re.search('_raw', filename)
        if illumination_result is None:
            image_info['apply_corrections'] = True
        else:
            image_info['apply_corrections'] = False

        overlay_result = re.search('_overlay', filename)
        if overlay_result is None:
            image_info['overlay_channels'] = False
        else:
            image_info['overlay_channels'] = True

        return image_info

    @classmethod
    def image_info_to_filename(cls, filename, fov_index=None, **projection_image_configuration):
        """Construct standardized filename from base name and image metadata.

        Parameters
        ----------
        filename : str
            Base filename
        fov_index : int, optional
            Field of view index (default: None)
        projection_image_configuration : keyword arguments

        Returns
        -------
        str
            Formatted filename with metadata embedded
        """

        projection_image_configuration = get_default_parameters(cls.make_projection_image) | projection_image_configuration

        # if 'fov_info' in self.__dict__.keys() and self.fov_info: # Or hasattr(self, 'fov_info')
        if fov_index is not None:
            # filename += f'_fov{self.fov_info["fov_chosen"]:03d}'
            filename += f'_fov{fov_index:03d}'

        projection_type = projection_image_configuration.get('projection_type', None)
        if projection_type is not None:
            filename += '_' + projection_type[:3]

        frame_range = projection_image_configuration.get('frame_range', None)
        if frame_range is not None:
            filename += str(range(*frame_range)).replace('range(', '_f').replace(', ', '-').replace(')', '')

        illumination = projection_image_configuration.get('illumination', None)
        if illumination is not None:  # and self.number_of_illuminations_in_movie > 1:
            if isinstance(illumination, str):
                illumination_index = cls.illuminations.index(illumination)
            elif isinstance(illumination, int):
                illumination_index = illumination
            else:
                raise ValueError('Invalid illumination type')
            filename += f'_i{illumination_index}'

        # if channel is not None:  # and self.number_of_illuminations_in_movie > 1:
        #     channel_index = cls.get_channel_from_name(channel).index
        #     filename += f'_i{channel_index}'

        apply_corrections = projection_image_configuration.get('apply_corrections', False)
        if apply_corrections is False:
            filename += '_raw'

        overlay_channels = projection_image_configuration.get('overlay_channels', False)
        if overlay_channels:
            filename += '_overlay'

        return filename

    # @property
    # def njit(self):
    #     from numba import njit
    #     return njit

    def __new__(cls, filepath, rotation=0, microscope=None):
        if cls is Movie:
            extension = Path(filepath).suffix.lower()

            if microscope is not None:
                subclass = cls.subclass_from_microscope(microscope)
            else:
                subclass = cls.subclass_from_filepath(filepath)

            try:
                return object.__new__(subclass)
            except KeyError:
                raise NotImplementedError('Filetype not supported')
        else:
            return object.__new__(cls)

    def __getnewargs__(self):
        return (self.filepath, self.rotation)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('file', None)
        return d

    def __setstate__(self, dict):
        self.__dict__.update(dict)

    def __init__(self, filepath, rotation=0, microscope=None):  # , **kwargs):
        """Initialize Movie object.

        Parameters
        ----------
        filepath : str or Path
            Path to the movie file
        rotation : int, optional
            Number of 90-degree rotations to apply to images (default: 0)
        """
        self.filepath = Path(filepath)
        self._with_counter = 0
        self.fov_index = None
        # self.filepaths = [Path(filepath) for filepath in filepaths] # For implementing multiple files, e.g. two channels over two files
        self.is_mapping_movie = False

        # self.rotation = rotation
        # self.correct_images = False

        self.chunk_size = 100
        self.use_dask = False

        self._data_type = np.dtype(np.uint16)
        self.intensity_range = (np.iinfo(self.data_type).min, np.iinfo(self.data_type).max)

        if not self.filepath.suffix == '.sifx':
            self.writepath = self.filepath.parent
            self.name = self.filepath.with_suffix('').name

        self._time = None

        self.channels = ['green', 'red']
        self.channel_arrangement = [[[0, 1]]]  # [[[0,1]]] # First level: frames, second level: y within frame, third level: x within frame

        self.channel_mapping = [self.unit_mapping,]*(self.number_of_channels-1)

        self.illumination_arrangement = [self.default_illumination]  # First level: frames, second level: illumination
        self._illumination_index_per_frame = None

        self._common_corrections = xr.Dataset()

        self.metadata_is_read = False

    def __enter__(self):
        """Context manager entry point. Opens file for reading."""
        if self._with_counter == 0:
            self.open()
        self._with_counter += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point. Closes file when exiting context."""
        self._with_counter -= 1
        if self._with_counter == 0:
            self.close()

    def __repr__(self):
        """Return string representation of Movie object."""
        return (f'{self.__class__.__name__}({str(self.filepath)})')

    def __getattr__(self, item):
        """Lazy-load header when accessing attributes before header is read."""

        if 'metadata_is_read' in self.__dict__.keys() and not self.metadata_is_read:
            # print(item+'2')
            self.read_metadata()
            return getattr(self, item)
        else:
            raise AttributeError(f'Attribute {item} not found')
        # return super().__getattribute__(item)

    @property
    def pixels_per_frame(self):
        """int : Total number of pixels in one frame (width × height)"""
        return self.width * self.height

    @property
    def bitdepth(self):
        """int : Bit depth of the image data (e.g., 8, 16, 32)"""
        return self.data_type.itemsize * 8  # 8 bits in a byte

    @property
    def bytes_per_frame(self):
        """int : Number of bytes per frame"""
        return self.data_type.itemsize * self.pixels_per_frame

    @property
    def time(self):
        """xr.DataArray : Time coordinate for each frame"""
        return self._time

    @time.setter
    def time(self, value):
        """Set time coordinate for frames."""
        self._time = value

    @property
    def number_of_illuminations(self):
        """int : Number of illumination channels in the movie"""
        return len(self.illuminations)

    @property
    def number_of_illuminations_in_movie(self):
        """ int : number of channels in the movie

        Setting the number of channels will divide the image horizontally in equally spaced channels.
        """
        return len(self.illumination_indices_in_movie)

    @property
    def number_of_channels(self):
        """ int : number of channels in the movie

        Setting the number of channels will divide the image horizontally in equally spaced channels.
        """
        return len(self.channels)

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type
        self.intensity_range = (np.iinfo(self.data_type).min, np.iinfo(self.data_type).max)

    @property
    def frame_indices(self):
        return xr.DataArray(np.arange(self.number_of_frames), dims='frame')

    @property
    def channel_indices(self):
        return xr.DataArray(np.array(self.channel_arrangement).flatten(), dims='channel')

    @property
    def number_of_channels_per_frame(self):
        return np.product(self.channel_arrangement.shape[1:])

    @property
    def illumination_arrangement(self):
        return self._illumination_arrangement

    @illumination_arrangement.setter
    def illumination_arrangement(self, illumination_arrangement):
        self._illumination_arrangement = np.array(illumination_arrangement)
        self._illumination_index_per_frame = None

    @property
    def illumination_indices(self):
        return xr.DataArray(list(range(len(self.illuminations))), dims='illumination')

    @property
    def illumination_index_per_frame(self):
        if self._illumination_arrangement is not None and self._illumination_index_per_frame is None:
            # frame_indices = self.frame_indices
            # illumination_indices = self.illumination_indices
            # self._illumination_index_per_frame = xr.DataArray(
            #     np.resize(self.illumination_arrangement, (len(frame_indices), len(illumination_indices))),
            #     dims=('frame', 'illumination'),
            #     coords={'frame': frame_indices, 'illumination': illumination_indices})
            self._illumination_index_per_frame = xr.DataArray(
                np.resize(self.illumination_arrangement, (self.number_of_frames)),
                dims=('frame'),
                coords={'frame': self.frame_indices}, name='illumination')
            # TODO: Add name to other indices or remove this name
        return self._illumination_index_per_frame

    @illumination_index_per_frame.setter
    def illumination_index_per_frame(self, illumination_index_per_frame):
        self._illumination_index_per_frame = illumination_index_per_frame
        self._illumination_arrangement = None

    @property
    def illumination_indices_in_movie(self):
        return np.unique(self.illumination_index_per_frame)

    @property
    def pixel_to_stage_coordinates_transformation(self):
        #TODO: Check whether the flipping implementation is correct for different microscopes, i.e. whether the stage coordinates are flipped with respect to the pixel coordinates and whether this is correctly implemented by flipping the stage coordinates in the translation part of the transformation.
        pixels_to_um = AffineTransform(scale=self.pixel_size)
        pixels_um_to_stage_coordinates_um = AffineTransform(translation=np.flip(self.stage_coordinates))
        pixels_to_stage_coordinates_um = pixels_to_um + pixels_um_to_stage_coordinates_um
        return pixels_to_stage_coordinates_um

    @property
    def width_metric(self):
        return self.width * self.pixel_size[0]

    @property
    def height_metric(self):
        return self.height * self.pixel_size[1]

    @property
    def boundaries(self):
        horizontal_boundaries = np.array([0, self.width])
        vertical_boundaries = np.array([0, self.height])
        return np.vstack([horizontal_boundaries, vertical_boundaries]).T

    @property
    def boundaries_metric(self):
        # Formatted as two coordinates, with the lowest and highest x and y values respectively
        horizontal_boundaries = np.array([0, self.width_metric])
        vertical_boundaries = np.array([0, self.height_metric])
        return np.vstack([horizontal_boundaries, vertical_boundaries]).T

    @property
    def boundaries_stage(self):
        return self.pixel_to_stage_coordinates_transformation(self.channels[0].boundaries)

    @property
    def channel_width(self):
        """int : Width of this channel in pixels (read-only)"""
        return self.width // len(self.channel_arrangement[0][0])

    @property
    def channel_height(self):
        """int : Height of this channel in pixels (read-only)"""
        return self.height // len(self.channel_arrangement[0])

    def read_metadata(self):
        """Read and parse file header.

        Calls the subclass-specific _read_metadata() method and applies
        image rotations if needed.
        """
        self._read_metadata()
        if not (self.rotation % 2 == 0):
            width = self.width
            height = self.height
            self.width = height
            self.height = width

        self.metadata_is_read = True

    def read_frame(self, frame_index, **kwargs):
        """Read a single frame from the movie.

        Parameters
        ----------
        frame_index : int
            Index of frame to read
        **kwargs
            Additional keyword arguments passed to read_frames()

        Returns
        -------
        np.ndarray or xr.DataArray
            Single frame image data
        """
        return self.read_frames([frame_index], **kwargs).squeeze(axis=0)

    def read_frames(self, frame_indices=None, apply_corrections=True, xarray=True, flatten_channels=False):
        """Read multiple frames from the movie.

        Parameters
        ----------
        frame_indices : list or np.ndarray, optional
            Indices of frames to read. If None, reads all frames (default: None)
        apply_corrections : bool, optional
            If True, apply flatfield/darkfield corrections (default: True)
        xarray : bool, optional
            If True, return xarray DataArray; else return numpy array (default: True)
        flatten_channels : bool, optional
            If True, combine channels back into spatial dimensions (default: False)

        Returns
        -------
        np.ndarray or xr.DataArray
            Array of image frames with shape (frame, channel, y, x)
        """
        if frame_indices is None:
            frame_indices = self.frame_indices.values

        frames = self._read_frames(frame_indices)
        frames = np.rot90(frames, self.rotation, axes=(1, 2))

        if len(self.channel_arrangement) > 1:
            raise NotImplementedError('Channel arrangement where frames indicated different channels not implemented')
            # Perhaps remove the outermost layer from channel_configuration
            # Or add this to separate and flatten channels

        frames = self.separate_channels(frames, self.channel_arrangement)

        if apply_corrections:  # and self.correct_images
            frames = self.apply_corrections(frames, frame_indices)

        if xarray:
            frames = self.frames_to_xarray_dataarray(frames, frame_indices)

        if flatten_channels:
            frames = self.flatten_channels(frames, self.channel_arrangement)

        return frames

    @property
    def channel_rows(self):
        """int : Number of channel rows in channel arrangement"""
        return len(self.channel_arrangement[0])

    @property
    def channel_columns(self):
        """int : Number of channel columns in channel arrangement"""
        return len(self.channel_arrangement[0][0])

    @staticmethod
    def separate_channels(frames, channel_arrangement):
        """Separate channels from spatial dimensions.

        Splits frames from single image dimension into separate channel dimension
        based on channel_arrangement pattern.

        Parameters
        ----------
        frames : np.ndarray or xr.DataArray
            Input frames with channels arranged spatially
        channel_rows : int
            Number of channel rows in channel arrangement
        channel_columns : int
            Number of channel columns in channel arrangement

        Returns
        -------
        np.ndarray or xr.DataArray
            Frames with channel as separate dimension
        """

        channel_arrangement = np.array(channel_arrangement)
        channel_frames, channel_rows, channel_columns = channel_arrangement.shape

        frames = xr.apply_ufunc(
            expand_axes, frames, input_core_dims=[['y', 'x']], output_core_dims=[['channel', 'y', 'x']],
            exclude_dims=set(['y', 'x']),
            kwargs={"expand_into": (channel_rows, channel_columns), "from_axes": (-2, -1),
                    "to_axes": (frames.ndim,) * 2, "new_axes_positions": [-3]}
        )

        order = channel_arrangement.flatten()
        inverse_order = np.argsort(order)
        frames = frames[..., inverse_order, :, :]

        return frames

    @staticmethod
    def flatten_channels(frames, channel_arrangement):
        """Combine channel dimension back into spatial dimensions.

        Parameters
        ----------
        frames : np.ndarray or xr.DataArray
            Frames with channel as separate dimension
        channel_rows : int
            Number of channel rows in channel arrangement
        channel_columns : int
            Number of channel columns in channel arrangement

        Returns
        -------
        np.ndarray or xr.DataArray
            Frames with channels arranged spatially
        """
        channel_arrangement = np.array(channel_arrangement)
        channel_frames, channel_rows, channel_columns = channel_arrangement.shape

        order = channel_arrangement.flatten()
        frames = frames[..., order, :, :]

        return xr.apply_ufunc(
            expand_axes, frames, input_core_dims=[['channel', 'y', 'x']], output_core_dims=[['y', 'x']],
            exclude_dims=set(['x', 'y']),
            kwargs={"expand_into": (channel_rows, channel_columns), "from_axes": (-2, -1),
                    "to_axes": (-3, -3),
                    "inverse": True, "squeeze": True}
        )

    def frames_to_xarray_dataarray(self, frames, frame_indices):
        """Convert frame array to xarray DataArray with coordinates.

        Parameters
        ----------
        frames : np.ndarray
            Frame array to convert
        frame_indices : array-like
            Frame indices for coordinates

        Returns
        -------
        xr.DataArray
            DataArray with frame, channel, y, x dimensions and coordinates
        """
        frames = xr.DataArray(frames,
                              dims=('frame', 'channel', 'y', 'x'),
                              coords={'frame': frame_indices,
                                      'illumination': self.illumination_index_per_frame[frame_indices],
                                      'channel': self.channel_indices})

        if self.time is not None:
            frames = frames.assign_coords(time=self.time[frames.frame])

        return frames

    def get_channel_indices_from_names(self, channel_names):
        if not (isinstance(channel_names, list) or isinstance(channel_names, tuple)):
            channel_names = [channel_names]
        channel_indices = []
        for channel_name in channel_names:
            if isinstance(channel_name, int):
                if channel_name < 0 or channel_name >= len(self.channels):
                    raise ValueError(f'Channel index "{channel_name}" out of range')
                channel_index = channel_name
            else:
                if channel_name not in self.channels:
                    raise ValueError(f'Unknown channel name "{channel_name}"')
                channel_index = self.channels.index(channel_name)
            channel_indices.append(channel_index)
        return channel_indices

    def saveas_tif(self):
        tif_filepath = self.writepath.joinpath(self.name + '.tif')
        tif_filepath.unlink(missing_ok=True)

        for i in range(self.number_of_frames):
            frame = self.read_frames([i], apply_corrections=False, xarray=False)
            tifffile.imwrite(tif_filepath, frame, append=True)

    def make_projection_image(self, projection_type='average', frame_range=(0,20), apply_corrections=True,
                              illumination=None, overlay_channels=False, flatten_channels=False):
        """ Construct a projection image
        Determine a projection image for a number_of_frames starting at start_frame.
        i.e. [start_frame, start_frame + number_of_frames)

        Parameters
        ----------
        projection_type : str
            'average' for average image
            'maximum' for maximum projection image
        start_frame : int
            Frame to start with
        number_of_frames : int
            Number of frames to average over
        write : bool
            If true, a tif file will be saved in writepath

        Returns
        -------
        np.ndarray
            2d image array with the projected image
        """

        frame_range = list(frame_range)
        # Make suitable for negative values
        if frame_range[0] > self.number_of_frames:
            raise ValueError(f'Invalid frame range {frame_range}')
        if frame_range[1] is None:
            frame_range = (frame_range[0], self.number_of_frames)
        if frame_range[1] > self.number_of_frames:
            frame_range[1] = self.number_of_frames
            warnings.warn(f'Frame range exceeds available frames, used frame range {frame_range} instead')

        frame_indices = self.frame_indices.values[slice(*frame_range)]

        illumination_indices = self.get_illumination_indices_from_names(illumination)
        illumination_index = np.intersect1d(illumination_indices, self.illumination_indices_in_movie)[0]

        # Select frame_indices with illumination
        frame_indices = frame_indices[self.illumination_index_per_frame.values[frame_indices] == illumination_index]

        # Calculate sum of frames and find mean
        image = self.separate_channels(np.zeros((self.height, self.width)).astype('float32'), self.channel_arrangement)

        frame_indices_subsets = np.array_split(frame_indices, len(frame_indices) // self.chunk_size + 1)

        if projection_type == 'average':
            number_of_frames = len(frame_indices)
            with self:
                for frame_indices_subset in tqdm.tqdm(frame_indices_subsets, desc='Average image'):
                    frames = self.read_frames(frame_indices_subset, apply_corrections=apply_corrections,
                                              xarray=False, flatten_channels=False)
                    image = image + frames.sum(axis=0)
                #TODO: Check whether this is a good way to average, i.e. do the values not get too big.
            image = (image / number_of_frames).astype('float32')
        elif projection_type == 'maximum':
            with self:
                for frame_indices_subset in tqdm.tqdm(frame_indices_subsets, desc='Maximum projection image'):
                    frames = self.read_frames(frame_indices_subset, xarray=False, flatten_channels=False)
                    image = np.maximum(image, frames.max(axis=0))

        if overlay_channels:
            for i in self.channel_indices[1:].values:
                image[i, :, :] = self.channel_mapping[i-1].transform_image(image[i, :, :], inverse=True)
            image = image.sum(axis=0, keepdims=True)

        if flatten_channels:
            image = self.flatten_channels(image, self.channel_rows, self.channel_columns)

        return image

    def save_projection_image(self, intensity_range=None, color_map='gray', path=None, filename=None, filetype='tif',
                              **projection_image_configuration):
        image = self.make_projection_image(**projection_image_configuration)

        if path is None:
            path = self.writepath

        if filename is None:
            filename = Movie.image_info_to_filename(self.name, **projection_image_configuration)

        filepath = path.joinpath(filename)

        if projection_image_configuration.get('overlay_channels', False):
            channel_names = 'overlay'
            channel_arrangement = np.array([[[0]]])
        else:
            channel_names = self.channels
            channel_arrangement = self.channel_arrangement

        save_image = self.flatten_channels(image, channel_arrangement)
        if filetype in ['tif', 'tiff']:
            if hasattr(self, 'pixel_size'):
                resolution = 1 / self.pixel_size
            else:
                resolution = None
            tifffile.imwrite(filepath.with_suffix('.tif'), save_image,
                             resolution=resolution,
                             imagej=True,
                             metadata={'unit': 'um',
                                       'axes': 'YX',
                                       'channel_arrangement': str(channel_arrangement),
                                       'labels': channel_names}
                             )
            # tifffile.imwrite(filepath.with_suffix('.tif'), image,
            #                  resolution=resolution,
            #                  imagej=True,
            #                  metadata={'unit': 'um',
            #                            'axes': 'CYX',
            #                            'labels': channel_names}
            #                  )
        elif filetype in ['png']:
            filepath = filepath.with_name(filepath.name + f'_v{intensity_range[0]}-{intensity_range[1]}')
            if intensity_range is None:
                intensity_range = self.intensity_range
            plt.imsave(filepath.with_suffix('.png'), save_image, vmin=intensity_range[0], vmax=intensity_range[1],
                       cmap=color_map)

        return image

    @staticmethod
    def load_projection_image(filepath, **projection_image_configuration):
        image_filename = Movie.image_info_to_filename(filepath.name, **projection_image_configuration)
        image_filepath = filepath.with_name(image_filename).with_suffix('.tif')

        if image_filepath.is_file():
            with tifffile.TiffFile(image_filepath) as tif:
                image = tif.asarray()
                metadata = tif.imagej_metadata
            channel_arrangement = np.array(json.loads(metadata['channel_arrangement']))
            return Movie.separate_channels(image, channel_arrangement)
        else:
            return None
            # raise FileNotFoundError(f'Projection image not found at {image_filepath}')

    def make_projection_images(self, projection_type='average', frame_range=(0, 20)):
        # Perhaps put this in make_projection_image as a special type of cmap
        for illumination_index in range(self.number_of_illuminations_in_movie):
            image = self.make_projection_image(projection_type, frame_range=(0,20), illumination=illumination_index,
                                               flatten_channels=False)
            channel_images = []
            for channel_index in range(self.number_of_channels):
                channel_image = image[channel_index]
                channel_image = (channel_image - self.intensity_range[0]) / (self.intensity_range[1] - self.intensity_range[0]) # TODO: make separate intensity range for each channel
                # channel_images.append(self.channels[channel_index].colour_map(channel_image, bytes=True))
                channel_images.append(channel_image)

            images_combined = np.hstack(channel_images)
            filename = Movie.image_info_to_filename(self.name, fov_index=self.fov_index, projection_type=projection_type,
                                                    frame_range=frame_range, illumination=illumination_index)
            filepath = self.writepath.joinpath(filename)
            plt.imsave(filepath.with_suffix('.png'), images_combined)

    def make_average_image(self, **kwargs):
        """ Construct an average image
        Determine average image for a number_of_frames starting at start_frame.
        i.e. [start_frame, start_frame + number_of_frames)

        Parameters
        ----------
        start_frame : int
            Frame to start with
        number_of_frames : int
            Number of frames to average over
        write : bool
            If true, the a tif file will be saved in the writepath

        Returns
        -------
        np.ndarray
            2d image array with the average image

        """
        return self.make_projection_image('average', **kwargs)

    def make_maximum_projection(self, **kwargs):
        """ Construct a maximum projection image
        Determine maximum projection image for a number_of_frames starting at start_frame.
        i.e. [start_frame, start_frame + number_of_frames)

        Parameters
        ----------
        start_frame : int
            Frame to start with
        number_of_frames : int
            Number of frames to average over
        write : bool
            If true, the a tif file will be saved in the writepath

        Returns
        -------
        np.ndarray
            2d image array with the maximum projection image
        """

        return self.make_projection_image('maximum', **kwargs)

    def show(self):
        return MoviePlotter(self)

    # Do we really need this?
    def determine_general_background_correction(self, method='median', frame_range=(0, 20), use_existing=False):
        from papylio.movie.background_correction import determine_single_value_background_correction
        #Todo: pass method kwargs
        if use_existing and 'general_background_correction' in self.corrections:
            return
        # self.temporal_background_correction = self.spatial_background_correction = None
        self.save_corrections(general_background_correction=None)

        frame_indices = self.frame_indices[slice(*frame_range)].values
        with self:
            frames = self.read_frames(frame_indices=frame_indices, apply_corrections=True, xarray=False)

        general_background_correction = xr.DataArray(0, dims=('illumination', 'channel'),
                                                      coords={'channel': self.channel_indices,
                                                              'illumination': self.illumination_indices},
                                                      name='general_background_correction')

        # corrections = self.corrections

        for illumination, channel in itertools.product(self.illumination_indices_in_movie,
                                                       np.array(self.channel_indices)):
            frame_indices_subset = (self.illumination_index_per_frame[frame_indices] == illumination).frame
            average_image = frames[frame_indices_subset, channel].mean(axis=0)

            correction = determine_single_value_background_correction(average_image, method)#, flatfield, darkfield)
            general_background_correction[dict(illumination=illumination, channel=channel)] = correction

        add_configuration_to_dataarray(general_background_correction, Movie.determine_general_background_correction,
                                       locals(), units='a.u.') # TODO: Link to units in movie metadata?
        self.save_corrections(general_background_correction=general_background_correction)

    def determine_temporal_background_correction(self, method='median', use_existing=False):
        from papylio.movie.background_correction import determine_temporal_background_correction
        #Todo: pass method kwargs
        if use_existing and 'temporal_background_correction' in self.corrections:
            return

        self.save_corrections(temporal_illumination_correction=None,
                              temporal_background_correction=None,
                              spatial_background_correction=None,
                              general_background_correction=None)

        frames = self.read_frames(frame_indices=None, apply_corrections=True, xarray=False)

        temporal_background_correction = xr.DataArray(0, dims=('frame', 'channel'),
                                                      coords={'frame': self.frame_indices,
                                                              'channel': self.channel_indices},
                                                      name='temporal_background_correction')

        for illumination, channel in itertools.product(self.illumination_indices_in_movie, np.array(self.channel_indices)):

            frame_indices_subset = (self.illumination_index_per_frame==illumination).frame
            frames_subset = frames[frame_indices_subset, channel]

            correction = determine_temporal_background_correction(frames_subset, method)#, flatfield, darkfield)
            temporal_background_correction[dict(frame=frame_indices_subset, channel=channel)] = correction

        add_configuration_to_dataarray(temporal_background_correction, Movie.determine_temporal_background_correction,
                                       locals(), units='a.u.') # TODO: Link to units in movie metadata?

        self.save_corrections(temporal_background_correction=temporal_background_correction)

    def determine_spatial_background_correction(self, method='median_filter', frame_range=(0, 20), use_existing=False,
                                                **kwargs):
        from papylio.movie.background_correction import determine_spatial_background_correction
        if use_existing and 'spatial_background_correction' in self.corrections:
            return

        self.save_corrections(spatial_background_correction=None, general_background_correction=None)

        frame_indices = self.frame_indices[slice(*frame_range)].values
        with self:
            frames = self.read_frames(frame_indices=frame_indices, apply_corrections=True, xarray=False)

        spatial_background_correction = xr.DataArray(np.zeros((self.number_of_illuminations,) + frames.shape[1:]),
                                                     dims=('illumination', 'channel', 'y', 'x'),
                                                     coords={'illumination': self.illumination_indices,
                                                             'channel': self.channel_indices, },
                                                     name='spatial_background_correction')

        for illumination, channel in itertools.product(self.illumination_indices_in_movie,
                                                       np.array(self.channel_indices)):
            frame_selection = (self.illumination_index_per_frame[frame_indices] == illumination).values
            average_image = frames[frame_selection, channel].mean(axis=0)

            correction = determine_spatial_background_correction(average_image, method, **kwargs)
            spatial_background_correction[dict(illumination=illumination, channel=channel)] = correction

        add_configuration_to_dataarray(spatial_background_correction, Movie.determine_spatial_background_correction,
                                       locals(), units='a.u.') # TODO: Link to units in movie metadata?

        self.save_corrections(spatial_background_correction=spatial_background_correction)#,

    @property
    def corrections(self):
        if hasattr(self, 'fov_index') and self.fov_index is not None:
            corrections_filepath = self.filepath.with_name(self.name + f'_fov{self.fov_index:03d}' + '_corrections.nc')
        else:
            corrections_filepath = self.filepath.with_name(self.name + '_corrections.nc')

        if corrections_filepath.exists():
            corrections = xr.load_dataset(corrections_filepath, engine='h5netcdf')
        else:
            corrections = xr.Dataset()
        corrections = corrections.merge(self._common_corrections, compat='override')
        return corrections

    @property
    def configuration(self):
        configuration = dict(rotation=self.rotation)
        for name, correction in self.corrections.data_vars.items():
            if 'configuration' in correction.attrs:
                configuration[name] = correction.attrs['configuration']
            else:
                configuration[name] = None
        return configuration

    @property
    def corrections_filepath(self):
        if hasattr(self, 'fov_index') and self.fov_index is not None:
            corrections_filepath = self.filepath.with_name(self.name + f'_fov{self.fov_index:03d}' + '_corrections.nc')
        else:
            corrections_filepath = self.filepath.with_name(self.name + '_corrections.nc')
        return corrections_filepath

    def reset_corrections(self):
        self.corrections_filepath.unlink(missing_ok=True)

    def save_corrections(self, **kwargs):
        corrections_filepath = self.corrections_filepath
        if corrections_filepath.exists():
            corrections = xr.load_dataset(corrections_filepath, engine='h5netcdf')
        else:
            corrections = xr.Dataset()
        for name, correction in kwargs.items():
            # correction = getattr(self, name)
            if correction is None:
                corrections = corrections.drop_vars(name, errors='ignore')
            else:
                corrections[name] = correction
        corrections.to_netcdf(corrections_filepath, mode='w', engine='h5netcdf')

# # @njit
    def apply_corrections(self, frames, frame_indices):
        illumination_indices = self.illumination_index_per_frame[frame_indices]
        frames = frames.astype(np.float32)
        corrections = self.corrections
        for illumination_index in np.unique(illumination_indices):
            frame_indices_with_illumination = np.array(illumination_indices == illumination_index)

            if 'darkfield_correction' in corrections:
                frames[frame_indices_with_illumination] -= corrections.darkfield_correction.values[None, illumination_index]

            if 'flatfield_correction' in corrections:
                frames[frame_indices_with_illumination] /= corrections.flatfield_correction.values[None, illumination_index]

            if 'temporal_illumination_correction' in corrections:
                frames[frame_indices_with_illumination] /= \
                    corrections.temporal_illumination_correction.values[frame_indices][frame_indices_with_illumination, None, None, None]

            if 'temporal_background_correction' in corrections:
                frames[frame_indices_with_illumination] -= \
                    corrections.temporal_background_correction.values[frame_indices][frame_indices_with_illumination, :, None, None]

            if 'spatial_background_correction' in corrections:
                frames[frame_indices_with_illumination] -= corrections.spatial_background_correction.values[None,
                                                           illumination_index, :, :, :]

            if 'general_background_correction' in corrections:
                frames[frame_indices_with_illumination] -= corrections.general_background_correction.values[None, illumination_index, :, None, None]

        return frames

    def show_correction(self, correction_name, save=True, **kwargs):
        correction = self.corrections[correction_name]
        number_of_illuminations = len(correction.illumination)
        figure, axes = plt.subplots(1, number_of_illuminations+1, gridspec_kw=dict(width_ratios=(4,)*number_of_illuminations + (0.15,)),
                                    figsize=(4*number_of_illuminations+0.15, 4), layout='tight')
        for i, illumination_index in enumerate(correction.illumination):
            axes[i].axis('off')
            image = axes[i].imshow(self.flatten_channels(correction.sel(illumination=illumination_index)), **kwargs)
            axes[i].set_title(f'Illumination {illumination_index.item()}', fontsize=8)

        cax = axes[-1]
        figure.colorbar(image, aspect=50, cax=cax)
        cax.set_ylabel('Intensity (a.u.)')
        if correction_name == 'flatfield_correction':
            cax.set_ylabel('Correction factor')

        # cax.axes.ticklabel_format(scilimits=(0, 0))
        for spine in cax.spines.values():
            spine.set_visible(False)

        figure.suptitle(f'{self.name} - {correction_name}', fontsize=8)

        if save:
            figure.savefig(self.filepath.with_name(f'{self.name} - {correction_name}.png'), bbox_inches='tight')

class MoviePlotter:
    # Adapted from Matplotlib Image Slices Viewer
    """Interactive image viewer for movie frames with scroll wheel navigation.

    Displays frames from a Movie object with ability to navigate using scroll wheel.
    """
    def __init__(self, movie):
        """Initialize MoviePlotter.

        Parameters
        ----------
        movie : Movie
            Movie object to visualize
        """
        fig, ax = plt.subplots(1, 1)
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.show()

        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.movie = movie
        self.slices, rows, cols = (movie.number_of_frames, movie.height, movie.width)
        self.ind = self.slices // 2

        self.im = ax.imshow(self.movie.read_frame(self.ind, flatten_channels=True, xarray=False))
        self.update()

    def on_scroll(self, event):
        """Handle scroll wheel events to navigate frames.

        Parameters
        ----------
        event : matplotlib.backend_bases.ScrollEvent
            Scroll event from matplotlib
        """
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """Update displayed frame."""
        self.im.set_data(self.movie.read_frame(self.ind, flatten_channels=True, xarray=False))
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def make_colour_map(colour, N=256):
    """Create a matplotlib colormap for the specified color.

    Creates a linear colormap transitioning from black to the specified color.

    Parameters
    ----------
    colour : str
        Color name: 'red', 'green', 'blue', or 'grey'
    N : int, optional
        Number of color levels in the map (default: 256)

    Returns
    -------
    matplotlib.colors.ListedColormap
        Colormap object for use with matplotlib
    """
    values = np.zeros((N, 3))
    if colour == 'grey':
        values[:, 0] = values[:, 1] = values[:, 2] = np.linspace(0, 1, N)
    elif colour == 'red':
        values[:, 0] = np.linspace(0, 1, N)
    elif colour == 'green':
        values[:, 1] = np.linspace(0, 1, N)
    elif colour == 'blue':
        values[:, 2] = np.linspace(0, 1, N)
    else:
        values[:, 0] = values[:, 1] = values[:, 2] = np.linspace(0, 1, N)

    return ListedColormap(values)


def expand_axes(frames, expand_into, from_axes=-1, to_axes=None, new_axes_positions=[], inverse=False, squeeze=False):
    """Expand/split array axes along specified dimensions.

    Splits frames along specified axes into multiple sub-arrays, useful for
    separating multi-channel or multi-frame data stored in single dimensions.

    Parameters
    ----------
    frames : np.ndarray
        Input array to expand
    expand_into : int or tuple of int
        Target sizes for expansion
    from_axes : int or tuple of int, optional
        Source axis/axes to expand from (default: -1)
    to_axes : int, tuple of int, or None, optional
        Target axis/axes to expand to (default: None)
    new_axes_positions : list, optional
        Positions for new axes (default: [])
    inverse : bool, optional
        If True, perform inverse operation (collapse instead of expand) (default: False)
    squeeze : bool, optional
        If True, remove singleton dimensions (default: False)

    Returns
    -------
    np.ndarray
        Expanded array with separated dimensions
    """
    if isinstance(expand_into, int):
        expand_into = (expand_into,)
    if isinstance(from_axes, int):
        from_axes = (from_axes,)
    if isinstance(to_axes, int) or to_axes is None:
        to_axes = (to_axes,) * len(from_axes)

    from_axes = list(from_axes)
    to_axes = list(to_axes)

    if inverse:
        expand_into = expand_into[::-1]
        from_axes, to_axes = to_axes[::-1], from_axes[::-1]

    ndim = frames.ndim

    # new_axes_created = 0
    for i, (from_axis, to_axis) in enumerate(zip(from_axes, to_axes)):
        # if from_axis is None:
        #     from_axes[i] = ndim-1
        if from_axis < 0:
            from_axes[i] = range(ndim)[from_axis]
        #
        # if to_axis is None:
        #     # if combine_new_axes and new_axes_created > 0:
        #     #     to_axes[i] = ndim
        #     # else:
        #     new_axes_positions.append(ndim)
        #     new_axes_created += 1
        if -ndim <= to_axis < 0:
            to_axes[i] = range(ndim)[to_axis]

        elif to_axis < -ndim:  # or to_axis > ndim-1:
            if to_axis not in new_axes_positions:
                new_axes_positions.append(to_axis)
            if to_axis < 0:
                to_axes[i] = ndim + new_axes_positions.index(to_axis)

            # to_axes[i] = None
            # new_axes_created += 1

    for i, (n, from_axis, to_axis) in enumerate(zip(expand_into, from_axes, to_axes)):
        # if frames.shape[-1] % n > 0:
        #     raise ValueError('Cannot split into equal parts')
        if to_axis > frames.ndim - 1:
            frames = np.moveaxis(frames, from_axis, -1)
            frames = frames.reshape(*frames.shape[:-1], n, frames.shape[-1] // n)
            frames = np.moveaxis(frames, -1, from_axis)
        elif inverse:
            frames = np.moveaxis(frames, [from_axis, to_axis], [-2, -1])
            frames = frames.reshape(*frames.shape[:-2], frames.shape[-2] // n, frames.shape[-1] * n)
            frames = np.moveaxis(frames, [-2, -1], [from_axis, to_axis])
        else:
            frames = np.moveaxis(frames, [from_axis, to_axis], [-1, -2])
            frames = frames.reshape(*frames.shape[:-2], frames.shape[-2] * n, frames.shape[-1] // n)
            frames = np.moveaxis(frames, [-1, -2], [from_axis, to_axis])

    if inverse and squeeze:
        for from_axis in np.sort(np.unique(from_axes))[::-1]:
            if frames.shape[from_axis] <= 1:
                frames = frames.squeeze(axis=from_axis)
        # frames = np.moveaxis(frames, from_axis, -1)

    if new_axes_positions:
        frames = np.moveaxis(frames, -np.arange(len(new_axes_positions))[::-1] - 1, new_axes_positions)

    # Test code
    # start = time.time()
    # a = np.stack(np.split(frames, 2, axis=2), axis=-1)
    # b = np.concatenate(np.split(a, 2, axis=1), axis=-1)
    # print(time.time() - start)
    #
    # start = time.time()
    # bb = split_image_channels(frames, (2, 2), axes=(1, 2), combine_new_axes=False)
    # print(time.time() - start)

    return frames

