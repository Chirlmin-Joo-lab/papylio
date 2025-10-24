import re
import sys
import itertools
import warnings
import tqdm
import re
from numba import njit
from pathlib import Path
import pandas as pd
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xarray as xr
import scipy.ndimage
from skimage.transform import AffineTransform

# from papylio.movie.background_correction import rollingball
from papylio.movie.background_correction import determine_temporal_background_correction, \
    determine_spatial_background_correction, determine_single_value_background_correction # remove_background, get_threshold
from papylio.timer import Timer
from papylio.log_functions import add_configuration_to_dataarray

class Illumination:
    def __init__(self, name, short_name='', other_names=[]):
        # self.movie = movie
        self.name = name
        self.short_name = short_name
        self.other_names = other_names

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

    @property
    def names(self):
        return [self.index, self.name, self.short_name] + self.other_names

    @property
    def index(self):
        try:
            return Movie.illuminations.index(self)
        except:
            pass


class Movie:
    @classmethod
    def type_dict(cls):
        # It is important to import all movie files to recognize them by subclasses.
        # Perhaps we can make this more elegant in some way.
        from papylio.movie.sifx import SifxMovie
        from papylio.movie.pma import PmaMovie
        from papylio.movie.tif import TifMovie
        from papylio.movie.nd2 import ND2Movie
        from papylio.movie.nsk import NskMovie
        from papylio.movie.binary import BinaryMovie
        return {extension: subclass for subclass in cls.__subclasses__() for extension in subclass.extensions}

    illuminations = [Illumination('green', 'g'), Illumination('red', 'r')]

    @classmethod
    def get_illumination_from_name(cls, illumination_name):
        """Get the channel index belonging to a specific channel (name)
        If

        Parameters
        ----------
        channel : str or int
            The name or number of a channel

        Returns
        -------
        i: int
            The index of the channel to which the channel name belongs

        """
        for illumination in cls.illuminations:
            if illumination_name in illumination.names or illumination_name == illumination:
                return illumination
        else:
            raise ValueError('Illumination name not found')

    @classmethod
    def get_illuminations_from_names(cls, illumination_names):
        """Get the channel index belonging to a specific channel (name)
        If

        Parameters
        ----------
        channel : str or int
            The name or number of a channel

        Returns
        -------
        i: int
            The index of the channel to which the channel name belongs

        """
        if illumination_names in [None, 'all']:
            return cls.illuminations

        if not isinstance(illumination_names, list):
            illumination_names = [illumination_names]

        return [cls.get_illumination_from_name(illumination_name) for illumination_name in illumination_names]

    @classmethod
    def get_illumination_indices_from_names(cls, illumination_names):
        illuminations = cls.get_illuminations_from_names(illumination_names)
        return [illumination.index for illumination in illuminations]

    @classmethod
    def image_info_from_filename(cls, filename):
        image_info = {}

        fov_index_result = re.search('(?<=_fov)\d*(?=[_.])', filename)
        if fov_index_result is not None:
            image_info['fov_index'] = int(fov_index_result.group())

        if '_ave' in filename:
            image_info['projection_type'] = 'average'
        elif '_max' in filename:
            image_info['projection_type'] = 'maximum'

        frame_start = re.search('(?<=_f)\d*(?=[-])', filename)
        if frame_start is not None:
            frame_end = re.search(f'(?<=_f{frame_start.group()}-)\d*(?=[-_.])', filename)
            frame_interval = re.search(f'(?<=_f{frame_start.group()}-{frame_end.group()}-)\d*(?=[_.])', filename)
            if frame_end is not None:
                frame_range = (int(frame_start.group()), int(frame_end.group()))
            else:
                raise ValueError('Invalid filename')
            if frame_interval is not None:
                frame_range += (int(frame_interval.group()),)
            image_info['frame_range'] = frame_range

        illumination_result = re.search('(?<=_i)\d*(?=[_.])', filename)
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

        return image_info

    @classmethod
    def image_info_to_filename(cls, filename, fov_index=None, projection_type=None, frame_range=None,
                               illumination=None, apply_corrections=None):
        # if 'fov_info' in self.__dict__.keys() and self.fov_info: # Or hasattr(self, 'fov_info')
        if fov_index is not None:
            # filename += f'_fov{self.fov_info["fov_chosen"]:03d}'
            filename += f'_fov{fov_index:03d}'

        if projection_type is not None:
            filename += '_' + projection_type[:3]

        if frame_range is not None:
            filename += str(range(*frame_range)).replace('range(', '_f').replace(', ', '-').replace(')', '')

        if illumination is not None:  # and self.number_of_illuminations_in_movie > 1:
            illumination_index = cls.get_illumination_from_name(illumination).index
            filename += f'_i{illumination_index}'

        # if channel is not None:  # and self.number_of_illuminations_in_movie > 1:
        #     channel_index = cls.get_channel_from_name(channel).index
        #     filename += f'_i{channel_index}'

        if apply_corrections is False:
            filename += '_raw'

        return filename

    def __new__(cls, filepath, rot90=0):
        if cls is Movie:
            extension = Path(filepath).suffix.lower()
            try:
                return object.__new__(cls.type_dict()[extension])
            except KeyError:
                raise NotImplementedError('Filetype not supported')
        else:
            return object.__new__(cls)

    def __getnewargs__(self):
        return (self.filepath, self.rot90)

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('file', None)
        return d

    def __setstate__(self, dict):
        self.__dict__.update(dict)

    def __init__(self, filepath, rot90=0):  # , **kwargs):
        self.filepath = Path(filepath)
        self._with_counter = 0
        self.fov_index = None
        # self.filepaths = [Path(filepath) for filepath in filepaths] # For implementing multiple files, e.g. two channels over two files
        self.is_mapping_movie = False

        self.rot90 = rot90
        # self.correct_images = False

        self.chunk_size = 100
        self.use_dask = False

        self._data_type = np.dtype(np.uint16)
        self.intensity_range = (np.iinfo(self.data_type).min, np.iinfo(self.data_type).max)

        if not self.filepath.suffix == '.sifx':
            self.writepath = self.filepath.parent
            self.name = self.filepath.with_suffix('').name

        self._time = None

        self.channels = [Channel(self, 'green', 'g', other_names=['donor', 'd']),
                         Channel(self, 'red', 'r', other_names=['acceptor', 'a'])]
        self.channel_arrangement = [[[0, 1]]]
        # [[[0,1]]] # First level: frames, second level: y within frame, third level: x within frame
        # self.channel_arrangement = xr.DataArray([[[0,1]]], dims=('frame','y','x'))

        self.illumination_arrangement = [0]  # First level: frames, second level: illumination
        # self.illumination_arrangement = xr.DataArray([[True, False]], dims=('frame', 'illumination'), coords={'illumination': [0,1]}) # TODO: np.array([0]) >> list of list It would be good to have a default illumination_arrangement of np.array([0]), i.e. illumination 0 all the time?
        self._illumination_index_per_frame = None

        # self._darkfield_correction = None
        # self._flatfield_correction = None
        # self._general_background_correction = None
        # self._spatial_background_correction = None
        # self._temporal_background_correction = None
        # self._temporal_illumination_correction = None

        self._common_corrections = xr.Dataset()

        # self.load_corrections()

        self.header_is_read = False

    def __enter__(self):
        if self._with_counter == 0:
            self.open()
            # print('open')
        self._with_counter += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._with_counter -= 1
        if self._with_counter == 0:
            self.close()
            # print('close')

    def __repr__(self):
        return (f'{self.__class__.__name__}({str(self.filepath)})')

    def __getattr__(self, item):
        # if '_initialized' in self.__dict__ and not self.header_is_read:
        # if item != 'header_is_read' and not self.header_is_read:
        # if item == '_with_counter':
        #     raise ValueError()
        # print(item)

        if 'header_is_read' in self.__dict__.keys() and not self.header_is_read:
            # print(item+'2')
            self.read_header()
            return getattr(self, item)
        else:
            raise AttributeError(f'Attribute {item} not found')
        # return super().__getattribute__(item)

    @property
    def pixels_per_frame(self):
        return self.width * self.height

    @property
    def bitdepth(self):
        return self.data_type.itemsize * 8  # 8 bits in a byte

    @property
    def bytes_per_frame(self):
        return self.data_type.itemsize * self.pixels_per_frame

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    # @property
    # def channel_grid(self):
    #     """ numpy.array : number of channels in the horizontal and vertical dimension
    #
    #     Setting the channel_grid variable will assume equally spaced channels
    #     """
    #     return self._channel_grid
    #
    # @channel_grid.setter
    # def channel_grid(self, channel_grid):
    #     channel_grid = np.array(channel_grid)
    #     # Possibly support multiple cameras by adding a third dimension
    #     if len(channel_grid) == 2 and np.all(np.array(channel_grid) > 0):
    #         self._channel_grid = channel_grid
    #         self._number_of_channels = np.product(channel_grid)
    @property
    def number_of_illuminations(self):
        """ int : number of channels in the movie

        Setting the number of channels will divide the image horizontally in equally spaced channels.
        """
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
    def channel_arrangement(self):
        return self._channel_arrangement

    @channel_arrangement.setter
    def channel_arrangement(self, channel_arrangement):
        self._channel_arrangement = np.array(channel_arrangement)

    @property
    def channel_indices(self):
        return xr.DataArray(self.channel_arrangement.flatten(), dims='channel')

    # @property
    # def channel_indices_per_image(self):
    #     if self._channel_indices_per_frame is None:
    #         # frame_indices = self.frame_indices
    #         # illumination_indices = self.illumination_indices
    #         # self._illumination_index_per_frame = xr.DataArray(
    #         #     np.resize(self.illumination_arrangement, (len(frame_indices), len(illumination_indices))),
    #         #     dims=('frame', 'illumination'),
    #         #     coords={'frame': frame_indices, 'illumination': illumination_indices})
    #         channel_indices_flattened = self.channel_arrangement.reshape(len(self.channel_arrangement), -1)
    #         self._channel_indices_per_frame= xr.DataArray(
    #             np.resize(channel_indices_flattened, (self.number_of_frames, len(channel_indices_flattened[0]))),
    #             dims=('frame', 'channel'),
    #             coords={'frame': self.frame_indices}).stack(image=('frame','channel'))
    #         #TODO: Add name to other indices or remove this name
    #     return self._illumination_index_per_frame

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
        return xr.DataArray([illumination.index for illumination in self.illuminations], dims='illumination')

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

    # @property
    # def image_indices(self):
    #     # index = pd.MultiIndex.from_arrays([*self.image_indices_from_frame_indices(self.frame_indices).T],
    #     #                                   names=('frame', 'illumination', 'channel'))
    #     # return xr.DataArray(index, dims='image')
    #     return self.image_indices_from_frame_indices(xarray=True)

    # def image_indices_from_frame_indices(self, frame_indices=None, xarray=False):
    #     if frame_indices is None:
    #         frame_indices = self.frame_indices
    #     if isinstance(frame_indices, xr.DataArray):
    #         frame_indices = frame_indices.values
    #     # return self.image_indices.sel(image=self.image_indices.frame.isin(frame_indices))
    #     image_frame_indices = np.repeat(frame_indices, self.number_of_channels_per_frame)
    #     image_illumination_indices = np.repeat(self.illumination_index_per_frame.values[frame_indices],
    #                                            self.number_of_channels_per_frame)
    #     image_channel_indices = np.resize(self.channel_indices.values, len(image_frame_indices))
    #
    #     image_indices = np.vstack([image_frame_indices, image_illumination_indices, image_channel_indices]).T
    #     if xarray:
    #         image_indices = self.image_indices_to_xarray(image_indices)
    #
    #     return image_indices

    # def image_indices_to_xarray(self, image_indices):
    #     index = pd.MultiIndex.from_arrays([*image_indices.T], names=('frame', 'illumination', 'channel'))
    #     return xr.DataArray(index, dims='image')

    # TODO: remove this
    # def create_frame_info(self):
    #     # TODO: Use xarray instead of pandas
    #     # Perhaps store time, illumination and channel separately
    #     # files = [0] # For implementing multiple files
    #     frames = range(self.number_of_frames)
    #
    #     index = pd.Index(data=frames, name='frame')
    #     frame_info = pd.DataFrame(index=index, columns=['time', 'illumination', 'channel'])
    #     # self.frame_info['file'] = len(self.frame_info) * [list(range(2))] # For implementing multiple files
    #     # self.frame_info = self.frame_info.explode('file') # For implementing multiple files
    #     frame_info['time'] = frame_info.index.to_frame()['frame'].values
    #     if self.illumination_arrangement is not None:
    #         if len(self.illumination_arrangement)>1:
    #             frame_info['illumination'] = self.illumination_arrangement.tolist() * (self.number_of_frames // self.illumination_arrangement.shape[0])
    #         else:
    #             frame_info['illumination'] = [0] * self.number_of_frames
    #     else:
    #         frame_info['illumination'] = [0] * self.number_of_frames
    #     frame_info['channel'] = self.channel_arrangement.tolist() * (self.number_of_frames // self.channel_arrangement.shape[0])
    #
    #     frame_info = frame_info.explode('channel').explode('channel')
    #
    #     categorical_columns = ['illumination', 'channel']
    #     frame_info[categorical_columns] = frame_info[categorical_columns].astype('category')
    #
    #     self.frame_info = frame_info

    @property
    def pixel_to_stage_coordinates_transformation(self):
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
    def boundaries_metric(self):
        # #         Formatted as two coordinates, with the lowest and highest x and y values respectively
        horizontal_boundaries = np.array([0, self.width_metric])
        vertical_boundaries = np.array([0, self.height_metric])
        return np.vstack([horizontal_boundaries, vertical_boundaries]).T

    @property
    def boundaries_stage(self):
        return self.pixel_to_stage_coordinates_transformation(self.channels[0].boundaries)

    def read_header(self):
        self._read_header()
        if not (self.rot90 % 2 == 0):
            width = self.width
            height = self.height
            self.width = height
            self.height = width

        self.header_is_read = True

    def read_frame(self, frame_index, **kwargs):
        return self.read_frames([frame_index], **kwargs).squeeze(axis=0)

    def read_frames(self, frame_indices=None, apply_corrections=True, xarray=True, flatten_channels=False):
        if frame_indices is None:
            frame_indices = self.frame_indices.values

        frames = self._read_frames(frame_indices)
        # frames = xr.DataArray(frames, dims=('frame', 'y', 'x'))
        frames = np.rot90(frames, self.rot90, axes=(1, 2))

        if len(self.channel_arrangement) > 1:
            raise NotImplementedError('Channel arrangement where frames indicated different channels not implemented')
            # Perhaps remove the outermost layer from channel_configuration
            # Or add this to separate and flatten channels

        frames = self.separate_channels(frames)
        # frames = np.stack([channel.crop_images(images) for channel in self.channels]

        if apply_corrections:  # and self.correct_images
            frames = self.apply_corrections(frames, frame_indices)

        if xarray:
            frames = self.frames_to_xarray_dataarray(frames, frame_indices)

        if flatten_channels:
            frames = self.flatten_channels(frames)

        return frames

    @property
    def channel_rows(self):
        return len(self.channel_arrangement[0])

    @property
    def channel_columns(self):
        return len(self.channel_arrangement[0, 0])

    def separate_channels(self, frames):
        # if frames.ndim == 2:
        #     frames = frames[None, :, :]
        # return expand_axes(frames, (channel_rows, channel_columns), from_axes=(1, 2))
        # return expand_axes(frames, (channel_rows, channel_columns), from_axes=(1, 2), to_axes=(0, 0))
        return xr.apply_ufunc(
            expand_axes, frames, input_core_dims=[['y', 'x']], output_core_dims=[['channel', 'y', 'x']],
            exclude_dims=set(['y', 'x']),
            kwargs={"expand_into": (self.channel_rows, self.channel_columns), "from_axes": (-2, -1),
                    "to_axes": (frames.ndim,) * 2, "new_axes_positions": [-3]}
        )
        # return xr.apply_ufunc(
        #     expand_axes, frames, input_core_dims=[['image', 'y', 'x'][-frames.ndim:]], output_core_dims=[['image', 'y', 'x']],
        #     exclude_dims=set(['image', 'y', 'x']),
        #     kwargs={"expand_into": (channel_rows, channel_columns), "from_axes": (-2, -1), "to_axes": (-3, -3)}
        # )

        # frames = frames.transpose('frame', 'y', 'x', ...)
        # new = split_along_axes(frames.values, (channel_rows, channel_columns), from_axes=(1, 2))
        # return xr.DataArray(new, dims=['frame','y','x','channel'])

    def flatten_channels(self, frames):
        # return split_along_axes(frames, (channel_rows, channel_columns), from_axes=(1, 2), inverse=True)
        # return split_along_axes(frames, (channel_rows, channel_columns), from_axes=(1, 2), inverse=True)
        # if frames.shape[-3]//channel_columns//channel_rows == 1:
        #     output_core_dims = [['y', 'x']]
        # else:
        #     output_core_dims = [['image', 'y', 'x']]

        return xr.apply_ufunc(
            expand_axes, frames, input_core_dims=[['channel', 'y', 'x']], output_core_dims=[['y', 'x']],
            exclude_dims=set(['x', 'y']),
            kwargs={"expand_into": (self.channel_rows, self.channel_columns), "from_axes": (-2, -1),
                    "to_axes": (-3, -3),
                    "inverse": True, "squeeze": True}
        )

    def frames_to_xarray_dataarray(self, frames, frame_indices):
        frames = xr.DataArray(frames,
                              dims=('frame', 'channel', 'y', 'x'),
                              coords={'frame': frame_indices,
                                      'illumination': self.illumination_index_per_frame[frame_indices],
                                      'channel': self.channel_indices})

        if self.time is not None:
            frames = frames.assign_coords(time=self.time[frames.frame])

        return frames

    def get_channel(self, image, channel='d'):
        if channel in [None, 'all']:
            return image

        if not isinstance(channel, Channel):
            channel = self.get_channel_from_name(channel)

        return channel.crop_image(image)

    def get_channel_from_name(self, channel_name):
        """Get the channel index belonging to a specific channel (name)
        If

        Parameters
        ----------
        channel : str or int
            The name or number of a channel

        Returns
        -------
        i: int
            The index of the channel to which the channel name belongs

        """
        for channel in self.channels:
            if channel_name in channel.names or channel_name == channel:
                return channel
        else:
            raise ValueError('Channel name not found')

    def get_channels_from_names(self, channel_names):
        """Get the channel index belonging to a specific channel (name)
        If

        Parameters
        ----------
        channel : str or int
            The name or number of a channel

        Returns
        -------
        i: int
            The index of the channel to which the channel name belongs

        """
        if channel_names in [None, 'all']:
            return self.channels

        if not isinstance(channel_names, list):
            channel_names = [channel_names]

        return [self.get_channel_from_name(channel_name) for channel_name in channel_names]

    def get_channel_indices_from_names(self, channel_names):
        channels = self.get_channels_from_names(channel_names)
        return [channel.index for channel in channels]

    def saveas_tif(self):
        tif_filepath = self.writepath.joinpath(self.name + '.tif')
        tif_filepath.unlink(missing_ok=True)

        for i in range(self.number_of_frames):
            frame = self.read_frames([i], apply_corrections=False, xarray=False)
            tifffile.imwrite(tif_filepath, frame, append=True)

            #     tifffile.imwrite(self.writepath.joinPath(f'{self.name}_fr{frame_number}.tif'), image,  photometric='minisblack')

    def make_projection_image(self, projection_type='average', frame_range=(0,20), apply_corrections=True, illumination=None, write=False,
                              return_image=True, flatten_channels=True, intensity_range=None, color_map='gray'):
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
        if frame_range[1] > self.number_of_frames:
            frame_range[1] = self.number_of_frames
            warnings.warn(f'Frame range exceeds available frames, used frame range {frame_range} instead')

        frame_indices = self.frame_indices.values[slice(*frame_range)]

        illumination_indices = self.get_illumination_indices_from_names(illumination)
        illumination_index = np.intersect1d(illumination_indices, self.illumination_indices_in_movie)[0]

        # Select frame_indices with illumination
        frame_indices = frame_indices[self.illumination_index_per_frame.values[frame_indices] == illumination_index]

        # Calculate sum of frames and find mean
        image = self.separate_channels(np.zeros((self.height, self.width)).astype('float32'))

        frame_indices_subsets = np.array_split(frame_indices, len(frame_indices) // self.chunk_size + 1)

        if projection_type == 'average':
            number_of_frames = len(frame_indices)
            # if len(frame_indices) > 100:
                # print(f'\n Making average image of {self.name}')
            with self:
                for frame_indices_subset in tqdm.tqdm(frame_indices_subsets, desc='Average image'):
                    # if len(frame_indices) > 100 and i % 13 == 0:
                    #     sys.stdout.write(
                    #         f'\r   Processing frame {frame_index} in {frame_indices[0]}-{frame_indices[-1]}')
                    frames = self.read_frames(frame_indices_subset, apply_corrections=apply_corrections,
                                              xarray=False, flatten_channels=False)
                    image = image + frames.sum(axis=0)
                #TODO: Check whether this is a good way to average, i.e. do the values not get too big.
            image = (image / number_of_frames).astype('float32')
        elif projection_type == 'maximum':
            # print(f'\n Making maximum projection image of {self.name}')
            with self:
                for frame_indices_subset in tqdm.tqdm(frame_indices_subsets, desc='Maximum projection image'):
                    # if i % 13 == 0:
                    #     sys.stdout.write(
                    #         f'\r   Processing frame {frame_index} in {frame_indices[0]}-{frame_indices[-1]}')
                    frames = self.read_frames(frame_indices_subset, xarray=False, flatten_channels=False)
                    image = np.maximum(image, frames.max(axis=0))
            # sys.stdout.write(f'\r   Processed frames {frame_indices[0]}-{frame_indices[-1]}\n')

        if write:
            filename = Movie.image_info_to_filename(self.name, fov_index=self.fov_index, projection_type=projection_type,
                                                    frame_range=frame_range, illumination=illumination_index, apply_corrections=apply_corrections)
            filepath = self.writepath.joinpath(filename)
            write_image = self.flatten_channels(image)
            if write in [True, 'tif']:
                if hasattr(self, 'pixel_size'):
                    resolution = 1/self.pixel_size
                else:
                    resolution = None
                tifffile.imwrite(filepath.with_suffix('.tif'), write_image,
                                 resolution=resolution,
                                 imagej=True,
                                 metadata={'unit': 'um',
                                           'axes': 'YX'}
                                 )
            elif write in ['png']:
                filepath = filepath.with_name(filepath.name + f'_v{intensity_range[0]}-{intensity_range[1]}')
                if intensity_range is None:
                    intensity_range = self.intensity_range
            # plt.imsave(filepath.with_suffix('.tif'), image, format='tif', cmap=colour_map, vmin=self.intensity_range[0], vmax=self.intensity_range[1])
                plt.imsave(filepath.with_suffix('.png'), write_image, vmin=intensity_range[0], vmax=intensity_range[1], cmap=color_map)

        if return_image:
            if flatten_channels:
                return self.flatten_channels(image)
            else:
                return image

    def make_projection_images(self, projection_type='average', frame_range=(0, 20)):
        # Perhaps put this in make_projection_image as a special type of cmap
        for illumination_index in range(self.number_of_illuminations_in_movie):
            image = self.make_projection_image(projection_type, frame_range=(0,20), illumination=illumination_index,
                                               write=True, return_image=True, flatten_channels=False)
            channel_images = []
            for channel_index in range(self.number_of_channels):
                channel_image = image[channel_index]
                channel_image = (channel_image - self.intensity_range[0]) / (self.intensity_range[1] - self.intensity_range[0]) # TODO: make separate intensity range for each channel
                channel_images.append(self.channels[channel_index].colour_map(channel_image, bytes=True))

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

            # if 'flatfield_correction' in corrections:
            #     flatfield = corrections.flatfield_correction.sel(illumination=illumination, channel=channel).values
            # else:
            #     flatfield = None
            #
            # if 'darkfield_correction' in corrections:
            #     darkfield = corrections.darkfield_correction.sel(illumination=illumination, channel=channel).values
            # else:
            #     darkfield = None

            correction = determine_single_value_background_correction(average_image, method)#, flatfield, darkfield)

            # if 'temporal_illumination_correction' in corrections:
            #     correction /= corrections.temporal_illumination_correction[illumination, channel].mean().item()
            #
            # if 'temporal_background_correction' in corrections:
            #     correction -= corrections.temporal_background_correction[frame_indices_subset, channel].mean().item()
            #
            # if 'spatial_background_correction' in corrections:
            #     correction -= corrections.spatial_background_correction[illumination, channel].mean().item()

            general_background_correction[dict(illumination=illumination, channel=channel)] = correction

        add_configuration_to_dataarray(general_background_correction, Movie.determine_general_background_correction,
                                       locals(), units='a.u.') # TODO: Link to units in movie metadata?

        # self.general_background_correction = general_background_correction
        # self.save_corrections(general_background_correction=general_background_correction,
        #                       temporal_background_correction=None, spatial_background_correction=None)
        self.save_corrections(general_background_correction=general_background_correction)

    def determine_temporal_background_correction(self, method='median', use_existing=False):
        #Todo: pass method kwargs
        if use_existing and 'temporal_background_correction' in self.corrections:
            return

        # self.spatial_background_correction = None
        self.save_corrections(temporal_illumination_correction=None,
                              temporal_background_correction=None,
                              spatial_background_correction=None,
                              general_background_correction=None)

        frames = self.read_frames(frame_indices=None, apply_corrections=True, xarray=False)

        temporal_background_correction = xr.DataArray(0, dims=('frame', 'channel'),
                                                      coords={'frame': self.frame_indices,
                                                              'channel': self.channel_indices},
                                                      name='temporal_background_correction')

        # corrections = self.corrections

        for illumination, channel in itertools.product(self.illumination_indices_in_movie, np.array(self.channel_indices)):

            frame_indices_subset = (self.illumination_index_per_frame==illumination).frame
            frames_subset = frames[frame_indices_subset, channel]

            # if 'flatfield_correction' in corrections:
            #     flatfield = corrections.flatfield_correction.sel(illumination=illumination, channel=channel).values
            # else:
            #     flatfield = None
            #
            # if 'darkfield_correction' in corrections:
            #     darkfield = corrections.darkfield_correction.sel(illumination=illumination, channel=channel).values
            # else:
            #     darkfield = None

            correction = determine_temporal_background_correction(frames_subset, method)#, flatfield, darkfield)

            # if 'general_background_correction' in corrections:
            #     correction -= corrections.general_background_correction[illumination, channel].item()

            # if 'spatial_background_correction' in corrections:
            #     correction -= corrections.spatial_background_correction[illumination, channel].mean().item()

            temporal_background_correction[dict(frame=frame_indices_subset, channel=channel)] = correction

        add_configuration_to_dataarray(temporal_background_correction, Movie.determine_temporal_background_correction,
                                       locals(), units='a.u.') # TODO: Link to units in movie metadata?

        # self.temporal_background_correction = temporal_background_correction
        # self.save_corrections(temporal_background_correction=temporal_background_correction,
        #                       spatial_background_correction=None)
        self.save_corrections(temporal_background_correction=temporal_background_correction)#,
                              # spatial_background_correction=None,
                              # general_background_correction=None)

    def determine_spatial_background_correction(self, method='median_filter', frame_range=(0, 20), use_existing=False,
                                                **kwargs):
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

        # corrections = self.corrections

        for illumination, channel in itertools.product(self.illumination_indices_in_movie,
                                                       np.array(self.channel_indices)):
            frame_selection = (self.illumination_index_per_frame[frame_indices] == illumination).values
            average_image = frames[frame_selection, channel].mean(axis=0)
            # if 'flatfield_correction' in corrections:
            #     flatfield = corrections.flatfield_correction.sel(illumination=illumination, channel=channel).values
            # else:
            #     flatfield = None
            #
            # if 'darkfield_correction' in corrections:
            #     darkfield = corrections.darkfield_correction.sel(illumination=illumination, channel=channel).values
            # else:
            #     darkfield = None

            correction = determine_spatial_background_correction(average_image, method, **kwargs) # flatfield, darkfield, **kwargs)

            # if 'general_background_correction' in corrections:
            #     correction -= corrections.general_background_correction[illumination, channel].item()

            # if 'temporal_background_correction' in corrections:
            #     correction -= corrections.temporal_background_correction[frame_indices_subset, channel].mean().item()

            spatial_background_correction[dict(illumination=illumination, channel=channel)] = correction

        add_configuration_to_dataarray(spatial_background_correction, Movie.determine_spatial_background_correction,
                                       locals(), units='a.u.') # TODO: Link to units in movie metadata?

        # self.spatial_background_correction = spatial_background_correction
        # self.save_corrections(spatial_background_correction=spatial_background_correction)
        self.save_corrections(spatial_background_correction=spatial_background_correction)#,
                              # temporal_background_correction=None,
                              #general_background_correction=None)
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

    # def load_corrections(self):
    #     corrections_filepath = self.filepath.with_name(self.name + '_corrections.nc')
    #     if corrections_filepath.exists():
    #         corrections = xr.load_dataset(corrections_filepath, engine='h5netcdf')
    #         # for key, correction in corrections.data_vars.items():
    #         #     self.__setattr__(key, correction)
    #     return corrections

    @property
    def configuration(self):
        configuration = dict(rot90=self.rot90)
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

#     def apply_corrections(self, frames, frame_indices):
#
#
#         return apply_corrections(frames, frame_indices, self.darkfield_correction.values,
#                                  self.flatfield_correction.values, self.illumination_correction.values,
#                                  self.background_correction.values)
#
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

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # divider = make_axes_locatable(axes[-1])
        # cax = divider.append_axes("right", "4%", pad="15%")
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


class Channel:
    def __init__(self, movie, name, short_name, other_names=[], colour_map=None):
        self.movie = movie
        self.name = name
        self.short_name = short_name
        self.other_names = other_names
        if colour_map is None:
            channel_colour = \
            list({'green', 'red', 'blue'}.intersection([self.name, self.short_name] + self.other_names))[0]
            self.colour_map = make_colour_map(channel_colour)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.name})')

    @property
    def names(self):
        return [self.index, str(self.index), self.name, self.short_name] + self.other_names

    @property
    def index(self):
        try:
            return self.movie.channels.index(self)
        except:
            pass

    @property
    def location(self):
        return [int(i) for i in np.where(self.movie.channel_arrangement == self.index)]

    @property
    def width(self):
        return self.movie.width // self.movie.channel_arrangement.shape[2]

    @property
    def height(self):
        return self.movie.height // self.movie.channel_arrangement.shape[1]
        # for frame_index, frame in enumerate(self.channel_arrangement):
        #     for y_index, y in enumerate(frame):
        #         try:
        #             x_index = y.index(channel_index)
        #             return frame_index, y_index, x_index
        #         except ValueError:
        #             pass

    @property
    def dimensions(self):
        return np.array([self.width, self.height])

    @property
    def origin(self):
        return [self.width * self.location[2],
                self.height * self.location[1]]

    @property
    def boundaries(self):
        # channel_boundaries: np.array
        # #         Formatted as two coordinates, with the lowest and highest x and y values respectively
        horizontal_boundaries = np.array([0, self.width]) + self.width * self.location[2]
        vertical_boundaries = np.array([0, self.height]) + self.height * self.location[1]
        return np.vstack([horizontal_boundaries, vertical_boundaries]).T

    @property
    def vertices(self):
        #     channel_vertices : np.array
        #         Four coordinates giving the four corners of the channel
        #         Coordinates form a closed shape
        channel_vertices = np.array([self.origin, ] * 4)
        channel_vertices[[1, 2], 0] += self.width
        channel_vertices[[2, 3], 1] += self.height
        return channel_vertices

    def crop_image(self, image):
        return image[self.boundaries[0, 1]:self.boundaries[1, 1],
               self.boundaries[0, 0]:self.boundaries[1, 0]]

    def crop_images(self, images):
        return images[:, self.boundaries[0, 1]:self.boundaries[1, 1],
               self.boundaries[0, 0]:self.boundaries[1, 0]]


class MoviePlotter:
    # Adapted from Matplotlib Image Slices Viewer
    def __init__(self, movie):
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
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.movie.read_frame(self.ind, flatten_channels=True, xarray=False))
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


def make_colour_map(colour, N=256):
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