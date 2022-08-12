import re
import sys
import itertools
import warnings
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
import scipy.optimize
from skimage.transform import AffineTransform

from trace_analysis.image_adapt.rolling_ball import rollingball
from trace_analysis.image_adapt.find_threshold import remove_background, get_threshold
from trace_analysis.timer import Timer
from trace_analysis.correction.shading_correction import get_photobleach


class Movie:
    @classmethod
    def type_dict(cls):
        # It is important to import all movie files to recognize them by subclasses.
        # Perhaps we can make this more elegant in some way.
        from trace_analysis.movie.sifx import SifxMovie
        from trace_analysis.movie.pma import PmaMovie
        from trace_analysis.movie.tif import TifMovie
        from trace_analysis.movie.nd2 import ND2Movie
        from trace_analysis.movie.nsk import NskMovie
        from trace_analysis.movie.binary import BinaryMovie
        return {extension: subclass for subclass in cls.__subclasses__() for extension in subclass.extensions}

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

        self.time = None

        self.channels = [Channel(self, 'green', 'g', other_names=['donor', 'd']),
                         Channel(self, 'red', 'r', other_names=['acceptor', 'a'])]
        self.channel_arrangement = [[[0, 1]]]
        # [[[0,1]]] # First level: frames, second level: y within frame, third level: x within frame
        # self.channel_arrangement = xr.DataArray([[[0,1]]], dims=('frame','y','x'))

        self.illuminations = [Illumination(self, 'green', 'g'), Illumination(self, 'red', 'r')]
        self.illumination_arrangement = [0]  # First level: frames, second level: illumination
        # self.illumination_arrangement = xr.DataArray([[True, False]], dims=('frame', 'illumination'), coords={'illumination': [0,1]}) # TODO: np.array([0]) >> list of list It would be good to have a default illumination_arrangement of np.array([0]), i.e. illumination 0 all the time?
        self._illumination_index_per_frame = None

        self.darkfield_correction = None
        self.flatfield_correction = None
        self.background_correction = None
        self.illumination_correction = None

        self.load_corrections()

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

    def get_illumination_from_name(self, illumination_name):
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
        for illumination in self.illuminations:
            if illumination_name in illumination.names or illumination_name == illumination:
                return illumination
        else:
            raise ValueError('Illumination name not found')

    def get_illuminations_from_names(self, illumination_names):
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
            return self.channels

        if not isinstance(illumination_names, list):
            illumination_names = [illumination_names]

        return [self.get_illumination_from_name(illumination_name) for illumination_name in illumination_names]

    def get_illumination_indices_from_names(self, illumination_names):
        illuminations = self.get_illuminations_from_names(illumination_names)
        return [illumination.index for illumination in illuminations]

    def saveas_tif(self):
        tif_filepath = self.writepath.joinpath(self.name + '.tif')
        tif_filepath.unlink(missing_ok=True)

        for i in range(self.number_of_frames):
            frame = self.read_frames([i], apply_corrections=False, xarray=False)
            tifffile.imwrite(tif_filepath, frame, append=True)

            #     tifffile.imwrite(self.writepath.joinPath(f'{self.name}_fr{frame_number}.tif'), image,  photometric='minisblack')

    def make_projection_image(self, projection_type='average', frame_range=(0,20), illumination=None, write=False,
                              return_image=True, flatten_channels=True):
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

        filename_addition = ''

        frame_range = list(frame_range)
        if frame_range[1] > self.number_of_frames:
            frame_range[1] = self.number_of_frames
            raise RuntimeWarning(f'Frame range exceeds available frames, used frame range {frame_range} instead')

        frame_indices = np.arange(*frame_range)

        illumination_indices = self.get_illumination_indices_from_names(illumination)
        illumination_index = np.intersect1d(illumination_indices, self.illumination_indices_in_movie)[0]

        # Select frame_indices with illumination
        frame_indices = frame_indices[self.illumination_index_per_frame.values[frame_indices] == illumination_index]

        # Calculate sum of frames and find mean
        image = self.separate_channels(np.zeros((self.height, self.width)).astype('float32'))

        frame_indices_subsets = np.array_split(frame_indices, len(frame_indices) // self.chunk_size + 1)

        if projection_type == 'average':
            number_of_frames = len(frame_indices)
            if len(frame_indices) > 100:
                print(f'\n Making average image of {self.name}')
            with self:
                for frame_indices_subset in frame_indices_subsets:
                    # if len(frame_indices) > 100 and i % 13 == 0:
                    #     sys.stdout.write(
                    #         f'\r   Processing frame {frame_index} in {frame_indices[0]}-{frame_indices[-1]}')
                    frames = self.read_frames(frame_indices_subset, xarray=False, flatten_channels=False)
                    image = image + frames.sum(axis=0)
            image = (image / number_of_frames)
        elif projection_type == 'maximum':
            print(f'\n Making maximum projection image of {self.name}')
            with self:
                for frame_indices_subset in frame_indices_subsets:
                    # if i % 13 == 0:
                    #     sys.stdout.write(
                    #         f'\r   Processing frame {frame_index} in {frame_indices[0]}-{frame_indices[-1]}')
                    frames = self.read_frames(frame_indices_subset, xarray=False, flatten_channels=False)
                    image = np.maximum(image, frames.max(axis=0))
            sys.stdout.write(f'\r   Processed frames {frame_indices[0]}-{frame_indices[-1]}\n')

        if write:
            filename = image_info_to_filename(self.name, image_type=projection_type, frame_range=frame_range,
                                              illumination_index=illumination_index)
            filepath = self.writepath.joinpath(filename)
            tifffile.imwrite(filepath.with_suffix('.tif'), self.flatten_channels(image),
                             resolution=1/self.pixel_size,
                             imagej=True,
                             metadata={'unit': 'um',
                                       'axes': 'YX'}
                             )
            # plt.imsave(filepath.with_suffix('.tif'), image, format='tif', cmap=colour_map, vmin=self.intensity_range[0], vmax=self.intensity_range[1])
            # plt.imsave(filepath.with_suffix('.png'), image, cmap=colour_map, vmin=self.intensity_range[0], vmax=self.intensity_range[1])

        if return_image:
            if flatten_channels:
                return self.flatten_channels(image)
            else:
                return image

    def make_projection_images(self, projection_type='average', start_frame=0, number_of_frames=20):
        # TODO: test this method
        images = []
        for illumination_index, channel_index in itertools.product(range(self.number_of_illuminations_in_movie), range(self.number_of_channels)):
            image = self.make_projection_image(projection_type, start_frame, number_of_frames,
                                               illumination_index, channel_index, write=True, return_image=True)
            image = (image - self.intensity_range[0]) / (self.intensity_range[1] - self.intensity_range[0])
            images.append(self.channels[channel_index].colour_map(image, bytes=True))

        images_combined = np.hstack(images)
        filepath = self.writepath.joinpath(self.name + '_' + projection_type[:3] + f'_{number_of_frames}fr')
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

    # TODO: This should be updated
    # def determine_static_background_correction(self, image, method='per_channel'):
    #     if method == 'rollingball':
    #         background = rollingball(image, self.width_pixels / 10)[1]  # this one is not used in pick_spots_akaze
    #         image_correct = image - background
    #         image_correct[image_correct < 0] = 0
    #         threshold = get_threshold(image_correct)
    #         return remove_background(image_correct, threshold)
    #     elif method == 'per_channel':  # maybe there is a better name
    #         sh = np.shape(image)
    #         threshold_donor = get_threshold(self.get_channel(image, 'donor'))
    #         threshold_acceptor = get_threshold(self.get_channel(image, 'acceptor'))
    #         background = np.zeros(np.shape(image))
    #         background[:, 0:sh[0] // 2] = threshold_donor
    #         background[:, sh[0] // 2:] = threshold_acceptor
    #         return remove_background(image, background)
    #
    #     # note: optionally a fixed threshold can be set, like with IDL
    #     # note 2: do we need a different threshold for donor and acceptor?

    def determine_temporal_background_correction(self, method='median'):
        frames = self.read_frames(frame_indices=None, apply_corrections=False, xarray=False)

        temporal_background_correction = xr.DataArray(0, dims=('frame', 'channel'),
                                                      coords={'frame': self.frame_indices,
                                                              'channel': self.channel_indices},
                                                      name='background_correction')

        for illumination, channel in itertools.product(self.illumination_indices_in_movie, np.array(self.channel_indices)):

            frame_indices_subset = (self.illumination_index_per_frame==illumination).frame
            frames_subset = frames[frame_indices_subset, channel]

            flatfield = self.flatfield_correction.sel(illumination=illumination, channel=channel).values
            darkfield = self.darkfield_correction.sel(illumination=illumination, channel=channel).values

            if method == 'BaSiC':
                channel_dimensions = frames.shape[-2:][::-1]  # size should be given in (x,y) for get_photobleach
                size = (channel_dimensions / np.max(channel_dimensions) * 256).astype(int)
                correction = get_photobleach(frames_subset, flatfield, darkfield, size=size).flatten()
            elif method == 'gaussian_filter':
                correction = np.array(
                    [scipy.ndimage.gaussian_filter(((frame - darkfield) / flatfield), sigma=0.5, mode='wrap').mean()
                     for frame in frames_subset])
                # This comes down to taking the mean of the corrected image
            elif method == 'mean':
                correction = np.array([((frame - darkfield) / flatfield).mean() for frame in frames_subset])
            elif method == 'median':
                correction = np.array([np.median((frame - darkfield) / flatfield) for frame in frames_subset])
            elif method == 'minimum_filter':
                correction = np.array(
                        [scipy.ndimage.minimum_filter(((frame - darkfield) / flatfield), size=15, mode='wrap').mean()
                         for frame in frames_subset])
            elif method == 'median_filter':
                #     temporal_background_correction[dict(channel=channel)] = \
                #         np.array([scipy.ndimage.minimum_filter(((frame - darkfield) / flatfield), size=15, mode='wrap').mean() for frame in frames_channel])
                correction = np.array(
                        [scipy.ndimage.median_filter(((frame - darkfield) / flatfield), size=15, mode='wrap').mean() for
                         frame in frames_subset])
            elif method == 'fit_background_peak':
                correction = [gaussian_maximum_fit((frame - darkfield) / flatfield) for frame in frames_subset]
            else:
                raise ValueError(f'Method {method} not found')

            temporal_background_correction[dict(frame=frame_indices_subset, channel=channel)] = correction

        self.background_correction = temporal_background_correction
        self.save_corrections(self.background_correction)

    def load_corrections(self):
        corrections_filepath = self.filepath.with_name(self.name + '_corrections.nc')
        if corrections_filepath.exists():
            corrections = xr.load_dataset(corrections_filepath, engine='h5netcdf')
            for key, correction in corrections.data_vars.items():
                self.__setattr__(key, correction)

    def save_corrections(self, corrections):
        corrections.to_netcdf(self.filepath.with_name(self.name + '_corrections.nc'), mode='a', engine='h5netcdf')

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
        for illumination_index in np.unique(illumination_indices):
            frame_indices_with_illumination = np.array(illumination_indices == illumination_index)

            if self.darkfield_correction is not None:
                frames[frame_indices_with_illumination] -= self.darkfield_correction.values[None, illumination_index]

            if self.flatfield_correction is not None:
                frames[frame_indices_with_illumination] /= self.flatfield_correction.values[None, illumination_index]

            if self.illumination_correction is not None:
                frames[frame_indices_with_illumination] /= \
                    self.illumination_correction.values[frame_indices][frame_indices_with_illumination, :, None, None]

            if self.background_correction is not None:
                frames[frame_indices_with_illumination] -= \
                    self.background_correction.values[frame_indices][frame_indices_with_illumination, :, None, None]

        return frames


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
        return [self.index, self.name, self.short_name] + self.other_names

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


class Illumination:
    def __init__(self, movie, name, short_name='', other_names=[]):
        self.movie = movie
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
            return self.movie.illuminations.index(self)
        except:
            pass


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

        self.im = ax.imshow(self.movie.read_frame(self.ind))
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.movie.read_frame(self.ind))
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



def image_info_from_filename(filename):
    image_info = {}

    fov_index_result = re.search('(?<=_fov)\d*(?=[_.])', filename)
    if fov_index_result is not None:
        image_info['fov_index'] = int(fov_index_result.group())

    if '_ave' in filename:
        image_info['image_type'] = 'average'
    elif '_max' in filename:
        image_info['image_type'] = 'maximum'

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
        image_info['illumination_index'] = None #list(self.illumination_indices.values)
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

    return image_info

def image_info_to_filename(filename, fov_index=None, image_type=None, frame_range=None, illumination_index=None):
    #if 'fov_info' in self.__dict__.keys() and self.fov_info: # Or hasattr(self, 'fov_info')
    if fov_index is not None:
        # filename += f'_fov{self.fov_info["fov_chosen"]:03d}'
        filename += f'_fov{fov_index:03d}'

    if image_type is not None:
        filename += '_' + image_type[:3]

    if frame_range is not None:
        filename += str(range(*frame_range)).replace('range(', '_f').replace(', ', '-').replace(')', '')

    if illumination_index is not None: # and self.number_of_illuminations_in_movie > 1:
        filename += f'_i{illumination_index}'

    return filename


def gaussian_maximum_fit(frame, width_around_peak_fitted=200):
    def gauss_function(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    count, edges = np.histogram(frame.flatten(), bins=width_around_peak_fitted)
    # max_bin_center = edges[count.argmax():count.argmax()+2].mean()

    bincenters = (edges[:-1] + edges[1:]) / 2
    max_bin_center = bincenters[count.argmax()]

    width = 200
    selection = np.vstack(
        [max_bin_center - width / 2 < bincenters, bincenters < max_bin_center + width / 2]).all(
        axis=0)
    x = bincenters[selection]
    y = count[selection]

    popt, pcov = scipy.optimize.curve_fit(gauss_function, x, y, p0=[count.max(), max_bin_center, 100])#, maxfev=2000)
    return popt[1]