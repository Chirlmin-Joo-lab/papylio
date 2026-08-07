# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:50:57 2020

@author: mwdoc
https://github.com/soft-matter/pims_nd2
read_header and read_frame adapted, def __init__ unchanged
"""

from pathlib import Path
import os, sys

import time
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from papylio.movie.movie import Movie


class ND2Movie(Movie):
    """Movie class for reading Nikon ND2 image files.

    Handles loading and processing of Nikon ND2 format microscopy images,
    including support for multi-frame acquisitions and multiple fields of view.
    Reads metadata including pixel size, stage coordinates, and illumination info.
    """
    extensions = ['.nd2']

    def __init__(self, arg, *args, **kwargs):
        """Initialize ND2Movie instance.

        Parameters
        ----------
        arg : str or Path
            Path to the ND2 file
        *args
            Additional positional arguments passed to Movie parent class
        **kwargs
            Additional keyword arguments passed to Movie parent class

        Notes
        -----
        - Supports multi-FOV (field of view) ND2 files
        - FOV index is extracted from filename if present (e.g., '_fov001')
        """
        super().__init__(arg, *args, **kwargs)
        # super().__init__(arg)

        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name

        if 'fov' in self.name:
            token_position = self.name.find('_fov')
            self.fov_index = int(self.name[token_position+4:])
            self.name = self.name[:token_position]
            self.filepath = self.filepath.with_name(self.name).with_suffix('.nd2')
        else:
            self.fov_index = None

        # setting for multi fov measurement
        # # TODO: Move fov info / fov selection here if possible
        # self.fov_info = None
        # if 'fov_info' in kwargs:
        #     self.fov_info = kwargs['fov_info']  # fov=Field of View

        self.threshold = {'view': (0, 200),
                          'point-selection': (45, 25)
                          }

        # We should probably put this in the configuration file
        # SHK: self.rotation should be set before reading the header.
        # self.rotation = 1

        # self.read_header()

        # self.time = self.time[self.fov_info['first_frame_of_each_fov'][self.fov_info['fov_chosen']]:(self.fov_info['last_frame_of_each_fov'][self.fov_info['fov_chosen']]+1)]
        # self.illumination = self.illumination[self.fov_info['first_frame_of_each_fov'][self.fov_info['fov_chosen']]:(self.fov_info['last_frame_of_each_fov'][self.fov_info['fov_chosen']]+1)]
        # self.create_frame_info()  # Possibly move to Movie later on

        # self._initialized = True

        self.file = None # Note this is for the tif file, not the File class.

    def open(self):
        """Open ND2 file for reading.

        Creates an ND2Reader object to access file contents and configures
        the iteration order for frames and channels.
        """
        # from nd2reader import ND2Reader
        # self.file = ND2Reader(str(self.filepath))
        # if 'c' in self.file.axes:
        #     self.file.iter_axes = 'tc'
        # else:
        #     self.file.iter_axes = 't'
        import nd2
        self.file = nd2.ND2File(self.filepath)

    def close(self):
        """Close the ND2 file."""
        self.file.close()

    def _read_metadata(self):
        """Read and parse ND2 file header and metadata.

        Extracts image dimensions, frame/FOV information, channel/illumination
        data, pixel calibration, and stage coordinates from ND2 metadata.
        Automatically detects multiple fields of view based on stage position changes.
        """
        with self:
            self.width = self.file.sizes['X']
            self.height = self.file.sizes['Y']
            self.number_of_frames = self.file.sizes['T']
            self.pixel_size = self.file.voxel_size()[0:2]

            time_ms =  np.full(self.number_of_frames, np.nan)

            stage_coordinates_per_frame = np.full((self.number_of_frames, 2), np.nan)

            for i in range(self.number_of_frames):
                frame_metadata = self.file.frame_metadata(i)
                time_ms[i] = frame_metadata.channels[0].time.relativeTimeMs
                stage_coordinates_per_frame[i] = frame_metadata.channels[0].position.stagePositionUm[0:2]

            position_tolerance = 10  # xy tol = tolerance in um
            stage_coordinates_per_frame_round = np.round(stage_coordinates_per_frame / position_tolerance) * position_tolerance
            stage_coordinates_round, indices, fov_per_frame = np.unique(stage_coordinates_per_frame_round, return_index=True, return_inverse=True, axis=0)
            self.stage_coordinates = stage_coordinates_per_frame[indices] # Stage coordinates of first frame of the FOv to keep accuracy
            self.number_of_fov = len(self.stage_coordinates)

            if self.fov_index is not None:
                self.fov_frames = np.where(fov_per_frame == self.fov_index)[0]
                self.number_of_frames = len(self.fov_frames)
            else:
                self.fov_frames = np.where(fov_per_frame == 0)[0]

            self.time = xr.DataArray(time_ms[self.fov_frames]/1000, dims='frame', coords={}, attrs={'units': 's'})

            self.stage_coordinates_in_pixels = self.stage_coordinates / self.pixel_size


    def _read_frame(self, frame_number):
        """Read a single frame from the ND2 file.

        Parameters
        ----------
        frame_number : int
            Index of frame to read (relative to current FOV if multi-FOV)

        Returns
        -------
        np.ndarray
            Single frame image array

        Notes
        -----
        - Applies frame_offset for multi-FOV files
        - Automatically handles out-of-range requests by returning last frame
        """
        with self:
            if frame_number > self.number_of_frames:
                frame_index = self.number_of_frames - 1
                print(f'Frame number out of range. The last frame (fr#{frame_index}) is loaded instead')
            image = self.file.read_frame(self.fov_frames[frame_number])
            return image

    def _read_frames(self, indices):
        """Read multiple frames from the ND2 file.

        Parameters
        ----------
        indices : list or np.ndarray
            Indices of frames to read

        Returns
        -------
        np.ndarray
            Requested frames stacked along first dimension

        Notes
        -----
        - Currently implemented by calling _read_frame iteratively
        - Could be optimized for better performance with large frame batches
        """

        with self:
            frames = np.stack([self._read_frame(i) for i in indices])
            # frames = self.file.to_dask()[self.fov_frames[indices]]
            # frames = frames.compute()

        return frames

#
# def get_fov_from_nd2(nd2_fullpath):
#     images = ND2Reader(str(nd2_fullpath))
#     y_positions = images._parser._raw_metadata.y_data  # nikon sample stage position
#     x_positions = images._parser._raw_metadata.x_data  # nikon sample stage position
#
#     # set the image data order in the nd2 file
#     if 'c' in images.axes:
#         images.iter_axes = 'tc'  # for alex measurements
#     else:
#         images.iter_axes = 't'
#
#     n_illumination = len(images.metadata["channels"])
#     n_frames = len(x_positions)
#     position_tolerance = 10  # xy tol = tolerance in um
#     first_frame_of_each_fov = [0]
#     last_frame_of_each_fov = []
#     for fri in range(n_frames - 1):
#         if abs(x_positions[fri] - x_positions[fri + 1]) > position_tolerance or abs(
#                 y_positions[fri] - y_positions[fri + 1]) > position_tolerance:
#             first_frame_of_each_fov.append(fri + 1)
#             last_frame_of_each_fov.append(fri)
#     last_frame_of_each_fov.append(n_frames - 1)
#     fov_info = {'number_of_fov': len(first_frame_of_each_fov),
#                 'first_frame_of_each_fov': first_frame_of_each_fov,
#                 'last_frame_of_each_fov': last_frame_of_each_fov}
#     return fov_info

if __name__ == "__main__":
    print('test')
