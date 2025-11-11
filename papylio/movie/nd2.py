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
from nd2reader import ND2Reader

from papylio.movie.movie import Movie, Illumination


class ND2Movie(Movie):
    extensions = ['.nd2']

    def __init__(self, arg, *args, **kwargs):
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
        # SHK: self.rot90 should be set before reading the header.
        # self.rot90 = 1

        # self.read_header()

        # self.time = self.time[self.fov_info['first_frame_of_each_fov'][self.fov_info['fov_chosen']]:(self.fov_info['last_frame_of_each_fov'][self.fov_info['fov_chosen']]+1)]
        # self.illumination = self.illumination[self.fov_info['first_frame_of_each_fov'][self.fov_info['fov_chosen']]:(self.fov_info['last_frame_of_each_fov'][self.fov_info['fov_chosen']]+1)]
        # self.create_frame_info()  # Possibly move to Movie later on

        # self._initialized = True

        self.file = None # Note this is for the tif file, not the File class.

    def open(self):
        self.file = ND2Reader(str(self.filepath))
        if 'c' in self.file.axes:
            self.file.iter_axes = 'tc'
        else:
            self.file.iter_axes = 't'

    def close(self):
        self.file.close()

    def _read_header(self):
        with self:
            y_positions = self.file._parser._raw_metadata.y_data  # nikon sample stage position
            x_positions = self.file._parser._raw_metadata.x_data  # nikon sample stage position

            # TODO: @Sung Hyun, I assume it doe snot matter for y and x-positions if self.file.iter_axis is called first, right?
            # Then we can remove the commented lines below
            # # set the image data order in the nd2 file
            # if 'c' in self.file.axes:
            #     self.file.iter_axes = 'tc'  # for alex measurements
            # else:
            #     self.file.iter_axes = 't'

            n_illumination = len(self.file.metadata["channels"])
            n_frames = len(x_positions)
            position_tolerance = 10  # xy tol = tolerance in um
            first_frame_of_each_fov = [0]
            last_frame_of_each_fov = []
            for fri in range(n_frames - 1):
                if abs(x_positions[fri] - x_positions[fri + 1]) > position_tolerance or abs(
                        y_positions[fri] - y_positions[fri + 1]) > position_tolerance:
                    first_frame_of_each_fov.append(fri + 1)
                    last_frame_of_each_fov.append(fri)
            last_frame_of_each_fov.append(n_frames - 1)

            self.number_of_fov = len(first_frame_of_each_fov)
            self.first_frame_of_each_fov = first_frame_of_each_fov
            self.last_frame_of_each_fov = last_frame_of_each_fov


            self.width = self.file.metadata['width']
            self.height = self.file.metadata['height']
            self.pixel_size = np.array([self.file.metadata['pixel_microns'], self.file.metadata['pixel_microns']])

            # self.number_of_fields_of_view = len(images.metadata["experiment"]["loops"])  # number of fov is now available from self.fov_info
            self.number_of_frames = len(self.file)

            self.illuminations = [Illumination(name) for name in self.file.metadata["channels"]]

            if self.fov_index is not None:
                self.frame_offset = self.first_frame_of_each_fov[self.fov_index] * self.number_of_illuminations
                frame_end = (self.last_frame_of_each_fov[self.fov_index]+1) * self.number_of_illuminations
                self.number_of_frames = frame_end - self.frame_offset
            else:
                self.frame_offset = 0
                frame_end = self.number_of_frames

                self.stage_coordinates = np.array([[x_positions[0], y_positions[0]]])
                self.stage_coordinates_in_pixels = self.stage_coordinates / self.pixel_size


            self.illumination_arrangement = np.arange(len(self.illuminations))

            if self.fov_index is not None:
                self.time = xr.DataArray(np.repeat(
                    self.file.timesteps[self.first_frame_of_each_fov[self.fov_index]:(self.last_frame_of_each_fov[self.fov_index]+1)],
                    self.number_of_illuminations)/1000, dims='frame', coords={}, attrs={'units': 's'})
            else:
                self.time = xr.DataArray(np.repeat(self.file.timesteps, self.number_of_illuminations)/1000, dims='frame',
                                         coords={}, attrs={'units': 's'})

            # self.exp_time = images.metadata['experiment']['loops'][0]['sampling_interval']
            # self.exp_time_start=images.metadata['experiment']['loops'][0]['start']
            # self.exp_time_duration=images.metadata['experiment']['loops'][0]['duration']
            # self.pixelmicron=images.metadata['experiment']['pixel_microns']



    def _read_frame(self, frame_number):
        with self:
            if frame_number < self.number_of_frames:
                im = self.file[frame_number + self.frame_offset]
            else:
                im = self.file[self.number_of_frames + self.frame_offset - 1]
                print(f'pageNb out of range. The last frame (fr#{self.number_of_frames + self.frame_offset - 1}) is loaded instead')
            # note: im is a Frame, which is pims.frame.Frame, a np. array with additional frame number and metadata
            return im

    def _read_frames(self, indices):
        # Can probably be implemented more efficiently
        return np.stack([self._read_frame(i) for i in indices])

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
