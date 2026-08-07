"""TIFF movie reader (TIF/TIFF) and simple helpers.

Provides a TifMovie subclass that uses tifffile to read multi-page TIFF stacks and metadata.
"""

import numpy as np
import pandas as pd
import xarray as xr

from papylio.movie.movie import Movie
from papylio.timer import Timer


class TifMovie(Movie):
    """Movie class for reading TIFF/TIF image files.

    Handles loading and processing of TIFF format images, supporting both
    single-file and multi-frame stacks. Extracts metadata from TIFF tags.
    """
    extensions = ['.tif', '.tiff']

    def __init__(self, arg, *args, **kwargs):
        """Initialize TifMovie instance.

        Parameters
        ----------
        arg : str or Path
            Path to the TIFF file
        *args
            Additional positional arguments passed to Movie parent class
        **kwargs
            Additional keyword arguments passed to Movie parent class
        """
        super().__init__(arg, *args, **kwargs)


        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name

        # self.threshold = {  'view':             (0,200),
        #                     'point-selection':  (45,25)
        #                     }

        self.file = None # Note this is for the tif file, not the File class.
        self._time = None

        # self.read_metadata()
        # self.create_frame_info()  # Possibly move to Movie later on

        # self._initialized = True

    def open(self):
        """Open TIFF file for reading.

        Creates a tifffile.TiffFile object to access file contents.
        """
        import tifffile
        self.file = tifffile.TiffFile(self.filepath)

    def close(self):
        """Close the TIFF file."""
        self.file.close()

    def _read_metadata(self):
        """Read and parse TIFF file header and metadata.

        Extracts image dimensions, data type, number of frames, and
        calibration/positioning information from TIFF tags and metadata.
        """
        with self:
            tif_tags = {}
            for tag in self.file.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
            self.width = tif_tags['ImageWidth']
            self.height = tif_tags['ImageLength']
            self.data_type = np.dtype(f"uint{tif_tags['BitsPerSample']}")

            self.number_of_frames = len(self.file.pages) #TODO: test whether this is slow to always execute (as this used to be only done in case no other metadata was found)
            self.pixel_size = np.array([1, 1])  # For TIR-T because it is not in metadata
            self.pixel_size_unit = ''  # For TIR-T because it is not in metadata
            # Or self.file.series[0].shape[0]

            # if 'DateTime' in self.file.pages[0].tags:
            #     # For which files is this?
            #     # Note that looping over tif pages will be slow.
            #     self.datetime = pd.to_datetime([page.tags['DateTime'].value for page in self.file.pages])
            #     self.time = xr.DataArray((self.datetime-self.datetime[0]).total_seconds(), dims='frame',
            #                              coords={}, attrs={'units': 's'})
            # elif 'DateTime' in self.file.pages[0].tags:
            #     # This is for tiff images from TIR-T and TIR-V measured by Solis software
            #     exposure_time = self.file.pages[0].tags['AndorExposureTime'].value # NOTE: "kinetic time (or cycle time)" is more accurate measure. But the difference is very small and often ignored.
            #     time_vector = exposure_time * np.arange(0, self.number_of_frames)
            #     self.time = xr.DataArray(time_vector, dims='frame', coords={}, attrs={'units': 's'})
    #
    @property
    def time(self):
        """Temporal information for each frame.

        Returns
        -------
        xr.DataArray
            Time coordinate for each frame, computed from either DateTime tags
            or exposure time. Dimension: (frame,). Units: seconds from start.

        Notes
        -----
        - For Andor/Solis TIFF files: computed from AndorExposureTime tags
        - For other files with DateTime: parsed from DateTime TIFF tags
        - Time is cached after first access
        """
        # TODO: Get time when in read_frame
        if self._time is None:
            with self:
                # print('Time is obtained')
                if 'DateTime' in self.file.pages[0].tags:
                    # For which files is this?
                    # Note that looping over tif pages will be slow.
                    self.datetime = pd.to_datetime([page.tags['DateTime'].value for page in self.file.pages])
                    self._time = xr.DataArray((self.datetime-self.datetime[0]).total_seconds(), dims='frame',
                                             coords={}, attrs={'units': 's'})
                elif 'AndorExposureTime' in self.file.pages[0].tags:
                    # This is for tiff images from TIR-T and TIR-V measured by Solis software
                    exposure_time = self.file.pages[0].tags['AndorExposureTime'].value # NOTE: "kinetic time (or cycle time)" is more accurate measure. But the difference is very small and often ignored.
                    time_vector = exposure_time * np.arange(0, self.number_of_frames)
                    self._time = xr.DataArray(time_vector, dims='frame', coords={}, attrs={'units': 's'})

                # self.create_frame_info()  # Possibly move to Movie later on
        return self._time

    def _read_frame(self, frame_number):
        """Read a single frame from the TIFF file.

        Parameters
        ----------
        frame_number : int
            Index of frame to read (0-based)

        Returns
        -------
        np.ndarray
            Single frame image array

        Raises
        ------
        IndexError
            If frame_number is >= number_of_frames
        """
        with self:
            # if self.number_of_frames == 1:
            #     # return -1,0,0,0
            #     im = tifpage[0].asarray()
            if (self.number_of_frames - 1) >= frame_number:
                im = self.file.pages[frame_number].asarray().astype(self.data_type)
            else:
                raise IndexError('Selected frame number larger than number of frames')
            #     im = tifpage[self.number_of_frames - 1].asarray()
            #     print('pageNb out of range, printed image {0} instead'.format(self.number_of_frames))
        return im

    def _read_frames(self, frame_indices=None):
        """Read multiple frames from the TIFF file.

        Parameters
        ----------
        frame_indices : list or np.ndarray
            Indices of frames to read

        Returns
        -------
        np.ndarray or dask.array
            Requested frames stacked along first dimension.
            Returns numpy array if use_dask=False, dask array otherwise.

        Notes
        -----
        - For single frames, uses tifffile directly
        - For multiple frames, uses tifffile.imread with key parameter
        - Can optionally use dask_image for lazy loading if use_dask=True
        """
        if not self.use_dask:
            if len(frame_indices) == 1:
                with self:
                    frames = np.stack([self.file.pages[i].asarray() for i in frame_indices])
            else:
                import tifffile
                frames = tifffile.imread(self.filepath, key=frame_indices)

        else:
            import dask_image.imread
            frames = dask_image.imread.imread(self.filepath, nframes=100)[frame_indices]
        return frames

if __name__ == "__main__":
    movie = TifMovie(r'.\Example_data\tif\movie.tif')

#
# import os
# from glob import glob
#
# try:
#     from skimage.io import imread as sk_imread
# except (AttributeError, ImportError):
#     pass
#
# from dask.base import tokenize
# from dask.array.core import Array
#
#
# def add_leading_dimension(x):
#     return x[None, ...]
#
#
# def imread(filename, imread=None, preprocess=None):
#     """Read a stack of images into a dask array
#
#     Parameters
#     ----------
#
#     filename: string
#         A globstring like 'myfile.*.png'
#     imread: function (optional)
#         Optionally provide custom imread function.
#         Function should expect a filename and produce a numpy array.
#         Defaults to ``skimage.io.imread``.
#     preprocess: function (optional)
#         Optionally provide custom function to preprocess the image.
#         Function should expect a numpy array for a single image.
#
#     Examples
#     --------
#
#     >>> from dask.array.image import imread
#     >>> im = imread('2015-*-*.png')  # doctest: +SKIP
#     >>> im.shape  # doctest: +SKIP
#     (365, 1000, 1000, 3)
#
#     Returns
#     -------
#
#     Dask array of all images stacked along the first dimension.  All images
#     will be treated as individual chunks
#     """
#     file = tifffile.TiffFile(filename)
#
#     imread = imread or sk_imread
#     filenames = sorted(glob(filename))
#     if not filenames:
#         raise ValueError("No files found under name %s" % filename)
#
#     name = "imread-%s" % tokenize(file.pages, map(os.path.getmtime, file.pages))
#
#     sample = file.pages[0].asarray()
#     if preprocess:
#         sample = preprocess(sample)
#
#     keys = [(name, i) + (0,) * len(sample.shape) for i in range(len(file.pages))]
#     if preprocess:
#         values = [
#             (add_leading_dimension, (tifffile.TiffPage.asarray, page)) for page in file.pages
#         ]
#     else:
#         values = [(add_leading_dimension, (tifffile.TiffPage.asarray, page)) for page in file.pages]
#     dsk = dict(zip(keys, values))
#
#     chunks = ((1,) * len(file.pages),) + tuple((d,) for d in sample.shape)
#
#     return Array(dsk, name, chunks, sample.dtype)