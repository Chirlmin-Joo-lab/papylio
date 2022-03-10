import numpy as np
import pandas as pd
import xarray as xr
import tifffile

from trace_analysis.movie.movie import Movie
from trace_analysis.timer import Timer


class TifMovie(Movie):
    extensions = ['.tif', '.tiff']

    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs)
        

        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name
        
        self.threshold = {  'view':             (0,200),
                            'point-selection':  (45,25)
                            }
        self.file = None # Note this is for the tif file, not the File class.
        # self.read_header()
        # self.create_frame_info()  # Possibly move to Movie later on

        self._initialized = True

    def open(self):
        self.file = tifffile.TiffFile(self.filepath)

    def close(self):
        self.file.close()

    def _read_header(self):
        with self:
            tif_tags = {}
            for tag in self.file.pages[0].tags.values():
                name, value = tag.name, tag.value
                tif_tags[name] = value
            self.width = tif_tags['ImageWidth']
            self.height = tif_tags['ImageLength']
            # self.number_of_frames = len(tif.pages) # very slow
            self.data_type = np.dtype(f"uint{tif_tags['BitsPerSample']}")

            #self.datetime = pd.to_datetime([page.tags['DateTime'].value for page in tif.pages])
            #self.time = xr.DataArray((self.datetime-self.datetime[0]).total_seconds(), dims='frame',
            #                         coords={}, attrs={'units': 's'})
            try:
                if self.file.metaseries_metadata:
                    self.number_of_frames = self.file.metaseries_metadata['SetInfo']['number-of-planes']
                    #TODO: Make sure this goes well when movie is rotated
                    pixel_size_x = self.file.metaseries_metadata['PlaneInfo']['spatial-calibration-x']
                    pixel_size_y = self.file.metaseries_metadata['PlaneInfo']['spatial-calibration-y']
                    self.pixel_size = np.array([pixel_size_x, pixel_size_y])
                    self.pixel_size_unit = self.file.metaseries_metadata['PlaneInfo']['spatial-calibration-units'].replace('um', 'µm')
                    #TODO: Make sure this goes well when movie is rotated
                    stage_position_x = self.file.metaseries_metadata['PlaneInfo']['stage-position-x']
                    stage_position_y = self.file.metaseries_metadata['PlaneInfo']['stage-position-y']
                    self.stage_coordinates = np.array([[stage_position_x, stage_position_y]])
                    self.stage_coordinates_in_pixels = self.stage_coordinates / self.pixel_size

            except AttributeError:
                pass

            self.create_frame_info()  # Possibly move to Movie later on

    def _read_frame(self, frame_number):
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

    def _read_frames(self, indices=None):
        if indices is None:
            indices = np.arange(self.number_of_frames)
        if not self.use_dask:
            with self:
                frames = np.stack([self.file.pages[i].asarray() for i in indices])
        else:
            import dask_image.imread
            frames = dask_image.imread.imread(self.filepath, nframes=100)[indices]
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