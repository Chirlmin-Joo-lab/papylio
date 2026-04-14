"""NSK (Andor) movie file reader utilities.

Reader for NSK/Andor file formats used by certain camera systems.

"""

import os
import numpy as np

from papylio.movie.movie import Movie


class NskMovie(Movie):
    """Movie class for reading NSK format image files.

    Handles loading and processing of NSK format images, which are raw
    binary files with a simple 4-byte header containing width and height.
    """
    extensions = ['.nsk']

    def __init__(self, arg, *args, **kwargs):
        """Initialize NskMovie instance.

        Parameters
        ----------
        arg : str or Path
            Path to the NSK file
        *args
            Additional positional arguments passed to Movie parent class
        **kwargs
            Additional keyword arguments passed to Movie parent class
        """
        super().__init__(arg, *args, **kwargs)

        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name
        self.rot90 = 1

        self.channel_arrangement = np.array([[[0, 1]]])

        self.data_type = np.dtype(np.uint16)

        # self.read_header()
        self.create_frame_info()  # Possibly move to Movie later on

        # self._initialized = True

    def _read_header(self):
        """Read and parse NSK file header.

        Extracts image dimensions (width and height as int16) from the first
        4 bytes of the file. Calculates number of frames from file size.
        """
        with self.filepath.open('rb') as fid:
            self.width = int(np.fromfile(fid, dtype=np.int16, count=1)[0])
            self.height = int(np.fromfile(fid, dtype=np.int16, count=1)[0])

        self.number_of_frames = int((os.path.getsize(self.filepath) - 4) / 2 / self.width / self.height)

    def _read_frame(self, frame_number):
        """Read a single frame from the NSK file.

        Parameters
        ----------
        frame_number : int
            Index of frame to read (0-based)

        Returns
        -------
        np.ndarray
            Single frame image array with shape (width, height)
        """
        with self.filepath.open('rb') as fid:
            fid.seek(4 + 2 * frame_number * int(self.width * self.height), os.SEEK_SET)
            image = np.fromfile(fid, dtype=np.uint16, count=self.width*self.height)
            image = np.reshape(image, (self.width, self.height))

        return image

    def _read_frames(self, indices):
        """Read multiple frames from the NSK file.

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
        - Could be optimized for better performance
        """
        # Can probably be implemented more efficiently
        return np.stack([self._read_frame(i) for i in indices])