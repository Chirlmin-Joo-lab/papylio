"""PMa file reader for Molecular Devices PMA-format movies.

Provides a lightweight interface to PMA files used by certain microscope acquisition software.
"""

import os
import numpy as np

from papylio.movie.movie import Movie


class PmaMovie(Movie):
    """Movie class for reading Photometrics PMA image files.

    Handles loading and processing of Photometrics PMA format images.
    Supports both 8-bit and 16-bit data based on filename convention.
    """
    extensions = ['.pma']

    def __init__(self, arg, *args, **kwargs):
        """Initialize PmaMovie instance.

        Parameters
        ----------
        arg : str or Path
            Path to the PMA file
        *args
            Additional positional arguments passed to Movie parent class
        **kwargs
            Additional keyword arguments passed to Movie parent class

        Notes
        -----
        - Detects bit depth from filename: '_16' suffix indicates 16-bit data
        - Default is 8-bit data
        """
        super().__init__(arg, *args, **kwargs)
        
        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name

        self.channel_arrangement = np.array([[[0,1]]])

        # Determine whether the image is 8 bits or 16 bits
        if (self.filepath.name[-7:-4]=='_16'):
            self.data_type = np.dtype(np.uint16)
        else:
            self.data_type = np.dtype(np.uint8)

        # Is this still used? [IS: 20-04-2021]
        self.threshold = {  'view':             (0,200),
                            'point-selection':  (45,25)
                            }

        # self.read_header()
        # self.create_frame_info()  # Possibly move to Movie later on



    def open(self):
        """Open PMA file for reading.

        Currently not implemented (placeholder for future optimization).
        """
        pass  # TODO: implement this

    def close(self):
        """Close the PMA file.

        Currently not implemented (placeholder for future optimization).
        """
        pass  # TODO: implement this

    def _read_header(self):
        """Read and parse PMA file header.

        Extracts image dimensions from the first 4 bytes of the file
        (2 int16 values: width and height). Calculates number of frames
        from file size.
        """
        statinfo = os.stat(self.filepath)
               
        with self.filepath.open('rb') as pma_file:
            self.width = np.fromfile(pma_file, np.int16, count=1)[0].astype(int)
            self.height = np.fromfile(pma_file, np.int16, count=1)[0].astype(int)
            self.number_of_frames = int((statinfo.st_size-4)/(self.width*self.height))

        # TODO: Import log file
        # self.exposure_time = np.genfromtxt(f'{self.absoluteFilePath}.log', max_rows=1)[2]
        # print(f'Exposure time set to {self.exposure_time} sec for {self.name}')
        # self.log_details = open(f'{self.absoluteFilePath}.log').readlines()
        # self.log_details = ''.join(self.log_details)

    def _read_frame(self, frame_number):
        """Read a single frame from the PMA file.

        Handles both 8-bit and 16-bit data. For 16-bit data, reads MSB and LSB
        separately and combines them.

        Parameters
        ----------
        frame_number : int
            Index of frame to read

        Returns
        -------
        np.ndarray
            Single frame image array with shape (width, height)
        """
        with self.filepath.open('rb') as pma_file:
            np.fromfile(pma_file, np.uint16, count=1)
            np.fromfile(pma_file, np.uint16, count=1)
        
            if self.bitdepth == 8:
                pma_file.seek(4+(frame_number*(self.width*self.height)), os.SEEK_SET)
                image = np.reshape(np.fromfile(pma_file, np.uint8, count=self.width*self.height), (self.width,self.height))
            else:
                pma_file.seek(4+2*frame_number*(self.width*self.height), os.SEEK_SET)
                msb = np.reshape(np.fromfile(pma_file, np.uint8, count=(self.width*self.height)), (self.width, self.height))
                lsb = np.reshape(np.fromfile(pma_file, np.uint8, count=(self.width*self.height)), (self.width, self.height))
                image = 256*msb+lsb

        return image

    def _read_frames(self, indices):
        """Read multiple frames from the PMA file.

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

if __name__ == "__main__":
    movie = PmaMovie(r'.\Example_data\pma\movie.pma')
    # movie.intensity_range = (0, 120)
    # movie.make_projection_images()
