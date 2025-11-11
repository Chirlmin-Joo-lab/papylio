import os
import numpy as np

from papylio.movie.movie import Movie


class NskMovie(Movie):
    extensions = ['.nsk']

    def __init__(self, arg, *args, **kwargs):
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
        with self.filepath.open('rb') as fid:
            self.width = int(np.fromfile(fid, dtype=np.int16, count=1)[0])
            self.height = int(np.fromfile(fid, dtype=np.int16, count=1)[0])

        self.number_of_frames = int((os.path.getsize(self.filepath) - 4) / 2 / self.width / self.height)

    def _read_frame(self, frame_number):
        with self.filepath.open('rb') as fid:
            fid.seek(4 + 2 * frame_number * int(self.width * self.height), os.SEEK_SET)
            image = np.fromfile(fid, dtype=np.uint16, count=self.width*self.height)
            image = np.reshape(image, (self.width, self.height))

        return image

    def _read_frames(self, indices):
        # Can probably be implemented more efficiently
        return np.stack([self._read_frame(i) for i in indices])