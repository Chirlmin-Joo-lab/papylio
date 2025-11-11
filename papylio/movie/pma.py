import os
import numpy as np

from papylio.movie.movie import Movie


class PmaMovie(Movie):
    extensions = ['.pma']

    def __init__(self, arg, *args, **kwargs):
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
        pass  # TODO: implement this

    def close(self):
        pass  # TODO: implement this

    def _read_header(self):
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
        # Can probably be implemented more efficiently
        return np.stack([self._read_frame(i) for i in indices])

if __name__ == "__main__":
    movie = PmaMovie(r'.\Example_data\pma\movie.pma')
    # movie.intensity_range = (0, 120)
    # movie.make_projection_images()
