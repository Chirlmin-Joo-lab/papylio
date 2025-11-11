import numpy as np
import matplotlib.pyplot as plt
from papylio.movie.movie import Movie


class BinaryMovie(Movie):
    extensions = ['.bin']

    def __init__(self, arg, *args, **kwargs):
        super().__init__(arg, *args, **kwargs)
        
        self.writepath = self.filepath.parent
        self.name = self.filepath.with_suffix('').name
        
        #determine 8 bits or 16 bits
        # self.bitdepth = 16 if (self.filepath.name[-7:-4]=='_16') else 8

        self.threshold = {  'view':             (0,200),
                            'point-selection':  (45,25)
                            }


        self.width = 250
        self.height = 250
        self.number_of_frames = 200
        self.illumination_arrangement = np.array([1, 0])
        self.channel_arrangement = np.array([[[1]], [[0]]])

        # self.read_header()

        self.create_frame_info() # Possibly move to Movie later on

    def _read_header(self):
        pass

    def _read_frame(self, frame_number):
        # t = time.time()
        # if (frame_number < 0) or (frame_number >= self.number_of_frames):
        #     raise ValueError('Frame number out of range')
        #
        # start_byte = frame_number*self.bytes_per_frame
        #
        # with self.filepath.open('rb') as bin:
        #     bin.seek(start_byte)
        #     data = np.fromfile(bin, dtype=self.dtype, count=self.pixels_per_frame)
        #     image = data.reshape((self.width, self.height))
        #
        # return image

        return self.read_frames(frame_number, 1).squeeze(0)

    def read_frames(self, start_frame=0, number_of_frames=None):
        if number_of_frames is None:
            number_of_frames = self.number_of_frames

        for frame_number in (start_frame, start_frame+number_of_frames-1):
            if (frame_number < 0) or (frame_number >= self.number_of_frames):
                raise ValueError('Frame number out of range')

        start_byte = start_frame * self.bytes_per_frame

        with self.filepath.open('rb') as bin_file:
            bin_file.seek(start_byte)
            data = np.fromfile(bin_file, dtype=self.data_type, count=self.pixels_per_frame*number_of_frames)
            image = data.reshape((number_of_frames, self.width, self.height))

        return image

if __name__ == "__main__":
    movie = BinaryMovie(r'.\Example_data\binary\movie.bin')
    test = movie.read_frames(2,10)
    plt.imshow(test[0])
    movie.make_projection_images()


