from papylio.movie.tif import TifMovie
import numpy as np

class TIRTMovie(TifMovie):
    microscope = "TIR-T"

    extensions = ['.tif', '.tiff']

    rotation = -1

    channels = ['green', 'red']
    channel_arrangement = np.array([[[0, 1]]])

    illuminations = ['green', 'red']
    default_illumination = 0

    psf_size = 1.291

    def correct_microscope(self):
        with self:
            return not (self.file.metaseries_metadata is not None or self.file.imagej_metadata is not None)
