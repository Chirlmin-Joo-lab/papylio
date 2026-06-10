from papylio.movie.tif import TifMovie
import numpy as np

class BNTIRFMovie(TifMovie):
    # date =

    microscope = "BN-TIRF"

    extensions = ['.tif', '.tiff']

    rotation = 1

    channels = ['green', 'red']
    channel_arrangement = np.array([[[0, 1]]])

    illuminations = ['green', 'red']
    default_illumination = 0

    psf_size = 1.01

    def _read_header(self):
        with self:
            super()._read_header()

            if self.file.metaseries_metadata:
                self.number_of_frames = self.file.metaseries_metadata['SetInfo']['number-of-planes']
                # TODO: Make sure this goes well when movie is rotated
                pixel_size_x = self.file.metaseries_metadata['PlaneInfo']['spatial-calibration-x']
                pixel_size_y = self.file.metaseries_metadata['PlaneInfo']['spatial-calibration-y']
                if 'Ti2 Optical Zoom' in self.file.metaseries_metadata['PlaneInfo'].keys():
                    microscope_optical_zoom = float(
                        self.file.metaseries_metadata['PlaneInfo']['Ti2 Optical Zoom'][:-1])
                    pixel_size_x /= microscope_optical_zoom
                    pixel_size_y /= microscope_optical_zoom
                self.pixel_size = np.array([pixel_size_x, pixel_size_y])
                self.pixel_size_unit = self.file.metaseries_metadata['PlaneInfo'][
                    'spatial-calibration-units'].replace('um', 'µm')
                # TODO: Make sure this goes well when movie is rotated
                stage_position_x = self.file.metaseries_metadata['PlaneInfo']['stage-position-x']
                stage_position_y = self.file.metaseries_metadata['PlaneInfo']['stage-position-y']
                self.stage_coordinates = np.array([[stage_position_x, stage_position_y]])
                self.stage_coordinates_in_pixels = self.stage_coordinates / self.pixel_size

            elif self.file.imagej_metadata:
                ## Extracting metadata from tiff files acquired via InScoper
                def extract_parameter(metadata_str, parameter_name):
                    """
                    Extract a specific parameter value from metadata string.
                    Returns None if parameter not found.
                    """
                    for line in metadata_str.strip().split('\n'):
                        if line.startswith(parameter_name + ' '):
                            return line[len(parameter_name) + 1:]  # +1 for the space
                    return None

                def get_numeric_parameter(metadata_str, parameter_name):
                    """Extract parameter and convert to appropriate numeric type."""
                    value = extract_parameter(metadata_str, parameter_name)
                    if value is None:
                        return None

                    try:
                        # Try integer first
                        if '.' not in value and 'E' not in value.upper():
                            return int(value)
                        else:
                            return float(value)
                    except ValueError:
                        return value  # Return as string if conversion fails

                stage_position_x = get_numeric_parameter(self.file.imagej_metadata['Info'],
                                                         'NikonTi2-xAxisPosition') / 1000
                stage_position_y = get_numeric_parameter(self.file.imagej_metadata['Info'],
                                                         'NikonTi2-yAxisPosition') / 1000
                pixel_size = get_numeric_parameter(self.file.imagej_metadata['Info'], 'PixelSizeUm')

                self.pixel_size = np.array([pixel_size, pixel_size])
                self.pixel_size_unit = 'µm'
                self.number_of_frames = len(self.file.pages)

                self.stage_coordinates = np.array([[stage_position_x, stage_position_y]])
                self.stage_coordinates_in_pixels = self.stage_coordinates / self.pixel_size


    def correct_microscope(self):
        with self:
            return self.file.metaseries_metadata is not None or self.file.imagej_metadata is not None