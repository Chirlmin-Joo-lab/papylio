from papylio.movie.movie import Movie


def test_image_info_from_filename():
    # movie = Movie('Abc.tif')
    # movie.number_of_frames = 100
    filename = 'Abc_fov005_ave_i0_f10-50-2.tif'
    image_info = Movie.image_info_from_filename(filename)
    image_info_expected = {'fov_index': 5, 'image_type': 'average', 'frame_range': (10, 50, 2),
                           'illumination_index': 0}
    assert image_info == image_info_expected


def test_image_info_to_filename():
    # movie = Movie('Abc.tif')
    # movie.number_of_frames = 100
    # movie.illumination_arrangement = [0, 1]
    image_info = {'fov_index': 5, 'image_type': 'average', 'frame_range': (10, 50, 2),
                  'illumination_index': 0}
    filename = Movie.image_info_to_filename(**image_info)
    filename_expected = 'Abc_ave_fov005_f10-50-2_i0'
    assert filename == filename_expected
