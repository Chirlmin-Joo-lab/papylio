import pytest
import numpy as np
from papylio.movie.movie import Movie

import pytest
import numpy as np
from pathlib import Path
from tests.integration.temporary_data import temporary_data

@pytest.fixture
def movie():
    relative_path = Path(r'Example dataset/ssHJ1/ssHJ1_1.tif')

    with temporary_data(relative_path) as path:
        from papylio.movie.movie import Movie
        movie = Movie(path)#, microscope='BN-TIRF')
        movie.rotation = 1
        yield movie

@pytest.fixture
def movie_one_channel(movie):
    movie.channels = ['green']
    movie.channel_arrangement = [[[0]]]
    movie.illuminations = ['green', 'red']
    movie.illumination_arrangement = [0]
    return movie

@pytest.fixture
def movie_two_channels(movie):
    movie.channels = ['green', 'red']
    movie.channel_arrangement = [[[0,
                                   1]]]  # For splitting in the horizontal dimension, with the green and red channels left and right, respectively.
    movie.illuminations = ['green', 'red']
    movie.illumination_arrangement = [0]
    return movie


@pytest.fixture
def movie_multiple_illuminations(movie):
    movie.channels = ['green', 'red']
    movie.channel_arrangement = [[[0,
                                   1]]]  # For splitting in the horizontal dimension, with the green and red channels left and right, respectively.
    movie.illuminations = ['green', 'red']
    movie.illumination_arrangement = [0] * 10 + [1] * 390
    return movie

def test_movie_name(movie):
    assert movie.name == 'ssHJ1_1'

@pytest.mark.parametrize(
    "image_configuration, result", [
    (dict(frames=0),(1, 2, 512, 256)),
    (dict(frames=[0]),(1, 2, 512, 256)),
    (dict(frames=slice(0,20)),(20, 2, 512, 256)),
    (dict(frames=range(10,30)),(20, 2, 512, 256)),
    (dict(frames=range(10,30,2)),(10, 2, 512, 256)),
    (dict(frames=[10, 20, 30]),(3, 2, 512, 256)),
    (dict(frames=np.array([10, 20, 30])),(3, 2, 512, 256)),
    (dict(frames=slice(0,20), channels='green'),(20, 1, 512, 256)),
    (dict(frames=slice(0,20), channels='red'),(20, 1, 512, 256)),
    (dict(frames=slice(0,20), projection='average'),(1, 2, 512, 256)),
    (dict(frames=slice(0,20), projection='maximum'),(1, 2, 512, 256)),
    (dict(frames=slice(0,20), overlay_channels=True),(20, 1, 512, 256)),
    (dict(frames=slice(0,20), apply_corrections=False),(20, 2, 512, 256)),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True),(1, 1, 512, 256)),
    (dict(frames=slice(0,20), projection='maximum', overlay_channels=True),(1, 1, 512, 256)),
])
def test_get_image(movie_two_channels, image_configuration, result):
    image = movie_two_channels.get_image(**image_configuration)
    assert image.shape == result

@pytest.mark.parametrize(
    "image_configuration, result", [
    (dict(frames=slice(0,20)),(20, 1, 512, 512)),
    (dict(frames=slice(0,20), channels='green'),(20, 1, 512, 512)),
    (dict(frames=slice(0,20), projection='average'),(1, 1, 512, 512)),
    (dict(frames=slice(0,20), overlay_channels=True),(20, 1, 512, 512)),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True),(1, 1, 512, 512)),
])
def test_get_image(movie_one_channel, image_configuration, result):
    image = movie_one_channel.get_image(**image_configuration)
    assert image.shape == result

@pytest.mark.parametrize(
    "image_configuration", [
        dict(frames=slice(0,20), projection='average'),
        dict(frames=slice(0,20), projection='average', load=True),
        dict(frames=slice(0,20), projection='average', overlay_channels=True),
])
def test_get_image_save_and_load(movie_two_channels, image_configuration):
    image_saved = movie_two_channels.get_image(**image_configuration)
    image_loaded = movie_two_channels.get_image(**image_configuration)
    assert (image_saved == image_loaded).all().item()

@pytest.mark.parametrize(
    "image_configuration", [
        dict(frames=slice(0,20), illuminations=0),
        dict(frames=slice(0,20), illuminations=1),
        dict(frames=slice(0,20), illuminations='green'),
        dict(frames=slice(0,20), illuminations='red'),
])
def test_get_image_multiple_illuminations(movie_multiple_illuminations, image_configuration):
    image = movie_multiple_illuminations.get_image(only_save_projections=False, **image_configuration)
    assert image.shape == (10, 2, 512, 256)

def test_get_image_any_kwarg(movie_two_channels):
    movie_two_channels.get_image(frames=range(0,20), projection='average')
    with pytest.raises(TypeError):
        movie_two_channels.get_image(frames=range(0,20), projection='average', test=True)

@pytest.mark.parametrize(
    "image_configuration, imshow_configuration", [
    (dict(frames=slice(0,20)), None),
    (dict(frames=slice(0,20), channels='red'),None),
    (dict(frames=slice(0,20), projection='average'), None),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True), None),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True), dict(vmin=0, vmax=100)),
])
def test_show_image(movie_two_channels, image_configuration, imshow_configuration):
    movie_two_channels.show_image(imshow_configuration=imshow_configuration, **image_configuration)

def test_show_image_custom_figures_and_axes(movie_two_channels):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2,1)
    image_configuration = dict(frames=slice(0,20))
    movie_two_channels.show_image(axes=axes, **image_configuration)

def test_clear_images(movie_two_channels):
    movie_two_channels.get_image(frames=slice(0,20), projection='average')
    assert movie_two_channels.images_directory.exists()
    assert len(list(movie_two_channels.images_directory.iterdir())) == 1

    movie_two_channels.clear_images()
    assert not movie_two_channels.images_directory.exists()

def test_spatial_background_correction(movie_two_channels):
    movie_two_channels.determine_spatial_background_correction()

def test_temporal_background_correction(movie_two_channels):
    movie_two_channels.determine_temporal_background_correction()

def test_general_background_correction(movie_two_channels):
    movie_two_channels.determine_general_background_correction()














#
# def test_make_projection_image_frame_range_out_of_bounds(movie, shared_datadir):
#     movie.make_projection_image(projection_type='average', frame_range=(390, 410), illumination=None, write=True,
#                                 return_image=True, flatten_channels=True)
#
# def test_determine_background_correction(experiment, shared_datadir):
#     movie = experiment.files[1].movie
#
#     movie.determine_general_background_correction()
#     movie.determine_temporal_background_correction()
#     movie.determine_spatial_background_correction(size=15)
#     assert (shared_datadir / 'BN_TIRF' / 'TIRF 561 0001_corrections.nc').is_file()
#
#     import xarray as xr
#     new_dataset = xr.load_dataset(shared_datadir / 'BN_TIRF' / 'TIRF 561 0001_corrections.nc')
#     new_dataset = new_dataset[['general_background_correction', 'temporal_background_correction', 'spatial_background_correction']]
#     test_dataset = xr.load_dataset(shared_datadir / 'BN_TIRF_output_test_movie' / 'TIRF 561 0001_corrections.nc')
#     assert new_dataset.identical(test_dataset)
#
#     movie.make_projection_image()
#
#
#
# import pytest
# import numpy as np
# from pathlib import Path
# from tests.integration.temporary_data import temporary_data
#
# @pytest.fixture
# def nd2_movie():
#     relative_path = Path(r'nd2/4 - 10 nt - 000.nd2')
#
#     with temporary_data(relative_path) as path:
#         from papylio.movie.movie import Movie
#         movie = Movie(path, microscope='TIR-S')
#         movie.dataset.set_attribute(illumination_arrangement = [1])
#         yield movie
#
# def test_nd2_fileformat(nd2_movie):
#     from papylio.movie.movie import Movie
#     # movie = Movie(shared_datadir / 'nd2' / 'Bead slide.nd2')
#     nd2_movie.make_image()