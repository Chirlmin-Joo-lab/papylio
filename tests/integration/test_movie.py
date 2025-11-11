import pytest
import tifffile
import numpy as np
from papylio.movie.movie import Movie


@pytest.fixture
def movie(shared_datadir):
    movie = Movie(shared_datadir / 'BN_TIRF' / 'TIRF 561 0001.tif')
    movie.rot90 = 1
    return movie


@pytest.fixture
def experiment(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'BN_TIRF')


def test_movie_name(movie):
    assert movie.name == 'TIRF 561 0001'


def test_make_projection_image(movie, shared_datadir):
    image_from_method = movie.make_projection_image(projection_type='average', frame_range=(0, 20), illumination=None, write=True,
                                                    return_image=True, flatten_channels=True)
    assert (shared_datadir / 'BN_TIRF' / 'TIRF 561 0001_ave_f0-20_i0.tif').is_file()
    image_from_file = tifffile.imread(shared_datadir / 'BN_TIRF' / 'TIRF 561 0001_ave_f0-20_i0.tif')
    assert (image_from_file == image_from_method).all()
    image_from_original_file = tifffile.imread(shared_datadir / 'BN_TIRF_output_test_movie' / 'TIRF 561 0001_ave_f0-20_i0.tif')
    assert (image_from_original_file == image_from_method).all()
    raw_images = tifffile.imread(shared_datadir / 'BN_TIRF' / 'TIRF 561 0001.tif', key=range(0, 20))
    raw_images = np.rot90(raw_images, axes=(1,2))
    assert ((image_from_file - raw_images.mean(axis=0)) < 1e-4).all()  # Not sure 1e-4 is accurate enough


def test_make_projection_image_frame_range_out_of_bounds(movie, shared_datadir):
    movie.make_projection_image(projection_type='average', frame_range=(390, 410), illumination=None, write=True,
                                return_image=True, flatten_channels=True)

def test_make_projection_images(movie, shared_datadir):
    movie.make_projection_images(projection_type='average', frame_range=(0,20))


def test_determine_background_correction(experiment, shared_datadir):
    movie = experiment.files[1].movie

    movie.determine_general_background_correction()
    movie.determine_temporal_background_correction()
    movie.determine_spatial_background_correction(size=15)
    assert (shared_datadir / 'BN_TIRF' / 'TIRF 561 0001_corrections.nc').is_file()

    import xarray as xr
    new_dataset = xr.load_dataset(shared_datadir / 'BN_TIRF' / 'TIRF 561 0001_corrections.nc')
    new_dataset = new_dataset[['general_background_correction', 'temporal_background_correction', 'spatial_background_correction']]
    test_dataset = xr.load_dataset(shared_datadir / 'BN_TIRF_output_test_movie' / 'TIRF 561 0001_corrections.nc')
    assert new_dataset.identical(test_dataset)

    movie.make_projection_image()

def test_corrections(experiment):
    movie = experiment.files[1].movie
    movie.make_projection_image()