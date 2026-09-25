import os
import pytest
import numpy as np
import shutil, tempfile
from pathlib import Path

# def create_full_experiment(src: Path, files) -> Path:
#     shutil.copytree(src, dst, dirs_exist_ok=True)

@pytest.fixture
def file():
    filepath = Path(r'ssHJ1/ssHJ1_1.tif')
    source = Path(__file__).parent / 'data' / 'Example dataset' / filepath

    with tempfile.TemporaryDirectory(prefix='papylio_test_') as temporary_directory:
        destination = Path(temporary_directory) / filepath.name
        print(temporary_directory)
        shutil.copy(source, destination)

        from papylio import Experiment
        exp = Experiment(destination.parent)
        file = exp.files[0]
        file.illumination_arrangement = [0]

        try:
            yield file
        finally:
            os.chdir('..') # To remove the temporary directory, we need to change the working directory away from it.

@pytest.mark.parametrize(
    "image_configuration, result", [
    (dict(frames=0),(1, 2, 512, 256)),
    (dict(frames=[0]),(1, 2, 512, 256)),
    (dict(frames=slice(0,20)),(20, 2, 512, 256)),
    (dict(frames=range(10,30)),(20, 2, 512, 256)),
    (dict(frames=range(10,30,2)),(10, 2, 512, 256)),
    (dict(frames=[10, 20, 30]),(3, 2, 512, 256)),
    (dict(frames=np.array([10, 20, 30])),(3, 2, 512, 256)),
    (dict(frames=slice(0,20), channel='green'),(20, 1, 512, 256)),
    (dict(frames=slice(0,20), channel='red'),(20, 1, 512, 256)),
    (dict(frames=slice(0,20), projection='average'),(1, 2, 512, 256)),
    (dict(frames=slice(0,20), projection='maximum'),(1, 2, 512, 256)),
    (dict(frames=slice(0,20), overlay_channels=True),(20, 1, 512, 256)),
    (dict(frames=slice(0,20), apply_corrections=False),(20, 2, 512, 256)),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True),(1, 1, 512, 256)),
    (dict(frames=slice(0,20), projection='maximum', overlay_channels=True),(1, 1, 512, 256)),
])
def test_get_image(file, image_configuration, result):
    image = file.get_image(**image_configuration)
    assert image.shape == result

def test_get_image_save_and_load(file):
    image_saved = file.get_image(frames=slice(0,20), projection='average')
    image_loaded = file.get_image(frames=slice(0, 20), projection='average')
    assert (image_saved == image_loaded).all().item()

@pytest.mark.parametrize(
    "image_configuration", [
        dict(frames=slice(0,20), illumination=0),
        dict(frames=slice(0,20), illumination=1)
])
def test_get_image_multiple_illuminations(file, image_configuration):
    file.illumination_arrangement = [0] * 10 + [1] * 390
    image = file.get_image(**image_configuration)
    assert image.shape == (10, 2, 512, 256)

@pytest.mark.parametrize(
    "image_configuration, imshow_configuration", [
    (dict(frames=slice(0,20)), None),
    (dict(frames=slice(0,20), channel='red'),None),
    (dict(frames=slice(0,20), projection='average'), None),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True), None),
    (dict(frames=slice(0,20), projection='average', overlay_channels=True), dict(vmin=0, vmax=100)),
])
def test_show_image(file, image_configuration, imshow_configuration):
    file.show_image(imshow_configuration=imshow_configuration, **image_configuration)
