import pytest
import numpy as np
from pathlib import Path
from tests.integration.temporary_data import temporary_data

@pytest.fixture
def file():
    relative_path = Path(r'Example dataset/ssHJ1/ssHJ1_1.tif')

    with temporary_data(relative_path) as path:
        from papylio import Experiment
        exp = Experiment(path.parent, microscope='BN-TIRF')
        file = exp.files[0]
        file.illumination_arrangement = [0]

        yield file

@pytest.fixture
def mapping_file():
    relative_path = Path(r'Example dataset/Bead slide/Bead slide.tif')

    with temporary_data(relative_path) as path:
        from papylio import Experiment
        exp = Experiment(path.parent, microscope='BN-TIRF')
        file = exp.files[0]
        file.illumination_arrangement = [0]

        yield file

def test_perform_mapping(mapping_file):
    mapping_file.perform_mapping()

def test_find_coordinates(file):
    file.find_coordinates(image_configuration=dict(frames=slice(0,20), projection='average'))
    assert len(file.coordinates) > 0

def test_extract_traces(file):
    file.find_coordinates(image_configuration=dict(frames=slice(0,20), projection='average'))
    file.extract_traces()
