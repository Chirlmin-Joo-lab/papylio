import pytest
import numpy as np
from pathlib import Path
from tests.integration.temporary_data import temporary_data

@pytest.fixture
def file():
    relative_path = Path(r'Example dataset/ssHJ1/ssHJ1_1.tif')

    with temporary_data(relative_path) as path:
        from papylio import Experiment
        exp = Experiment(path.parent, microscope=None)#, microscope='BN-TIRF')
        file = exp.files[0]
        file.rotation = 1
        file.channels = ['green', 'red']
        file.channel_arrangement = [[[0,1]]]
        file.illuminations = ['green', 'red']
        file.illumination_arrangement = [0]

        file.psf_size = 1.01

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

def test_show_traces(file):
    test_extract_traces(file)
    file.show_traces()

def test_create_selection(file):
    test_extract_traces(file)
    file.create_selection(variable='intensity_total', channel=None, aggregator='max', operator='<', threshold=10000)

def test_clear_selection(file):
    test_create_selection(file)
    file.clear_selections()

def test_apply_selections(file):
    test_extract_traces(file)
    file.create_selection(variable='intensity_total', channel=None, aggregator='max', operator='<', threshold=10000, name='selection_intensity_total_maximum')
    file.create_selection(variable='FRET', channel=None, aggregator='mean', operator='>', threshold=0.5, name='selection_FRET_mean')
    file.apply_selections()
    file.apply_selections(None)
    file.apply_selections('selection_intensity_total_maximum', 'selection_FRET_mean')
    file.apply_selections('selection_intensity_total_maximum', add_to_current=True)
    file.clear_selections()
    file.apply_selections()

def test_create_classification(file):
    test_extract_traces(file)
    selection = file.selected
    selection[20:] = False
    file.set_variable(selected=selection, overwrite=True)
    file.create_classification(name='classification_threshold', classification_type='threshold',
                                      variable='intensity_total', classification_configuration=dict(threshold=500, rolling='median', window_size=5))
    file.create_classification(name='classification_hmm', classification_type='hmm', variable='FRET')
    # file.create_classification(**json.loads(file_output.classification_test.attrs['configuration']))

def test_clear_classifications(file):
    test_create_classification(file)
    file.clear_classifications()

def test_apply_classifications(file):
    test_create_classification(file)
    file.apply_classifications(classification_threshold=-1, classification_hmm=[None, 0, 1])
    file.apply_classifications(add_to_current=True, classification_hmm=[None, 2, 3])

