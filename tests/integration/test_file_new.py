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
        exp = Experiment(path.parent, microscope=None) #microscope='BN-TIRF')
        file = exp.files[0]
        file.rotation = 1
        file.channels = ['green', 'red']
        file.channel_arrangement = [[[0,1]]]
        file.illuminations = ['green', 'red']
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

def test_apply_trace_corrections(file):
    test_extract_traces(file)
    file.apply_trace_corrections(background=(6700,1e4), alpha=0, gamma=1)

def test_show_traces(file):
    test_apply_trace_corrections(file)
    file.show_traces()

def test_create_selection(file):
    test_extract_traces(file)
    file.create_selection(variable='intensity_total', channel=None, aggregator='max', operator='<', threshold=10000)

def test_clear_selection(file):
    test_create_selection(file)
    file.clear_selections()

def test_apply_selections(file):
    test_apply_trace_corrections(file)
    file.create_selection(variable='intensity_total', channel=None, aggregator='max', operator='<', threshold=10000, name='selection_intensity_total_maximum')
    file.create_selection(variable='FRET', channel=None, aggregator='mean', operator='>', threshold=0.5, name='selection_FRET_mean')
    file.apply_selections()
    file.apply_selections(None)
    file.apply_selections('selection_intensity_total_maximum', 'selection_FRET_mean')
    file.apply_selections('selection_intensity_total_maximum', add_to_current=True)
    file.clear_selections()
    file.apply_selections()

def test_create_classification(file):
    test_apply_trace_corrections(file)
    selection = file.selected
    selection[20:] = False
    file.set_variables(selected=selection, overwrite=True)
    file.create_classification(name='classification_bleaching', classification_type='threshold',
                               variable='intensity_total', classification_configuration=dict(threshold=5000, rolling='median', window_size=5))
    file.create_classification(name='classification_threshold', classification_type='threshold',
                               variable='FRET', classification_configuration=dict(threshold=0.5))
    file.create_classification(name='classification_hmm', classification_type='hmm', variable='FRET')
    # file.create_classification(**json.loads(file_output.classification_test.attrs['configuration']))

def test_clear_classifications(file):
    test_create_classification(file)
    file.clear_classifications()

def test_apply_classifications(file):
    test_create_classification(file)
    file.apply_classifications(classification_threshold=-1, classification_hmm=[None, 0, 1])
    file.apply_classifications(add_to_current=True, classification_hmm=[None, 2, 3])

def test_determine_dwells_from_classification(file):
    test_create_classification(file)
    file.apply_classifications(classification_threshold=[0, 1], classification_bleaching=-1)
    file.determine_dwells_from_classification(selected=True)

def test_analyze_dwells(file):
    test_determine_dwells_from_classification(file)
    file.analyze_dwells(method='maximum_likelihood_estimation', number_of_exponentials=[1,2], state_names=None,
                        truncation=None, P_bounds=(-1, 1), k_bounds=(1e-9, np.inf), plot=False,
                        fit_dwell_times_configuration={}, plot_dwell_analysis_configuration={})

