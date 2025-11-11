import pytest
import tifffile
import numpy as np


@pytest.fixture
def experiment(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'BN_TIRF_sequencing')

@pytest.fixture
def file(experiment):
    return experiment.files[1]


def test_sequencing_dataset_property_in_file(file):
    sequencing_data = file.sequencing_data
    sequencing_data.dataset.attrs['test'] = 'test'
    file.sequencing_data = sequencing_data
    assert file.sequencing_data.attrs['test'] == 'test'


def test_get_sequencing_data(file):
    file.get_sequencing_data(margin=5, mapping_name='All files')


def test_generate_sequencing_match(file):
    file.generate_sequencing_match(overlapping_points_threshold=25, excluded_sequence_names=['MapSeq', 'CalSeq', '*'])


def test_insert_sequencing_data_into_file_dataset(file):
    file.insert_sequencing_data_into_file_dataset()