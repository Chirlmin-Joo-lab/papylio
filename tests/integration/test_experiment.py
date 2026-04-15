import pytest
import tifffile
import numpy as np
import json

from pytest_datadir.plugin import shared_datadir

@pytest.fixture
def experiment_hj(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'Papylio example dataset - analyzed')

def test_load_mapping(experiment_hj):
    experiment_hj.files[0].perform_mapping()
    experiment_hj.load_mappings()