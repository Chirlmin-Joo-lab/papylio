import pytest
from papylio.plugins.sequencing.sequence_generation import generate_sequences
from papylio.plugins.sequencing.sequence_plotting import plot_sequence_density

@pytest.fixture
def sequences():
    return generate_sequences(base_composition=['N', 'AC', 'GC', 'ACTG'])


def test_plot_sequence_density(sequences):
    plot_sequence_density(sequences, expected_seq=None, start=None, end=None, row_length=None,
                          figure=None, save=False, title='')

