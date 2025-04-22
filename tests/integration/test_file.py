import pytest
import tifffile
import numpy as np


@pytest.fixture
def experiment(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'BN_TIRF')

@pytest.fixture
def file(experiment):
    return experiment.files[1]

@pytest.fixture
def experiment_hj(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'Papylio example dataset - analyzed')

@pytest.fixture
def file_hj(experiment_hj):
    return experiment_hj.files.select('HJ')[0]

@pytest.fixture
def experiment_output(shared_datadir):
    from papylio import Experiment
    return Experiment(shared_datadir / 'BN_TIRF_output_test_file')

@pytest.fixture
def file_output(experiment_output):
    return experiment_output.files[1]

@pytest.fixture
def file_output_with_selected(file_output):
    #TODO: This is not optimal, perhaps add selection to the file.
    selected = file_output.selected
    selected[[0,5,33]] = True
    file_output.set_variable(selected)
    return file_output

def test_projection_image(file, shared_datadir):
    image_newly_made = file.projection_image()
    image_from_original_file = tifffile.imread(shared_datadir / 'BN_TIRF_output_test_file' / 'TIRF 561 0001_ave_f0-10_i0.tif')
    assert (image_newly_made == image_from_original_file).all()
    image_loaded = file.projection_image()
    assert (image_loaded == image_from_original_file).all()

def test_average_image(file, shared_datadir):
    image_newly_made = file.average_image()
    image_from_original_file = tifffile.imread(shared_datadir / 'BN_TIRF_output_test_file' / 'TIRF 561 0001_ave_f0-10_i0.tif')
    assert (image_newly_made == image_from_original_file).all()
    image_loaded = file.average_image()
    assert (image_loaded == image_from_original_file).all()

def test_maximum_projection_image(file, shared_datadir):
    image_newly_made = file.maximum_projection_image()
    image_from_original_file = tifffile.imread(shared_datadir / 'BN_TIRF_output_test_file' / 'TIRF 561 0001_max_f0-10_i0.tif')
    assert (image_newly_made == image_from_original_file).all()
    image_loaded = file.maximum_projection_image()
    assert (image_loaded == image_from_original_file).all()

def test_perform_mapping(experiment, shared_datadir):
    import matchpoint as mp
    experiment.files[0].perform_mapping()
    mapping_control = mp.MatchPoint.load(shared_datadir / 'BN_TIRF_output_test_file' / 'beads.mapping')
    assert ((experiment.files[0].mapping.transformation.params - mapping_control.transformation.params) < 1e-2).all()

def test_find_molecules(file):
    file.find_coordinates()

def test_extract_traces(file):
    file.find_coordinates()
    file.extract_traces()
    file.extract_traces(mask_size=None, neighbourhood_size=None,
                        background_correction=(-150,-30),
                        alpha_correction=0.075,
                        gamma_correction=1.2)

def test_property_coordinates(file_output):
    file_output.coordinates

def test_determine_psf_size(file):
    psf_size = file.determine_psf_size()
    assert np.isclose(psf_size, 1.01, atol=0.005)

def test_show_histogram(file_output):
    file_output.show_histogram('intensity')
    file_output.show_histogram('FRET', bins=100, range=(0,1))

def test_show_traces(file_output):
    file_output.show_traces(selected=False)

def test_save_dataset_selected(file_output_with_selected):
    file_output_with_selected.save_dataset_selected()
    import xarray as xr
    ds = xr.load_dataset(file_output_with_selected.absoluteFilePath.parent / (file_output_with_selected.name + '_selected.nc'))
    indices_selected = np.nonzero(ds.molecule_in_file.values)[0]
    assert (indices_selected == np.array([0,5,33])).all().item()

def test_classify_hmm(file_output):
    selection = file_output.selected
    selection[0:20] = True
    file_output.set_variable(selection, name='selected')
    file_output.apply_classifications()
    file_output.classify_hmm('FRET')
    file_output.classify_hmm(file_output.intensity.sel(channel=0, drop=True))

def test_use_for_darkfield_correction(file):
    file.use_for_darkfield_correction()

def test_analyze_dwells(file_hj):
    file_hj.analyze_dwells(method='maximum_likelihood_estimation', number_of_exponentials=[1, 2, 3],
                           state_names={0: 'Low FRET',  1:'High FRET'},
                           plot=False)

def test_plot_dwell_analysis(file_hj):
    file_hj.analyze_dwells(method='maximum_likelihood_estimation', number_of_exponentials=[2],
                           state_names={0: 'Low FRET',  1:'High FRET'},
                           plot=False, fit_dwell_times_kwargs=dict())
    file_hj.plot_dwell_analysis(plot_type='pdf', log=False, plot_range=(0,3))