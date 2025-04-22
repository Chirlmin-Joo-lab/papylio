import pytest
import tifffile
import numpy as np


from papylio.analysis.dwell_time_analysis import (ExponentialDistribution, fit_dwell_times, plot_dwell_analysis,
                                                  plot_dwell_analysis_state, analyze_dwells)
from tests.integration.test_dwell_time_extraction import dwells

@pytest.fixture
def dwell_times_single_exponential():
    return np.random.exponential(1, 1000)

@pytest.fixture
def dwell_times_double_exponential():
    # return np.hstack([np.random.exponential(1 / 0.3, 1000), np.random.exponential(1 / 0.01, 1000)])
    return np.hstack([np.random.exponential(1 / 0.3, 1000), np.random.exponential(1 / 0.1, 1000)])

def test_exponential_distribution():
    for i in range(3):
        ExponentialDistribution(i)

def test_maximum_likelihood_estimation(dwell_times_double_exponential):
    optimal_parameters = ExponentialDistribution(2).maximum_likelihood_estimation(dwell_times_double_exponential)

def test_histogram_fitting(dwell_times_double_exponential):
    optimal_parameters, _ = ExponentialDistribution(2).histogram_fit(dwell_times_double_exponential)

def test_cdf_fitting(dwell_times_double_exponential):
    optimal_parameters, _ = ExponentialDistribution(2).cdf_fit(dwell_times_double_exponential)

def test_fit_dwell_times(dwell_times_double_exponential):
    dwell_analysis = fit_dwell_times(dwell_times_double_exponential, 'maximum_likelihood_estimation', number_of_exponentials=[1,2,3])

    dwell_analysis = fit_dwell_times(dwell_times_double_exponential, 'maximum_likelihood_estimation',
        number_of_exponentials=[1, 2, 3], k_bounds=(0,100),
        analyze_dwells_kwargs=dict(scipy_optimization_method='dual_annealing'))

def test_plot_dwell_analysis_state(dwell_times_double_exponential):
    dwell_analysis = fit_dwell_times(dwell_times_double_exponential, 'maximum_likelihood_estimation',
                                     number_of_exponentials=[1,2,3])
    plot_dwell_analysis_state(dwell_analysis, dwell_times_double_exponential, log=True)

def test_analyze_dwells(dwells):
    dwell_analysis = analyze_dwells(dwells, state_names=None)

def test_plot_dwell_analysis(dwells):

    dwell_analysis = analyze_dwells(dwells, method='maximum_likelihood_estimation', number_of_exponentials=[1, 2, 3],
                                    state_names=None, P_bounds=(-1, 1), k_bounds=(0,np.inf),
                                    fit_dwell_times_kwargs=dict(free_truncation_min=False))
    plot_dwell_analysis(dwell_analysis, dwells, plot_range=(0, 2), axes=None, log=False, sharey=True, plot_type='pdf')

    dwell_analysis = analyze_dwells(dwells, method='histogram_fit', number_of_exponentials=[1,2,3], state_names=None,
                                    fit_dwell_times_kwargs=dict(maxfev=1e6), P_bounds=(-1,1), k_bounds=(0,np.inf))
    plot_dwell_analysis(dwell_analysis, dwells, plot_range=(0,2), axes=None, log=False, sharey=True, plot_type='pdf_binned')

    dwell_analysis = analyze_dwells(dwells, method='cdf_fit', number_of_exponentials=[1,2,3], state_names=None,
                                    fit_dwell_times_kwargs=dict(maxfev=1e6), P_bounds=(-1,1), k_bounds=(0,np.inf))
    plot_dwell_analysis(dwell_analysis, dwells, plot_range=(0,2), axes=None, log=False, sharey=True, plot_type='cdf')

    dwell_analysis = analyze_dwells(dwells, method='maximum_likelihood_estimation', number_of_exponentials=[1, 2, 3],
                                    state_names=None, P_bounds=(-1, 1), k_bounds=(0,10),
                                    fit_dwell_times_kwargs=dict(scipy_optimize_method='dual_annealing'))
    plot_dwell_analysis(dwell_analysis, dwells, plot_range=(0, 2), axes=None, log=False, sharey=True, plot_type='pdf')



def test_parameters_to_dataset(dwell_times_double_exponential):
    optimal_parameters, _ = ExponentialDistribution(2).histogram_fitting(dwell_times_double_exponential)
    ExponentialDistribution(2).parameters_to_dataset(optimal_parameters, dwell_times=dwell_times_double_exponential)