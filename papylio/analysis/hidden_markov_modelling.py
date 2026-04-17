"""Hidden Markov Model (HMM) utilities for trace classification.

Provides functions to fit HMMs to single-molecule traces, classify traces using
fitted models, extract state parameters, transition matrices, and compute
transition rates with uncertainty handling.
"""
import xarray as xr
import numpy as np
from itertools import accumulate, groupby
import pomegranate as pg
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM
import tqdm
from objectlist import ObjectList
from copy import deepcopy
import scipy.linalg

# file.FRET, file.classification, file.selected
def classify_hmm(traces, classification, selection, n_states=2, threshold_state_mean=None, level='molecule', seed=0):
    """Classify traces using Hidden Markov Models (HMMs).

    Fits HMMs either per-molecule or per-file and returns a Dataset with
    classification and state/transition statistics.

    Parameters
    ----------
    traces : xr.DataArray
        Trace data with dims ('molecule','frame',...)
    classification : xr.DataArray
        Initial classification or mask used to select segments for fitting
    selection : xr.DataArray or boolean array
        Selection mask of molecules to include
    n_states : int, optional
        Number of HMM hidden states (default: 2)
    threshold_state_mean : float, optional
        Optional threshold to filter low-amplitude states
    level : {'molecule','file'}, optional
        Fit HMMs per molecule or a single HMM per file (default: 'molecule')
    seed : int, optional
        RNG seed for reproducibility (default: 0)

    Returns
    -------
    xr.Dataset
        Dataset containing HMM classification, state parameters and transition rates
    """
    np.random.seed(seed)
    if level == 'molecule':
        models_per_molecule = fit_hmm_to_individual_traces(traces, classification, selection, parallel=False, n_states=n_states, threshold_state_mean=threshold_state_mean)
    elif level == 'file':
        model = fit_hmm_to_file(traces, classification, selection, n_states=n_states, threshold_state_mean=threshold_state_mean)
        number_of_molecules = np.shape(traces)[0]
        models_per_molecule = [deepcopy(model) for _ in range(number_of_molecules)]
    else:
        raise RuntimeError('Hidden markov modelling can be performed on the molecule of file level. Indicate this with level=\'molecule\' or level=\'file\'')

    ds = xr.Dataset()
    if models_per_molecule is None:
        # TODO: Is this if statement still necessary, now that we set the return_none_if_all_none to False
        # ds['selection_complex_rates'] = xr.ones_like(selection, dtype=bool)
        # ds['selection_lower_rate_limit'] = xr.ones_like(selection, dtype=bool)
        # ds['classification_hmm'] = -xr.ones_like(classification)
        # return ds
        raise RuntimeError('If you see this error please let Ivo know.')

    ds['number_of_states'] = number_of_states_from_models(models_per_molecule)
    state_parameters = state_parameters_from_models(models_per_molecule, n_states=n_states)
    transition_matrices = transition_matrices_from_models(models_per_molecule, n_states=n_states)
    classification_hmm = trace_classification_models(traces, classification, models_per_molecule).astype('int8')
    # ds['state_parameters'], ds['transition_matrix'] = \
    state_parameters, transition_matrices, classification_hmm = \
        sort_states_in_data(state_parameters, transition_matrices, classification_hmm)
    ds['classification'] = classification_hmm

    ds['state_mean'] = state_parameters.sel(parameter=0)
    ds['state_standard_deviation'] = state_parameters.sel(parameter=1)
    ds['transition_probability'] = transition_matrices[:, :n_states, :n_states]
    ds['start_probability'] = transition_matrices[:, -2, :n_states]
    ds['end_probability'] = transition_matrices[:, :n_states, -1]

    # TODO: Perhaps we should not add additional selections, just encode the selections as negative values for the whole trace in classifications?

    number_of_frames = len(traces.frame)
    frame_rate = 1 / traces.time.diff('frame').mean().item()
    transition_rates = determine_transition_rates_from_probabilities(ds.number_of_states, ds.transition_probability,
                                                                     frame_rate)
    transition_rates, ds['selection_complex_rates'] = complex_transition_rates_to_nan(transition_rates)

    ds['transition_rate'], ds['selection_lower_rate_limit'] = \
        transition_rates_outside_measurement_resolution_to_nan(transition_rates, number_of_frames, frame_rate)

    return ds



def BIC(model, xis):
    """Compute Bayesian Information Criterion (BIC) for a fitted HMM model and datasets.

    Parameters
    ----------
    model : pomegranate.HiddenMarkovModel or pomegranate.Distribution
        Fitted model object
    xis : sequence of arrays
        List of observed sequences used to compute likelihood

    Returns
    -------
    float
        BIC value (-2 * log-likelihood + k * log(n))
    """
    if isinstance(model, Normal):
        k = 2
        likelihood = model.probability(np.concatenate(xis)).prod()
    else:
        state_count = len(model.distributions)
        k = state_count * (state_count - 1) + state_count * 2
        # likelihood = model.predict_proba(xi).max(axis=1).prod()
        log_likelihood = np.sum([model.log_probability(xii.reshape(1, -1, 1)) for xii in xis])

    n = len(np.concatenate(xis))
    bic_value = -2 * log_likelihood + k * np.log(n)
    return bic_value

    #     likelihood = np.prod([model.probability(xii) for xii in xis])
    #
    # n = len(np.concatenate(xis))
    # bic_value_old = -2*np.log(likelihood) + k * np.log(n)


def split_by_classification(xi, classification):
    """Split an observation sequence xi and a matching classification array into contiguous segments.

    Returns two lists: the list of xi segments and the corresponding classification segments.
    """
    split_indices = np.nonzero(np.diff(classification))[0] + 1
    cis = np.split(classification, split_indices)
    xis = np.split(xi, split_indices)
    # cis = [cii[0] for cii in cis]
    return xis, cis


def hmm1and2(input):
    """Fit 1- and 2-state HMMs to input traces and choose the better model by BIC.

    Parameters
    ----------
    input : tuple
        (xi, classification, selected) where xi is a trace array for a molecule,
        classification is a per-frame classification, and selected is a boolean.

    Returns
    -------
    pomegranate.HiddenMarkovModel or None
        Best model selected by BIC, or None if fitting failed or selection criteria not met.
    """
    xi, classification, selected = input
    # xi = molecule.FRET.values
    # classification = molecule.classification.values
    if not selected:
        return None

    classification_st_0 = classification < 0
    if classification_st_0.all():
        return None
    if (~classification_st_0).sum() < 2:
        return None

    included_frame_selection = classification >= 0
    xis, cis = split_by_classification(xi, included_frame_selection)

    xis = [xii for cii, xii in zip(cis, xis) if cii[0]]

    dist1 = Normal().fit(np.concatenate(xis))
    model1 = DenseHMM.from_matrix([[1]], [dist1], [1])
    # model1 = pg.HiddenMarkovModel.fit(pg.distributions.Normal, n_components=1, X=[xi])
    model2 = DenseHMM([Normal(), Normal()])
    model2.fit(xis)

    bic_model1 = BIC(model1, xis)
    bic_model2 = BIC(model2, xis)

    if bic_model1 < bic_model2:
        # parameters = np.array(model1.parameters)
        # transition_matrix = np.array([1])
        return model1
    elif bic_model2 < bic_model1:
        # parameters = np.vstack([model2.states[0].distribution.parameters, model2.states[1].distribution.parameters])
        # transition_matrix = model2.dense_transition_matrix()
        return model2
    else:
        # parameters = None
        # transition_matrix = None
        return None
        # raise(ValueError)

    # return parameters, transition_matrix


def hmm_n_states(input, n_states=2, threshold_state_mean=None, level='molecule'):
    """Fit HMMs with up to n_states and return the best model by BIC.

    Parameters
    ----------
    input : tuple or list
        (xi, classification, selected) or lists when level='file'
    n_states : int
        Maximum number of states to consider
    threshold_state_mean : float or None
        Minimum separation required between state means to consider models distinct
    level : {'molecule','file'}
        Whether to fit per-molecule segments or treat the entire file as one sequence

    Returns
    -------
    pomegranate.HiddenMarkovModel or None
        Best-fit model or None if no suitable model found
    """

    xi, classification, selected = input
    xi = xi.astype(np.float32)

    if np.sum(selected) == 0:
        return None

    classification_st_0 = classification < 0
    if classification_st_0.all():
        return None
    if (~classification_st_0).sum() < 2:
        return None

    included_frame_selection = classification >= 0
    xis, cis = split_by_classification(xi, included_frame_selection)
    xis = [xii.reshape(-1,1) for xii in xis]

    if level == 'molecule':
        xis = [xii for cii, xii in zip(cis, xis) if cii[0]]
    elif level == 'file':
        xis_new = []
        for xii, cii in zip(xis, cis):
            if len(xii[cii]) > 0:
                xis_new.append(xii[cii])
        xis = xis_new
    else:
        raise RuntimeError('Hidden markov modelling can be performed on the molecule of file level. Indicate this with level=\'molecule\' or level=\'file\'')

    best_model = None
    best_bic = np.inf

    for state in range(1, n_states + 1):
        if state == 1:
            dist1 = Normal().fit(np.concatenate(xis))
            model = DenseHMM([dist1], [[1]], [1], ends=[1])
        else:
            try:
                mean1 = dist1.means[0]
                cov1 = dist1.covs[0,0]
                model = DenseHMM([Normal([mean1+cov1*state], [[cov1]]) for state in range(1, state+1)])
                model.fit(xis)
            except ValueError:
                continue

        bic = BIC(model, xis)

        if threshold_state_mean:
            means = np.array([dist.means for dist in model.distributions])
            result = bool(np.any(np.abs(means - means.T)[np.triu_indices(n_states, 1)] > threshold_state_mean))
        else:
            result = True

        if bic < best_bic and result:
            best_bic = bic
            best_model = model

    if best_model is not None:
        return best_model
    else:
        return None

def fit_hmm_to_individual_traces(traces, classification, selected, parallel=False, n_states=2, threshold_state_mean=None):
    """Fit HMMs to each molecule's trace individually (optionally in parallel).

    Returns a list of models (one per molecule), or None entries when fitting failed.
    """
    cf = ObjectList(list(zip(traces.values, classification.values, selected.values)), return_none_if_all_none=False)
    cf.use_parallel_processing = parallel
    models_per_molecule = cf.map(hmm_n_states)(n_states=n_states, threshold_state_mean=threshold_state_mean)  # New taking sections into account 5540 traces: 5:02
        # Old not taking sections into account: 5092 traces 4:00 minutes (2:37 on server)
    return models_per_molecule


def fit_hmm_to_file(traces, classification, selected, n_states=2, threshold_state_mean=None):
    """Fit a single HMM to the entire file (all molecules concatenated or treated as sections).

    Returns a single model instance applicable to every molecule.
    """
    input_values = [traces.values, classification.values, selected.values]
    models = hmm_n_states(input_values, n_states=n_states, threshold_state_mean=threshold_state_mean, level='file')
    return models


def number_of_states_from_models(models):
    """Return DataArray with number of hidden states (excluding start/end) per model."""

    number_of_states = [len(model.distributions) if model is not None else 0 for model in models]
    # state_count = []
    # for model in models:
    #     if model is None:
    #         state_count.append(0)
    #     else:
    #         state_count.append(model.state_count()-2)

    return xr.DataArray(number_of_states, dims='molecule')


def state_parameters_from_models(models, n_states=2):
    """Extract state distribution parameters (mean, std) from each model into an xr.DataArray.

    Returns array shaped (molecule, state, parameter) where parameter 0=mean, 1=std.
    """
    max_number_of_states = n_states
    number_of_parameters = 2

    state_parameters = np.full((len(models), max_number_of_states, number_of_parameters), np.nan)
    for i, model in enumerate(models):
        if model is not None:
            means = np.array([distribution.means.item() for distribution in model.distributions])
            stds = np.array([np.sqrt(distribution.covs.item()).item() for distribution in model.distributions])
            sp = np.hstack([means[:, None], stds[:, None]])
            state_parameters[i, :sp.shape[0], :] = sp
    return xr.DataArray(state_parameters, dims=('molecule', 'state', 'parameter'))


def transition_matrices_from_models(models, n_states=2):
    """Extract transition matrices from models and pack into xr.DataArray.

    Each matrix is padded/truncated to (n_states+2)x(n_states+2) to include start/end rows/cols.
    """
    max_number_of_states = n_states
    transition_matrix = np.full((len(models), max_number_of_states+2, max_number_of_states+2), np.nan)
    for i, model in enumerate(models):
        if model is not None:
            tm = model.edges.numpy()
            number_of_states = tm.shape[0]
            transition_matrix[i, :number_of_states, :number_of_states] = tm[:number_of_states, :number_of_states]
            transition_matrix[i, -2, :number_of_states] = model.starts.numpy()
            transition_matrix[i, :number_of_states, -1] = model.ends.numpy()
            # transition_matrix[i, -2:, -2:] = tm[-2:, -2:]
    transition_matrix = np.exp(transition_matrix) # As apparently pomegranate give log probabilities.

    #         start_index = max_number_of_states+2-tm.shape[0]
    #         transition_matrix[i,start_index:,start_index:] = tm
    # transition_matrix = transition_matrix[:, [1,0,2,3],:][:,:,[1,0,2,3]] # In case of a single state this put the state at index 0.
    return xr.DataArray(transition_matrix, dims=('molecule','from_state','to_state'))


def sort_states_in_data(state_parameters, transition_matrices, classification_hmm):
    """Sort states by their mean value and reorder transition matrices and classifications accordingly.

    This ensures a consistent ordering (e.g., low->high FRET) across molecules.
    """
    sort_indices = state_parameters[:, :, 0].argsort(axis=1)

    sort_indices_start_end_states = xr.DataArray([[2, 3]] * len(state_parameters.molecule), dims=('molecule', 'state'))
    sort_indices_transition_matrix = xr.concat([sort_indices, sort_indices_start_end_states], dim='state')

    state_parameters = state_parameters.sel(state=xr.DataArray(sort_indices, dims=('molecule', 'state')))
    transition_matrices = transition_matrices.sel(from_state=sort_indices_transition_matrix.rename(state='from_state'),
                                              to_state=sort_indices_transition_matrix.rename(state='to_state'))

    classification_hmm_sorted = xr.DataArray(np.zeros_like(classification_hmm), dims=('molecule', 'frame'))
    for molecule in range(len(sort_indices.molecule)):
        mapping = {val: i for i, val in enumerate(sort_indices[molecule].values)}
        mapping.update({val: -1 for val in classification_hmm[molecule, :].values if val < 0})
        classification_hmm_sorted[molecule, :] = np.vectorize(mapping.get)(classification_hmm[molecule, :].values)
    classification_hmm[:] = classification_hmm_sorted[:]

    return state_parameters, transition_matrices, classification_hmm


def trace_classification_model(traces, model):
    """Apply a fitted model to traces and return per-frame class assignments as DataArray."""
    classification = np.vstack([model.predict(traces[m].values) for m in traces.molecule.values])
    classification = xr.DataArray(classification, dims=traces.dims)
    return classification


def trace_classification_models(traces, classifications, models):
    """Use per-molecule models to classify traces respecting original selection masks.

    For molecules with model==None, returns -1 for all frames (indicating no classification).
    """
    new_classifications = []
    for model, xi, ci in zip(models, traces.values, classifications.values):
        if model is not None:
            included_frame_selection = ci >= 0
            xis, _ = split_by_classification(xi, included_frame_selection)
            cis, _ = split_by_classification(ci, included_frame_selection)

            new_classification = []
            for xii, cii in zip(xis, cis):
                if cii[0] >= 0:
                    new_classification.append(model.predict(xii.reshape(1,-1, 1)))
                else:
                    new_classification.append(-np.ones_like(cii))
            new_classification = np.hstack(new_classification)
            new_classifications.append(new_classification)
        else:
            new_classifications.append(-np.ones_like(xi))
    return xr.DataArray(np.vstack(new_classifications), dims=('molecule','frame'))


def determine_transition_rates_from_probabilities(number_of_states, transition_probabilities, frame_rate):
    """Convert transition probability matrices to transition rate matrices via matrix logarithm.

    Parameters
    ----------
    number_of_states : xr.DataArray or array-like
        Number of states per molecule
    transition_probabilities : xr.DataArray
        Transition probability matrices per molecule
    frame_rate : float
        Frame rate in Hz to scale discrete probabilities to rates

    Returns
    -------
    xr.DataArray
        Transition rates with the same dims as input probabilities (molecule, from_state, to_state)
    """
    # transition_rates = np.full_like(transition_probabilities, np.nan)
    dims = transition_probabilities.dims
    transition_rates = np.full_like(transition_probabilities, np.nan, dtype=np.complex64)
    transition_probabilities = np.array(transition_probabilities)
    number_of_states = np.array(number_of_states)

    for i in range(len(transition_probabilities)):
        if number_of_states[i] > 0:
            transition_rates[i, :number_of_states[i], :number_of_states[i]] = \
                scipy.linalg.logm(transition_probabilities[i, :number_of_states[i], :number_of_states[i]].T).T

    transition_rates = transition_rates * frame_rate

    return xr.DataArray(transition_rates, dims=dims)


def complex_transition_rates_to_nan(transition_rates):
    """Mask transition rate matrices that contain complex entries, returning real part and valid mask."""
    is_complex = xr.DataArray((np.iscomplex(transition_rates) & ~np.isnan(transition_rates)).any(axis=2).any(axis=1), dims=('molecule'))
    transition_rates[is_complex, :, :] = np.nan
    return np.real(transition_rates), ~is_complex


def transition_rates_outside_measurement_resolution_to_nan(transition_rates, number_of_frames, frame_rate):
    """Set transition rates that are too small to be resolved (below 1 / (observation time)) to NaN.

    Returns modified transition_rates and a boolean selection mask indicating which molecules pass the resolution test.
    """
    # For more than two states we likely only have to take the off diagonal components
    off_diagonal_terms = transition_rates.values[:, ~np.eye(*transition_rates.shape[1:], dtype=bool)]
    has_too_low_rate = xr.DataArray((np.abs(off_diagonal_terms) < frame_rate/number_of_frames).any(axis=1), dims='molecule')
    # has_too_low_rate = xr.DataArray(np.diagonal(np.abs(transition_rates), axis1=1, axis2=2).any(axis=1), dims='molecule')
    # has_too_high_rate = (np.abs(ds.transition_rate) > frame_rate).any(axis=2).any(axis=1)
    # transition_rates[has_too_low_rate | has_too_high_rate, :, :] = np.nan
    transition_rates[has_too_low_rate, :, :] = np.nan
    return transition_rates, ~has_too_low_rate #, ~has_too_high_rate


def histogram_1d_state_means(ds, name, save_path, number_of_states=1, state_index=0):
    """Plot and save a histogram of 1D state means (e.g., FRET) for molecules with a given state count."""
    fig, ax = plt.subplots(figsize=(6.5,3.5), tight_layout=True)
    ds_subset = ds.sel(molecule=ds.number_of_states==number_of_states)
    if state_index > number_of_states-1:
        raise ValueError('State index larger than number of states')

    ax.hist(ds_subset.state_mean.sel(state=state_index), bins=50, range=(0,1))
    ax.set_xlabel('Mean FRET')
    ax.set_ylabel('Molecule count')
    title = name + f' - FRET histogram - state {state_index+1} out of {number_of_states}'
    ax.set_title(title)
    fig.savefig(save_path / (title + '.png'))

    # counts, bins = np.histogram(parameters_one_state[:,0], bins=50, range=(0,1))
    # print("Max at E=", (bins[counts.argmax()]+bins[counts.argmax()+1])/2)

import matplotlib.pyplot as plt
from matplotlib import cm
def histogram_2d_state_means(ds, name, save_path, number_of_states=2, state_indices=[0,1]):
    """Plot and save 2D histogram of mean values for two specified states."""
    fig, ax = plt.subplots(figsize=(8,6.5), tight_layout=True)
    ds_subset = ds.sel(molecule=ds.number_of_states==number_of_states)
    for state_index in state_indices:
        if state_index > number_of_states-1:
            raise ValueError('State index larger than number of states')

    ax.hist2d(*ds_subset.state_mean.sel(state=state_indices).T, bins=50, range=((0, 1), (0, 1)))
    ax.set_xlabel(f'Mean FRET - state {state_indices[0]+1}')
    ax.set_ylabel(f'Mean FRET - state {state_indices[1]+1}')
    cax = fig.colorbar(ax.collections[0], ax=ax, label='Molecule count')
    ax.set_aspect(1)
    title = name + f' - FRET histogram - States {state_indices[0]+1} and {state_indices[1]+1} out of {number_of_states}'
    ax.set_title(title)
    fig.savefig(save_path / (title + '.png'))



def histogram_2d_transition_rates(ds, name, save_path, frame_rate, number_of_states=2, state_indices=[0,1]):
    """Plot and save 2D histogram of transition rates between two states."""
    fig, ax = plt.subplots(figsize=(8, 6.5), tight_layout=True)
    ds_subset = ds.sel(molecule=ds.number_of_states == number_of_states)
    for state_index in state_indices:
        if state_index > number_of_states - 1:
            raise ValueError('State index larger than number of states')

    state_A_to_B = ds_subset.transition_rate.sel(from_state=state_indices[0], to_state=state_indices[1])
    state_B_to_A = ds_subset.transition_rate.sel(from_state=state_indices[1], to_state=state_indices[0])

    # ax.hist2d(state_A_to_B, state_B_to_A, bins=50, range=((0, frame_rate), (0, frame_rate)))
    ax.hist2d(state_A_to_B, state_B_to_A, bins=50, range=((0, 16), (0, 16)))
    ax.set_xlabel(f'Transition rate (/s) - state {state_indices[0] + 1}')
    ax.set_ylabel(f'Transition rate (/s) - state {state_indices[1] + 1}')
    cax = fig.colorbar(ax.collections[0], ax=ax, label='Molecule count')
    ax.set_aspect(1)
    title = name + f' - Transition rate histogram - States {state_indices[0] + 1} and {state_indices[1] + 1} out of {number_of_states}'
    ax.set_title(title)
    fig.savefig(save_path / (title + '.png'))


def transition_rate_fit(ds, frame_rate, number_of_states=2, from_state=0, to_state=1):
    """Fit a Gaussian to the kernel density estimate of transition rates and return mean/std.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing transition_rate variables
    frame_rate : float
        Frame rate in Hz used to set plotting range
    number_of_states : int
        Number of states to select molecules by
    from_state, to_state : int
        State indices for which to extract transition rates

    Returns
    -------
    (mean, std) : tuple of floats
        Mean and standard deviation of fitted Gaussian to KDE of rates
    """
    # fig, ax = plt.subplots(figsize=(8, 6.5), tight_layout=True)
    ds_subset = ds.sel(molecule=ds.number_of_states == number_of_states)

    transition_rates = ds_subset.transition_rate.sel(from_state=from_state, to_state=to_state).values
    transition_rates = transition_rates[~np.isnan(transition_rates)]

    import scipy.stats
    kernel = scipy.stats.gaussian_kde(transition_rates)
    def gaussian_single(x, a, mean, std):
        # print(x,a,mean,std)
        return a * np.exp(-1/2 * (x-mean)**2 / std**2)

    import scipy.optimize
    x = np.linspace(0,frame_rate,200)
    popt, pcov = scipy.optimize.curve_fit(gaussian_single, x, kernel(x), bounds=((0,-np.inf,0),(np.inf, np.inf, np.inf)))

    a, mean, std = popt
    plt.figure()
    plt.plot(x,kernel(x))
    plt.plot(x,gaussian_single(x, *popt))

    return mean, std

