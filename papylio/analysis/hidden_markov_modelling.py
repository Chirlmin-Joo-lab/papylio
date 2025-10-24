import xarray as xr
import numpy as np
from itertools import accumulate, groupby
from hmmlearn import hmm
import pomegranate as pg
import tqdm
from objectlist import ObjectList
from copy import deepcopy
import scipy.linalg
from papylio.log_functions import add_configuration_to_dataarray

# file.FRET, file.classification, file.selected
def classify_hmm(traces, classification, selection, n_states=2, threshold_state_mean=None, level='molecule', seed=0):
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
    if isinstance(model, pg.NormalDistribution):
        k = 2
        likelihood = model.probability(np.concatenate(xis)).prod()
    else:
        state_count = (model.state_count() - 2)
        k = state_count * (state_count - 1) + state_count * 2
        # likelihood = model.predict_proba(xi).max(axis=1).prod()
        log_likelihood = np.sum([model.log_probability(xii) for xii in xis])

    n = len(np.concatenate(xis))
    bic_value = -2 * log_likelihood + k * np.log(n)
    return bic_value

    #     likelihood = np.prod([model.probability(xii) for xii in xis])
    #
    # n = len(np.concatenate(xis))
    # bic_value_old = -2*np.log(likelihood) + k * np.log(n)


def split_by_classification(xi, classification):
    split_indices = np.nonzero(np.diff(classification))[0] + 1
    cis = np.split(classification, split_indices)
    xis = np.split(xi, split_indices)
    # cis = [cii[0] for cii in cis]
    return xis, cis


def hmm1and2(input):
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

    dist1 = pg.NormalDistribution.from_samples(np.concatenate(xis))
    model1 = pg.HiddenMarkovModel.from_matrix([[1]], [dist1], [1])
    # model1 = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution, n_components=1, X=[xi])
    model2 = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution, n_components=2, X=xis)

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

    xi, classification, selected = input

    if np.sum(selected) == 0:
        return None

    classification_st_0 = classification < 0
    if classification_st_0.all():
        return None
    if (~classification_st_0).sum() < 2:
        return None

    included_frame_selection = classification >= 0
    xis, cis = split_by_classification(xi, included_frame_selection)

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
            dist1 = pg.NormalDistribution.from_samples(np.concatenate(xis))
            model = pg.HiddenMarkovModel.from_matrix([[1]], [dist1], [1])
        else:
            try:
                model = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution, n_components=state, X=xis)
            except ValueError:
                continue

        bic = BIC(model, xis)

        if threshold_state_mean:
            state_means = []
            for state in model.states:
                if state.distribution:
                    if isinstance(state.distribution, pg.NormalDistribution):
                        state_means.append(state.distribution.parameters[0])

            def check_difference(state_means, threshold=threshold_state_mean):
                for i in range(len(state_means)):
                    for j in range(i + 1, len(state_means)):
                        if abs(state_means[i] - state_means[j]) < threshold:
                            return False
                return True

            result = check_difference(state_means, threshold_state_mean)

            if bic < best_bic and result:
                best_bic = bic
                best_model = model
        else:
            if bic < best_bic:
                best_bic = bic
                best_model = model

    if best_model is not None:
        return best_model
    else:
        return None

def fit_hmm_to_individual_traces(traces, classification, selected, parallel=False, n_states=2, threshold_state_mean=None):
    cf = ObjectList(list(zip(traces.values, classification.values, selected.values)), return_none_if_all_none=False)
    cf.use_parallel_processing = parallel
    models_per_molecule = cf.map(hmm_n_states)(n_states=n_states, threshold_state_mean=threshold_state_mean)  # New taking sections into account 5540 traces: 5:02
        # Old not taking sections into account: 5092 traces 4:00 minutes (2:37 on server)
    return models_per_molecule

def fit_hmm_to_file(traces, classification, selected, n_states=2, threshold_state_mean=None):
    input_values = [traces.values, classification.values, selected.values]
    models = hmm_n_states(input_values, n_states=n_states, threshold_state_mean=threshold_state_mean, level='file')
    return models

def number_of_states_from_models(models):

    number_of_states = [model.state_count()-2 if model is not None else 0 for model in models]
    # state_count = []
    # for model in models:
    #     if model is None:
    #         state_count.append(0)
    #     else:
    #         state_count.append(model.state_count()-2)

    return xr.DataArray(number_of_states, dims='molecule')

def state_parameters_from_models(models, n_states=2):
    max_number_of_states = n_states
    number_of_parameters = 2

    state_parameters = np.full((len(models), max_number_of_states, number_of_parameters), np.nan)
    for i, model in enumerate(models):
        if model is not None:
            sp = np.vstack([state.distribution.parameters for state in model.states[:-2]])
            state_parameters[i, :sp.shape[0], :] = sp
    return xr.DataArray(state_parameters, dims=('molecule', 'state', 'parameter'))

def transition_matrices_from_models(models, n_states=2):
    max_number_of_states = n_states
    transition_matrix = np.full((len(models), max_number_of_states+2, max_number_of_states+2), np.nan)
    for i, model in enumerate(models):
        if model is not None:
            tm = model.dense_transition_matrix()
            number_of_states = tm.shape[0]-2
            transition_matrix[i, :number_of_states, :number_of_states] = tm[:number_of_states, :number_of_states]
            transition_matrix[i, -2:, :number_of_states] = tm[-2:, :number_of_states]
            transition_matrix[i, :number_of_states, -2:] = tm[:number_of_states, -2:,]
            transition_matrix[i, -2:, -2:] = tm[-2:, -2:]
    #         start_index = max_number_of_states+2-tm.shape[0]
    #         transition_matrix[i,start_index:,start_index:] = tm
    # transition_matrix = transition_matrix[:, [1,0,2,3],:][:,:,[1,0,2,3]] # In case of a single state this put the state at index 0.
    return xr.DataArray(transition_matrix, dims=('molecule','from_state','to_state'))

def sort_states_in_data(state_parameters, transition_matrices, classification_hmm):
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
    classification = np.vstack([model.predict(traces[m].values) for m in traces.molecule.values])
    classification = xr.DataArray(classification, dims=traces.dims)
    return classification

def trace_classification_models(traces, classifications, models):
    new_classifications = []
    for model, xi, ci in zip(models, traces.values, classifications.values):
        if model is not None:
            included_frame_selection = ci >= 0
            xis, _ = split_by_classification(xi, included_frame_selection)
            cis, _ = split_by_classification(ci, included_frame_selection)

            new_classification = []
            for xii, cii in zip(xis, cis):
                if cii[0] >= 0:
                    new_classification.append(model.predict(xii))
                else:
                    new_classification.append(-np.ones_like(cii))
            new_classification = np.hstack(new_classification)
            new_classifications.append(new_classification)
        else:
            new_classifications.append(-np.ones_like(xi))
    return xr.DataArray(np.vstack(new_classifications), dims=('molecule','frame'))

def determine_transition_rates_from_probabilities(number_of_states, transition_probabilities, frame_rate):
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
    is_complex = xr.DataArray((np.iscomplex(transition_rates) & ~np.isnan(transition_rates)).any(axis=2).any(axis=1), dims=('molecule'))
    transition_rates[is_complex, :, :] = np.nan
    return np.real(transition_rates), ~is_complex

def transition_rates_outside_measurement_resolution_to_nan(transition_rates, number_of_frames, frame_rate):
    # For more than two states we likely only have to take the off diagonal components
    off_diagonal_terms = transition_rates.values[:, ~np.eye(*transition_rates.shape[1:], dtype=bool)]
    has_too_low_rate = xr.DataArray((np.abs(off_diagonal_terms) < frame_rate/number_of_frames).any(axis=1), dims='molecule')
    # has_too_low_rate = xr.DataArray(np.diagonal(np.abs(transition_rates), axis1=1, axis2=2).any(axis=1), dims='molecule')
    # has_too_high_rate = (np.abs(ds.transition_rate) > frame_rate).any(axis=2).any(axis=1)
    # transition_rates[has_too_low_rate | has_too_high_rate, :, :] = np.nan
    transition_rates[has_too_low_rate, :, :] = np.nan
    return transition_rates, ~has_too_low_rate #, ~has_too_high_rate


def histogram_1d_state_means(ds, name, save_path, number_of_states=1, state_index=0):
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




















def decode_traces(model, dataarray):
    classifications = []
    for i in tqdm.tqdm(dataarray.molecule):
        d_i = dataarray.sel(molecule=i).values.T
        log_prob, classification_HMM = model.decode(d_i)
        classifications.append(classification_HMM)
    return xr.DataArray(np.vstack(classifications), dims=('molecule','frame'))


# exp = ta.Experiment(r'D:\20200918 - Test data\Single-molecule data small')
# exp = ta.Experiment(r'P:\SURFdrive\Promotie\Data\Test data')

# ds = xr.open_mfdataset(file_paths, concat_dim='molecule', combine='nested')
# classification_background = ~(ds.intensity < 5000).any('channel')

# file = files_green_laser[0]
# for file in files:
#
#     traces = file.dataset.FRET #ds.intensity.sel(channel=0, drop=True)

# n_components=2, covariance_type="full", n_iter=100

def hmm_traces(traces, initial_boolean_classification=None, **kwargs):
    if initial_boolean_classification is None:
        initial_boolean_classification = [None]*len(traces)

    classification = -xr.ones_like(traces).astype(int)
    classification.name = 'classification'
    hmm_transition_matrix = xr.DataArray(np.ones((len(traces.molecule), 2, 2))*np.nan, coords={},
                                         dims=['molecule', 'from_state', 'to_state'], name='hmm_transition_matrix')
    hmm_means = xr.DataArray(np.ones((len(traces.molecule), 2))*np.nan, coords={}, dims=['molecule','state'], name='hmm_means')
    hmm_log_probs = xr.DataArray(np.ones(len(traces.molecule))*np.nan, coords={}, dims='molecule', name='hmm_log_probabilities')

    for molecule in tqdm(traces.molecule.values):
        hmm_transition_matrix[molecule],  hmm_means[molecule], classification[molecule], hmm_log_probs[molecule] = \
            hmm_trace(traces[molecule], initial_boolean_classification=initial_boolean_classification[molecule], **kwargs)

    return xr.merge([hmm_transition_matrix, hmm_means, hmm_log_probs, classification])


    # hmm_transition_matrix.name = 'hmm_transition_matrix'
    # hmm_transition_matrix.to_netcdf(file.relativeFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')
    # hmm_means.name = 'hmm_means'
    # hmm_means.to_netcdf(file.relativeFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')
    # classification.name = 'classification'
    # classification.to_netcdf(file.relativeFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')

    # nds = xr.Dataset({'hmm_transition_matrix': hmm_transition_matrix, 'hmm_means': hmm_means, 'classification': classification})
    # nds.load()
    # nds.to_netcdf(file.relativeFilePath.with_suffix('.nc'), engine='h5netcdf', mode='a')

# def hmm_traces(traces, **kwargs):
#     hmm_transition_matrix, hmm_means, classification, hmm_log_probs =\
#         xr.apply_ufunc(
#             hmm_trace,
#             traces,
#             kwargs=kwargs,
#             input_core_dims=[['frame']],
#             output_core_dims=[['from_state','to_state'],['state'],['frame'],[]],
#             # dask="parallelized",
#             output_dtypes=[np.ndarray, np.ndarray, np.ndarray, float],
#             dask_gufunc_kwargs=dict(output_sizes={"from_state": 2, "to_state": 2, 'state': 2}),
#             vectorize=True
#         )
#     hmm_transition_matrix.name = 'hmm_transition_matrix'
#     hmm_means.name = 'hmm_means'
#     classification.name = 'classification'
#     hmm_log_probs.name = 'hmm_log_probabilities'
#     return xr.merge([hmm_transition_matrix, hmm_means, hmm_log_probs, classification])


def hmm_trace(trace, initial_boolean_classification=None, **kwargs):
    # TODO: Test handling of initial_boolean_classification
    # print(trace)
    # FRET = np.atleast_2d(traces.sel(molecule=molecule_index)).T
    trace_ = np.array(trace).reshape(-1, 1)
    # print(trace_)
    if initial_boolean_classification is not None:
        # initial_boolean_classification = np.ones_like(trace).astype(bool)
        trace_ = trace_[initial_boolean_classification]
        segment_lengths = [sum(1 for _ in group) for value, group in groupby(initial_boolean_classification) if value]
    else:
        segment_lengths = None
    model = hmm.GaussianHMM(**kwargs)
    model.fit(trace_, segment_lengths)
    try:
        log_prob, classification_HMM = model.decode(trace_)
        #classification_HMM = xr.DataArray(classification_HMM, coords=trace.coords)
        if initial_boolean_classification is not None:
            classification_HMM[~initial_boolean_classification] = -1
    except ValueError:
        print('!!!')
        log_prob = np.nan
        classification_HMM = -np.ones_like(trace).astype(int)

    return model.transmat_, model.means_.squeeze(), classification_HMM, log_prob



# p_stay = np.diagonal(hmm_transition_matrix, axis1=1, axis2=2)
#
# dt = 0.1 #s
# tau = -dt/np.log(p_stay)


# for file_path in file_paths:
#     with xr.open_dataset(file_path) as ds:
#         # ds = ds.reset_index('molecule', drop=True)
#         # # ds = ds.reset_index('dim_1', drop=True)
#         # ds = ds.reset_index('frame', drop=True)
#         # ds = ds.reset_index('from_state', drop=True)
#         # ds = ds.reset_index('to_state', drop=True)
#         ds.load()
#     ds['channel'] = [0,1]
#
#     ds.to_netcdf(file_path, engine='h5netcdf', mode='w')


