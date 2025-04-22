import numpy as np
import xarray as xr

def dwell_frames_from_classification(classification):
    # This assumes continuous monitoring with a specific cycle time.
    single_true_array = np.ones((classification.shape[0],1)).astype(bool)
    is_state_transition = np.hstack([single_true_array, classification[:,:-1] != classification[:,1:], single_true_array])
    state_transition_molecule, state_transition_frame = np.where(is_state_transition)
    is_endpoint = state_transition_frame == classification.shape[1]
    dwell_states = classification[state_transition_molecule[~is_endpoint], state_transition_frame[~is_endpoint]]
    dwell_frames = np.diff(state_transition_frame)[~is_endpoint[:-1]]
    dwell_molecules = state_transition_molecule[~is_endpoint]
    return dwell_molecules, dwell_states, dwell_frames

def determine_dwell_means(traces_flattened, dwell_frames):
    mean_trace = np.mean(traces_flattened)
    values_cumsum = np.concatenate([[0], np.cumsum(traces_flattened-mean_trace)])
    oneD_indices = np.concatenate([[0],dwell_frames.cumsum()])
    dwell_means = np.diff(values_cumsum[oneD_indices]) / dwell_frames + mean_trace
    return dwell_means

def set_states(dwell_molecules, dwell_states, at_trace_edges=True, around_negative_states=True, to_state=-2):
    states_to_set = np.zeros(len(dwell_states), dtype=bool)

    switched_molecule = np.diff(dwell_molecules).astype(bool)
    start_and_end_trace = np.concatenate([[True], switched_molecule]) | np.concatenate([switched_molecule, [True]])

    negative_states = dwell_states < 0

    if at_trace_edges:
        states_to_set |= start_and_end_trace

    if around_negative_states:
        negative_state_neighbours = np.concatenate([[False], negative_states[:-1]]) | \
                                    np.concatenate([negative_states[1:], [False]])
        states_to_set |= negative_state_neighbours & ~start_and_end_trace

    states_to_set[negative_states] = False

    dwell_states[states_to_set] = to_state
    return dwell_states


def dwell_times_from_classification(classification, traces=None, cycle_time=None, inactivate_start_and_end_states=True):
    if isinstance(classification, xr.DataArray):
        molecule_coords = {n: c.values for n, c in classification.coords.items() if c.dims[0] == 'molecule' and len(c.dims) == 1}
        classification = classification.values
    else:
        molecule_coords = None
    dwell_molecules, dwell_states, dwell_frames = dwell_frames_from_classification(classification)
    if inactivate_start_and_end_states:
        dwell_states = set_states(dwell_molecules, dwell_states, to_state=-128)
        # Probably better to indicate for each dwell whether it is at a trace edge or around a negative state

    ds = xr.Dataset()
    if molecule_coords is not None:
        for n, c in molecule_coords.items():
            ds[n] = ('dwell', c[dwell_molecules])
    else:
        ds['molecule'] = ('dwell', dwell_molecules)

    ds['state'] = ('dwell', dwell_states)
    ds['frame_count'] = ('dwell', dwell_frames)

    if cycle_time is not None:
        dwell_times = dwell_frames * cycle_time
        ds['duration'] =('dwell', dwell_times)

    if traces is not None:
        # if isinstance(traces, xr.Dataset):
        #     for name, da in traces.data_vars.items():
        #         dwell_means = determine_dwell_means(da.values.flatten(), dwell_frames)
        #         ds['mean'] = xr.DataArray(dwell_means, dims=['dwell'])
        name = ''
        if isinstance(traces, xr.DataArray):
            if traces.name is not None:
                name = '_' + traces.name
            traces = traces.values

        dwell_means = determine_dwell_means(traces.flatten(), dwell_frames)
        ds['mean'+name] = ('dwell', dwell_means)

    return ds












