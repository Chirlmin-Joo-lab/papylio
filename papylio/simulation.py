# Simulation of single-molecule FRET intensity traces.
# Output: xarray DataArray with dimensions (molecule, channel, frame), with the true states stored,
# so analysis results can be compared with what was put in.
#
# Two versions:
#   simulate_intensity_traces                  -> one state per frame
#   simulate_intensity_traces_continuous_time  -> switches can happen during a frame (Ivo's option 2)

import numpy as np
import xarray as xr
from scipy.linalg import expm


def make_switching_probability_matrix(transition_rate_matrix, time_per_frame_in_seconds):
    """Turn the rates (per second) into the chance of switching within one frame.

    transition_rate_matrix[i, j] is the rate from state i to state j (diagonal is ignored).
    Returns P, where P[i, j] is the chance to be in state j one frame later if the molecule is in state i now.
    """
    # For a single rate this would just be 1 - exp(-k*dt).
    # For more states I use the matrix version P = expm(Q*dt), where Q is the rate matrix with
    # minus the total leaving rate on the diagonal (so every row of Q adds up to 0).
    # This also takes into account molecules that switch more than once within one frame.
    rate_matrix_with_diagonal = np.array(transition_rate_matrix, dtype=float)
    np.fill_diagonal(rate_matrix_with_diagonal, 0)
    total_rate_of_leaving_each_state = rate_matrix_with_diagonal.sum(axis=1)
    np.fill_diagonal(rate_matrix_with_diagonal, -total_rate_of_leaving_each_state)

    return expm(rate_matrix_with_diagonal * time_per_frame_in_seconds)


def determine_equilibrium_fraction_of_each_state(transition_rate_matrix):
    """Fraction of time spent in each state after a long time.

    I use this to pick the start state of each molecule. Before, I started all molecules in state 0,
    but then the beginning of the traces was biased towards state 0.
    For two states this gives k10 / (k01 + k10) and k01 / (k01 + k10).
    """
    # after many steps the molecule forgets where it started, so every row of P^n becomes the same
    switching_probability_matrix = make_switching_probability_matrix(transition_rate_matrix, 1.0)
    equilibrium_fraction_of_each_state = np.linalg.matrix_power(switching_probability_matrix, 100000)[0]
    return equilibrium_fraction_of_each_state / equilibrium_fraction_of_each_state.sum()   # remove rounding errors


def simulate_states_of_molecules(transition_rate_matrix, number_of_molecules, number_of_frames,
                                 time_per_frame_in_seconds, seed=None):
    """State (0, 1, 2, ...) of every molecule in every frame, shape (molecule, frame). One state per frame."""
    random_number_generator = np.random.default_rng(seed)
    number_of_states = len(transition_rate_matrix)

    switching_probability_matrix = make_switching_probability_matrix(transition_rate_matrix, time_per_frame_in_seconds)

    # How I pick the next state: take the row for the current state, e.g. [0.80, 0.15, 0.05],
    # and add it up step by step -> [0.80, 0.95, 1.00]. These are borders between 0 and 1.
    # A random number below 0.80 means state 0, between 0.80 and 0.95 state 1, above 0.95 state 2.
    cumulative_probability_matrix = np.cumsum(switching_probability_matrix, axis=1)

    states_of_all_molecules = np.zeros((number_of_molecules, number_of_frames), dtype=int)

    equilibrium_fraction_of_each_state = determine_equilibrium_fraction_of_each_state(transition_rate_matrix)
    states_of_all_molecules[:, 0] = random_number_generator.choice(number_of_states, size=number_of_molecules,
                                                                   p=equilibrium_fraction_of_each_state)

    # loop over the frames, but do all molecules at once (much faster than also looping over molecules)
    for frame_index in range(1, number_of_frames):
        states_in_previous_frame = states_of_all_molecules[:, frame_index - 1]
        cumulative_probabilities_for_each_molecule = cumulative_probability_matrix[states_in_previous_frame]
        random_numbers_for_each_molecule = random_number_generator.random(number_of_molecules)

        # the number of borders the random number is above = the new state
        new_states = np.sum(random_numbers_for_each_molecule[:, None] > cumulative_probabilities_for_each_molecule,
                            axis=1)

        # the last border can be 0.9999999 instead of 1 because of rounding, this makes sure
        # we never get a state number that doesn't exist
        states_of_all_molecules[:, frame_index] = np.minimum(new_states, number_of_states - 1)

    return states_of_all_molecules


def simulate_switching_times_of_one_molecule(transition_rate_matrix, total_duration_in_seconds, start_state,
                                             random_number_generator):
    """Exact switching times of one molecule (Gillespie algorithm).

    Returns the times at which the molecule enters a state (starting at 0) and which state that is.
    """
    transition_rate_matrix = np.array(transition_rate_matrix, dtype=float)
    np.fill_diagonal(transition_rate_matrix, 0)

    times_of_entering_a_state = [0.0]
    states_entered = [start_state]
    current_time = 0.0
    current_state = start_state

    while True:
        rates_of_leaving_to_each_state = transition_rate_matrix[current_state]
        total_rate_of_leaving = rates_of_leaving_to_each_state.sum()

        # instead of checking every frame, draw directly how long it stays (exponential, average 1/k)
        current_time = current_time + random_number_generator.exponential(1 / total_rate_of_leaving)
        if current_time >= total_duration_in_seconds:
            break

        # next state: chance proportional to the rate, e.g. rates [0, 1.0, 3.0] -> chances [0, 0.25, 0.75]
        current_state = random_number_generator.choice(len(rates_of_leaving_to_each_state),
                                                       p=rates_of_leaving_to_each_state / total_rate_of_leaving)
        times_of_entering_a_state.append(current_time)
        states_entered.append(current_state)

    return np.array(times_of_entering_a_state), np.array(states_entered)


def calculate_time_fraction_per_frame_in_each_state(times_of_entering_a_state, states_entered,
                                                    number_of_frames, time_per_frame_in_seconds, number_of_states):
    """Fraction of every frame spent in each state, shape (frame, state). Every row adds up to 1.

    E.g. if the molecule switches from 0 to 1 after 30% of a frame, that frame gets [0.3, 0.7].
    """
    # The idea: for every state I keep a "stopwatch" that only runs while the molecule is in that state.
    # Time in a state during one frame = stopwatch at the end of the frame - stopwatch at the start.
    total_duration_in_seconds = number_of_frames * time_per_frame_in_seconds
    times_where_something_changes = np.append(times_of_entering_a_state, total_duration_in_seconds)
    duration_of_each_dwell = np.diff(times_where_something_changes)
    times_of_frame_edges = np.arange(number_of_frames + 1) * time_per_frame_in_seconds

    time_fraction_per_frame_in_each_state = np.zeros((number_of_frames, number_of_states))
    for state in range(number_of_states):
        time_in_this_state_during_each_dwell = np.where(states_entered == state, duration_of_each_dwell, 0)
        cumulative_time_in_this_state = np.concatenate(([0], np.cumsum(time_in_this_state_during_each_dwell)))

        # between two switches the stopwatch runs at constant speed (or stands still),
        # so linear interpolation gives the reading at the frame edges
        cumulative_time_at_frame_edges = np.interp(times_of_frame_edges, times_where_something_changes,
                                                   cumulative_time_in_this_state)
        time_fraction_per_frame_in_each_state[:, state] = np.diff(cumulative_time_at_frame_edges) / time_per_frame_in_seconds

    return time_fraction_per_frame_in_each_state


def _check_simulation_inputs(number_of_states, transition_rate_matrix, intensity_of_each_state_per_channel, channel_names):
    # check the sizes first, so a mistake gives a clear error instead of a weird result later
    transition_rate_matrix = np.array(transition_rate_matrix, dtype=float)
    intensity_of_each_state_per_channel = np.array(intensity_of_each_state_per_channel, dtype=float)
    if transition_rate_matrix.shape != (number_of_states, number_of_states):
        raise ValueError('transition_rate_matrix should have shape (number_of_states, number_of_states)')
    if intensity_of_each_state_per_channel.shape != (number_of_states, len(channel_names)):
        raise ValueError('intensity_of_each_state_per_channel should have shape (number_of_states, number_of_channels)')
    return transition_rate_matrix, intensity_of_each_state_per_channel


def _make_traces_data_array(intensity, states_of_all_molecules, time_per_frame_in_seconds,
                            standard_deviation_of_intensity_noise, channel_names, simulation_method):
    # true_state is saved as a coordinate so we can compare it with the HMM result later,
    # and the settings go in attrs so it's always clear how the data was made
    number_of_molecules, _, number_of_frames = intensity.shape
    simulated_traces = xr.DataArray(
        intensity,
        dims=('molecule', 'channel', 'frame'),
        coords={'molecule': np.arange(number_of_molecules),
                'channel': list(channel_names),
                'frame': np.arange(number_of_frames),
                'time': ('frame', np.arange(number_of_frames) * time_per_frame_in_seconds),
                'true_state': (('molecule', 'frame'), states_of_all_molecules)},
        name='intensity')
    simulated_traces.attrs['time_per_frame_in_seconds'] = time_per_frame_in_seconds
    simulated_traces.attrs['standard_deviation_of_intensity_noise'] = standard_deviation_of_intensity_noise
    simulated_traces.attrs['simulation_method'] = simulation_method
    return simulated_traces


def simulate_intensity_traces(number_of_states, transition_rate_matrix, number_of_molecules, number_of_frames,
                              time_per_frame_in_seconds, intensity_of_each_state_per_channel,
                              standard_deviation_of_intensity_noise=0.0, channel_names=('donor', 'acceptor'),
                              seed=None):
    """Simulated intensity traces with one state per frame, as xarray DataArray (molecule, channel, frame).

    transition_rate_matrix[i][j]: rate from state i to state j (per second)
    intensity_of_each_state_per_channel[state][channel]: e.g. [[800, 200], [300, 700]]
        -> state 0: donor 800, acceptor 200; state 1: donor 300, acceptor 700
    standard_deviation_of_intensity_noise: gaussian noise added to every frame
    seed: same seed gives the same result
    """
    transition_rate_matrix, intensity_of_each_state_per_channel = _check_simulation_inputs(
        number_of_states, transition_rate_matrix, intensity_of_each_state_per_channel, channel_names)

    random_number_generator = np.random.default_rng(seed)

    states_of_all_molecules = simulate_states_of_molecules(transition_rate_matrix, number_of_molecules, number_of_frames,
                                                           time_per_frame_in_seconds, seed=random_number_generator)

    # indexing with the states gives (molecule, frame, channel), transpose to (molecule, channel, frame)
    intensity_without_noise = np.transpose(intensity_of_each_state_per_channel[states_of_all_molecules], (0, 2, 1))
    noise = random_number_generator.normal(0, standard_deviation_of_intensity_noise, size=intensity_without_noise.shape)

    return _make_traces_data_array(intensity_without_noise + noise, states_of_all_molecules, time_per_frame_in_seconds,
                                   standard_deviation_of_intensity_noise, channel_names, 'frame by frame')


def simulate_intensity_traces_continuous_time(number_of_states, transition_rate_matrix, number_of_molecules,
                                              number_of_frames, time_per_frame_in_seconds,
                                              intensity_of_each_state_per_channel,
                                              standard_deviation_of_intensity_noise=0.0,
                                              channel_names=('donor', 'acceptor'), seed=None):
    """Same inputs and output as simulate_intensity_traces, but switches can happen during a frame.

    The camera collects light during the whole frame, so a frame with a switch gets a mix of both
    intensities (weighted by the time in each state). true_state is the state in which most of the frame was spent.
    """
    transition_rate_matrix, intensity_of_each_state_per_channel = _check_simulation_inputs(
        number_of_states, transition_rate_matrix, intensity_of_each_state_per_channel, channel_names)

    random_number_generator = np.random.default_rng(seed)
    total_duration_in_seconds = number_of_frames * time_per_frame_in_seconds
    equilibrium_fraction_of_each_state = determine_equilibrium_fraction_of_each_state(transition_rate_matrix)

    intensity_without_noise = np.zeros((number_of_molecules, len(channel_names), number_of_frames))
    state_with_most_time_in_each_frame = np.zeros((number_of_molecules, number_of_frames), dtype=int)

    # here I loop over molecules, because every molecule has a different number of switches
    for molecule_index in range(number_of_molecules):
        start_state = random_number_generator.choice(number_of_states, p=equilibrium_fraction_of_each_state)
        times_of_entering_a_state, states_entered = simulate_switching_times_of_one_molecule(
            transition_rate_matrix, total_duration_in_seconds, start_state, random_number_generator)

        time_fraction_per_frame_in_each_state = calculate_time_fraction_per_frame_in_each_state(
            times_of_entering_a_state, states_entered, number_of_frames, time_per_frame_in_seconds, number_of_states)

        # frame intensity = sum over states of (fraction of the frame in that state x intensity of that state)
        # as a matrix multiplication: (frame, state) @ (state, channel) -> (frame, channel), then .T -> (channel, frame)
        intensity_without_noise[molecule_index] = (time_fraction_per_frame_in_each_state @ intensity_of_each_state_per_channel).T
        state_with_most_time_in_each_frame[molecule_index] = np.argmax(time_fraction_per_frame_in_each_state, axis=1)

    noise = random_number_generator.normal(0, standard_deviation_of_intensity_noise, size=intensity_without_noise.shape)

    return _make_traces_data_array(intensity_without_noise + noise, state_with_most_time_in_each_frame,
                                   time_per_frame_in_seconds, standard_deviation_of_intensity_noise, channel_names,
                                   'continuous time')
