"""Simulation of single-molecule intensity traces.

Functions to simulate molecules that switch between states (Markov chain) and
turn the state of every frame into donor and acceptor intensities.
The output is an xarray DataArray with dimensions (molecule, channel, frame).

Two methods are available:
- simulate_intensity_traces: the state is decided once per frame
- simulate_intensity_traces_continuous_time: switches can happen at any moment,
  so a frame in which a switch happens gets a mix of the intensities of both states
"""
import numpy as np
import xarray as xr
from scipy.linalg import expm


def make_switching_probability_matrix(transition_rate_matrix, time_per_frame_in_seconds):
    """Convert a transition rate matrix into switching probabilities per frame.

    Parameters
    ----------
    transition_rate_matrix : array-like, shape (number_of_states, number_of_states)
        transition_rate_matrix[i, j] is the rate (per second) to go from state i to state j.
        The diagonal is ignored.
    time_per_frame_in_seconds : float
        Time between two frames.

    Returns
    -------
    np.ndarray, shape (number_of_states, number_of_states)
        result[i, j] is the chance to be in state j one frame later when the molecule is in state i now.
    """
    # the diagonal has to be minus the total rate of leaving that state for the matrix exponential
    rate_matrix_with_diagonal = np.array(transition_rate_matrix, dtype=float)
    np.fill_diagonal(rate_matrix_with_diagonal, 0)
    total_rate_of_leaving_each_state = rate_matrix_with_diagonal.sum(axis=1)
    np.fill_diagonal(rate_matrix_with_diagonal, -total_rate_of_leaving_each_state)

    # expm (matrix exponential) is the multi-state version of exp(-k*dt)
    return expm(rate_matrix_with_diagonal * time_per_frame_in_seconds)


def determine_equilibrium_fraction_of_each_state(transition_rate_matrix):
    """Fraction of time spent in each state after a long time (equilibrium).

    Parameters
    ----------
    transition_rate_matrix : array-like, shape (number_of_states, number_of_states)

    Returns
    -------
    np.ndarray, shape (number_of_states,)
    """
    # after a very long time a molecule forgets where it started,
    # so the probability matrix to a high power gives the equilibrium fractions
    switching_probability_matrix = make_switching_probability_matrix(transition_rate_matrix, 1.0)
    equilibrium_fraction_of_each_state = np.linalg.matrix_power(switching_probability_matrix, 100000)[0]
    return equilibrium_fraction_of_each_state / equilibrium_fraction_of_each_state.sum()


def simulate_states_of_molecules(transition_rate_matrix, number_of_molecules, number_of_frames,
                                 time_per_frame_in_seconds, seed=None):
    """Simulate the state of every molecule in every frame (one state per frame).

    Parameters
    ----------
    transition_rate_matrix : array-like, shape (number_of_states, number_of_states)
        Rates in switches per second, transition_rate_matrix[i, j] is from state i to state j.
    number_of_molecules : int
    number_of_frames : int
    time_per_frame_in_seconds : float
    seed : int, np.random.Generator or None
        Seed for the random number generator, use the same seed to get the same result.

    Returns
    -------
    np.ndarray, shape (number_of_molecules, number_of_frames)
        State number (0, 1, 2, ...) of every molecule in every frame.
    """
    random_number_generator = np.random.default_rng(seed)
    number_of_states = len(transition_rate_matrix)

    switching_probability_matrix = make_switching_probability_matrix(transition_rate_matrix, time_per_frame_in_seconds)

    # cumulative sum along each row, e.g. row [0.8, 0.15, 0.05] becomes [0.8, 0.95, 1.0]
    # a random number between 0 and 1 then falls in one of the intervals and that tells the next state
    cumulative_probability_matrix = np.cumsum(switching_probability_matrix, axis=1)

    # start state drawn from the equilibrium fractions
    equilibrium_fraction_of_each_state = determine_equilibrium_fraction_of_each_state(transition_rate_matrix)
    states_of_all_molecules = np.zeros((number_of_molecules, number_of_frames), dtype=int)
    states_of_all_molecules[:, 0] = random_number_generator.choice(number_of_states, size=number_of_molecules,
                                                                   p=equilibrium_fraction_of_each_state)

    for frame_index in range(1, number_of_frames):
        states_in_previous_frame = states_of_all_molecules[:, frame_index - 1]

        # for every molecule take the row of the cumulative matrix that belongs to its current state
        cumulative_probabilities_for_each_molecule = cumulative_probability_matrix[states_in_previous_frame]

        # count how many interval borders the random number is above, that number is the new state
        random_numbers_for_each_molecule = random_number_generator.random(number_of_molecules)
        new_states = np.sum(random_numbers_for_each_molecule[:, None] > cumulative_probabilities_for_each_molecule, axis=1)

        # safety for tiny rounding errors (last value of a row can be 0.9999999 instead of 1)
        states_of_all_molecules[:, frame_index] = np.minimum(new_states, number_of_states - 1)

    return states_of_all_molecules


def simulate_switching_times_of_one_molecule(transition_rate_matrix, total_duration_in_seconds, start_state,
                                             random_number_generator):
    """Simulate the exact switching moments of one molecule in continuous time (Gillespie algorithm).

    Parameters
    ----------
    transition_rate_matrix : array-like, shape (number_of_states, number_of_states)
    total_duration_in_seconds : float
    start_state : int
    random_number_generator : np.random.Generator

    Returns
    -------
    times_of_entering_a_state : np.ndarray
        Times (in seconds) at which the molecule enters a new state, starting with 0.
    states_entered : np.ndarray
        The state entered at each of those times.
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

        # time spent in the current state is exponentially distributed with mean 1 / total rate
        time_spent_in_current_state = random_number_generator.exponential(1 / total_rate_of_leaving)
        current_time = current_time + time_spent_in_current_state
        if current_time >= total_duration_in_seconds:
            break

        # which state it goes to: chance proportional to the rate towards that state
        current_state = random_number_generator.choice(len(rates_of_leaving_to_each_state),
                                                       p=rates_of_leaving_to_each_state / total_rate_of_leaving)
        times_of_entering_a_state.append(current_time)
        states_entered.append(current_state)

    return np.array(times_of_entering_a_state), np.array(states_entered)


def calculate_time_fraction_per_frame_in_each_state(times_of_entering_a_state, states_entered,
                                                    number_of_frames, time_per_frame_in_seconds, number_of_states):
    """Fraction of each frame that the molecule spent in each state.

    Parameters
    ----------
    times_of_entering_a_state, states_entered : np.ndarray
        Output of simulate_switching_times_of_one_molecule.
    number_of_frames : int
    time_per_frame_in_seconds : float
    number_of_states : int

    Returns
    -------
    np.ndarray, shape (number_of_frames, number_of_states)
        Each row sums to 1.
    """
    # for every state keep track of the total time spent in that state up to every moment,
    # the difference between the start and end of a frame is the time spent in that state during the frame
    total_duration_in_seconds = number_of_frames * time_per_frame_in_seconds
    times_where_something_changes = np.append(times_of_entering_a_state, total_duration_in_seconds)
    duration_of_each_dwell = np.diff(times_where_something_changes)
    times_of_frame_edges = np.arange(number_of_frames + 1) * time_per_frame_in_seconds

    time_fraction_per_frame_in_each_state = np.zeros((number_of_frames, number_of_states))
    for state in range(number_of_states):
        time_in_this_state_during_each_dwell = np.where(states_entered == state, duration_of_each_dwell, 0)
        cumulative_time_in_this_state = np.concatenate(([0], np.cumsum(time_in_this_state_during_each_dwell)))
        # between switches the cumulative time grows linearly, so we can interpolate
        cumulative_time_at_frame_edges = np.interp(times_of_frame_edges, times_where_something_changes,
                                                   cumulative_time_in_this_state)
        time_fraction_per_frame_in_each_state[:, state] = np.diff(cumulative_time_at_frame_edges) / time_per_frame_in_seconds

    return time_fraction_per_frame_in_each_state


def _check_simulation_inputs(number_of_states, transition_rate_matrix, intensity_of_each_state_per_channel, channel_names):
    transition_rate_matrix = np.array(transition_rate_matrix, dtype=float)
    intensity_of_each_state_per_channel = np.array(intensity_of_each_state_per_channel, dtype=float)
    if transition_rate_matrix.shape != (number_of_states, number_of_states):
        raise ValueError('transition_rate_matrix should have shape (number_of_states, number_of_states)')
    if intensity_of_each_state_per_channel.shape != (number_of_states, len(channel_names)):
        raise ValueError('intensity_of_each_state_per_channel should have shape (number_of_states, number_of_channels)')
    return transition_rate_matrix, intensity_of_each_state_per_channel


def _make_traces_data_array(intensity, states_of_all_molecules, time_per_frame_in_seconds,
                            standard_deviation_of_intensity_noise, channel_names, simulation_method):
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
    """Simulate intensity traces, with one state per frame.

    Parameters
    ----------
    number_of_states : int
    transition_rate_matrix : array-like, shape (number_of_states, number_of_states)
        Rates in switches per second, transition_rate_matrix[i, j] is from state i to state j.
    number_of_molecules : int
    number_of_frames : int
    time_per_frame_in_seconds : float
    intensity_of_each_state_per_channel : array-like, shape (number_of_states, number_of_channels)
        Mean intensity of each channel in each state.
    standard_deviation_of_intensity_noise : float
        Standard deviation of the gaussian noise added to every frame.
    channel_names : tuple of str
    seed : int, np.random.Generator or None

    Returns
    -------
    xr.DataArray, dims (molecule, channel, frame)
        Simulated intensities, with coordinates time and true_state.
    """
    transition_rate_matrix, intensity_of_each_state_per_channel = _check_simulation_inputs(
        number_of_states, transition_rate_matrix, intensity_of_each_state_per_channel, channel_names)

    random_number_generator = np.random.default_rng(seed)

    states_of_all_molecules = simulate_states_of_molecules(transition_rate_matrix, number_of_molecules, number_of_frames,
                                                           time_per_frame_in_seconds, seed=random_number_generator)

    # replace every state by its intensity in each channel: (molecule, frame, channel) -> (molecule, channel, frame)
    intensity_without_noise = np.transpose(intensity_of_each_state_per_channel[states_of_all_molecules], (0, 2, 1))
    noise = random_number_generator.normal(0, standard_deviation_of_intensity_noise, size=intensity_without_noise.shape)

    return _make_traces_data_array(intensity_without_noise + noise, states_of_all_molecules, time_per_frame_in_seconds,
                                   standard_deviation_of_intensity_noise, channel_names, 'frame by frame')


def simulate_intensity_traces_continuous_time(number_of_states, transition_rate_matrix, number_of_molecules,
                                              number_of_frames, time_per_frame_in_seconds,
                                              intensity_of_each_state_per_channel,
                                              standard_deviation_of_intensity_noise=0.0,
                                              channel_names=('donor', 'acceptor'), seed=None):
    """Simulate intensity traces in continuous time, so switches can happen during a frame.

    Same parameters and output as simulate_intensity_traces. A frame in which a switch happens
    gets a mix of the intensities of the states, weighted by the time spent in each state.
    The true_state coordinate gives the state in which most of the frame was spent.
    """
    transition_rate_matrix, intensity_of_each_state_per_channel = _check_simulation_inputs(
        number_of_states, transition_rate_matrix, intensity_of_each_state_per_channel, channel_names)

    random_number_generator = np.random.default_rng(seed)
    total_duration_in_seconds = number_of_frames * time_per_frame_in_seconds
    equilibrium_fraction_of_each_state = determine_equilibrium_fraction_of_each_state(transition_rate_matrix)

    intensity_without_noise = np.zeros((number_of_molecules, len(channel_names), number_of_frames))
    state_with_most_time_in_each_frame = np.zeros((number_of_molecules, number_of_frames), dtype=int)

    for molecule_index in range(number_of_molecules):
        start_state = random_number_generator.choice(number_of_states, p=equilibrium_fraction_of_each_state)
        times_of_entering_a_state, states_entered = simulate_switching_times_of_one_molecule(
            transition_rate_matrix, total_duration_in_seconds, start_state, random_number_generator)

        time_fraction_per_frame_in_each_state = calculate_time_fraction_per_frame_in_each_state(
            times_of_entering_a_state, states_entered, number_of_frames, time_per_frame_in_seconds, number_of_states)

        # (frame, state) @ (state, channel) -> (frame, channel), then transpose to (channel, frame)
        intensity_without_noise[molecule_index] = (time_fraction_per_frame_in_each_state @ intensity_of_each_state_per_channel).T
        state_with_most_time_in_each_frame[molecule_index] = np.argmax(time_fraction_per_frame_in_each_state, axis=1)

    noise = random_number_generator.normal(0, standard_deviation_of_intensity_noise, size=intensity_without_noise.shape)

    return _make_traces_data_array(intensity_without_noise + noise, state_with_most_time_in_each_frame,
                                   time_per_frame_in_seconds, standard_deviation_of_intensity_noise, channel_names,
                                   'continuous time')
