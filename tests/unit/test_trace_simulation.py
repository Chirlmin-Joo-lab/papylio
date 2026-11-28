# Tests for papylio/simulation.py
# Run with:  pytest tests/unit/test_trace_simulation.py
#
# Most tests use small simulations so they run fast. The statistical tests use enough
# molecules that the result should be close to theory, with a tolerance that is wide
# enough to not fail by chance.

import numpy as np
import pytest

from papylio.simulation import (make_switching_probability_matrix,
                                determine_equilibrium_fraction_of_each_state,
                                check_starting_probabilities,
                                add_blinking_and_bleaching_states,
                                calculate_time_fraction_per_frame_in_each_state,
                                simulate_switching_times_of_one_molecule,
                                simulate_intensity_traces,
                                simulate_intensity_traces_continuous_time)

# settings used in most tests: 2 states, 0 -> 1 with 2 per second, 1 -> 0 with 1 per second
transition_rate_matrix_two_states = [[0, 2.0],
                                     [1.0, 0]]
intensity_of_each_state_per_channel_two_states = [[800, 200],
                                                  [300, 700]]

# both simulation methods have the same inputs and output, so most tests are run for both
both_simulation_functions = [simulate_intensity_traces, simulate_intensity_traces_continuous_time]


def simulate_with_default_settings(simulation_function, number_of_molecules=20, number_of_frames=100,
                                   standard_deviation_of_intensity_noise=50, seed=1):
    return simulation_function(number_of_states=2,
                               transition_rate_matrix=transition_rate_matrix_two_states,
                               number_of_molecules=number_of_molecules,
                               number_of_frames=number_of_frames,
                               time_per_frame_in_seconds=0.1,
                               intensity_of_each_state_per_channel=intensity_of_each_state_per_channel_two_states,
                               standard_deviation_of_intensity_noise=standard_deviation_of_intensity_noise,
                               seed=seed)


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def test_switching_probability_matrix_rows_add_up_to_one():
    # every row is "where can the molecule be one frame later", so the chances should add up to 1
    transition_rate_matrix_three_states = [[0, 1.0, 0.5],
                                           [2.0, 0, 1.0],
                                           [0.5, 0.5, 0]]
    switching_probability_matrix = make_switching_probability_matrix(transition_rate_matrix_three_states, 0.1)
    assert np.allclose(switching_probability_matrix.sum(axis=1), 1)
    assert np.all(switching_probability_matrix >= 0)


def test_switching_probability_matrix_for_short_frames():
    # for a very short frame the chance to switch should be close to rate x frame time
    time_per_frame_in_seconds = 1e-4
    switching_probability_matrix = make_switching_probability_matrix(transition_rate_matrix_two_states, time_per_frame_in_seconds)
    assert switching_probability_matrix[0, 1] == pytest.approx(2.0 * time_per_frame_in_seconds, rel=1e-3)
    assert switching_probability_matrix[1, 0] == pytest.approx(1.0 * time_per_frame_in_seconds, rel=1e-3)


def test_equilibrium_fractions_two_states():
    # for two states: f0 = k10 / (k01 + k10) and f1 = k01 / (k01 + k10)
    equilibrium_fraction_of_each_state = determine_equilibrium_fraction_of_each_state(transition_rate_matrix_two_states)
    assert equilibrium_fraction_of_each_state == pytest.approx([1 / 3, 2 / 3])


def test_time_fractions_per_frame_simple_example():
    # molecule starts in state 0 and switches to state 1 after 0.13 s, frames of 0.1 s:
    # frame 0 is fully state 0, frame 1 is 30% state 0 and 70% state 1, frame 2 is fully state 1
    time_fraction_per_frame_in_each_state = calculate_time_fraction_per_frame_in_each_state(
        times_of_entering_a_state=np.array([0.0, 0.13]),
        states_entered=np.array([0, 1]),
        number_of_frames=3,
        time_per_frame_in_seconds=0.1,
        number_of_states=2)
    expected = [[1.0, 0.0],
                [0.3, 0.7],
                [0.0, 1.0]]
    assert np.allclose(time_fraction_per_frame_in_each_state, expected)


def test_switching_times_are_increasing_and_inside_measurement():
    random_number_generator = np.random.default_rng(1)
    times_of_entering_a_state, states_entered = simulate_switching_times_of_one_molecule(
        transition_rate_matrix_two_states, total_duration_in_seconds=50, start_state=0,
        random_number_generator=random_number_generator)
    assert times_of_entering_a_state[0] == 0
    assert np.all(np.diff(times_of_entering_a_state) > 0)
    assert times_of_entering_a_state[-1] < 50
    # with two states the molecule has to alternate 0, 1, 0, 1, ...
    assert np.all(np.diff(states_entered) != 0)


# ---------------------------------------------------------------------------
# main simulation functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('simulation_function', both_simulation_functions)
def test_output_shape_and_coordinates(simulation_function):
    simulated_traces = simulate_with_default_settings(simulation_function, number_of_molecules=20, number_of_frames=100)
    assert simulated_traces.dims == ('molecule', 'channel', 'frame')
    assert simulated_traces.shape == (20, 2, 100)
    assert list(simulated_traces.channel.values) == ['donor', 'acceptor']
    assert simulated_traces.true_state.shape == (20, 100)
    assert np.allclose(simulated_traces.time.values, np.arange(100) * 0.1)


@pytest.mark.parametrize('simulation_function', both_simulation_functions)
def test_same_seed_gives_same_result(simulation_function):
    first_simulation = simulate_with_default_settings(simulation_function, seed=5)
    second_simulation = simulate_with_default_settings(simulation_function, seed=5)
    different_seed = simulate_with_default_settings(simulation_function, seed=6)
    assert np.array_equal(first_simulation.values, second_simulation.values)
    assert not np.array_equal(first_simulation.values, different_seed.values)


@pytest.mark.parametrize('simulation_function', both_simulation_functions)
def test_wrong_input_sizes_give_an_error(simulation_function):
    # 3 states given, but the rate matrix is 2x2
    with pytest.raises(ValueError):
        simulation_function(number_of_states=3,
                            transition_rate_matrix=transition_rate_matrix_two_states,
                            number_of_molecules=5, number_of_frames=10, time_per_frame_in_seconds=0.1,
                            intensity_of_each_state_per_channel=intensity_of_each_state_per_channel_two_states)
    # intensities for 3 channels, but only 2 channel names
    with pytest.raises(ValueError):
        simulation_function(number_of_states=2,
                            transition_rate_matrix=transition_rate_matrix_two_states,
                            number_of_molecules=5, number_of_frames=10, time_per_frame_in_seconds=0.1,
                            intensity_of_each_state_per_channel=[[800, 200, 100], [300, 700, 100]])


def test_intensities_without_noise_match_true_states():
    # without noise and with one state per frame, every frame should have exactly the intensity of its state
    simulated_traces = simulate_with_default_settings(simulate_intensity_traces, standard_deviation_of_intensity_noise=0)
    intensity_of_each_state_per_channel = np.array(intensity_of_each_state_per_channel_two_states)
    expected_intensity = np.transpose(intensity_of_each_state_per_channel[simulated_traces.true_state.values], (0, 2, 1))
    assert np.allclose(simulated_traces.values, expected_intensity)


def test_continuous_time_without_noise_stays_between_state_intensities():
    # mixed frames are a weighted average of the states, so the donor intensity can never go
    # outside the range of the state intensities, and donor + acceptor stays 1000 here
    simulated_traces = simulate_with_default_settings(simulate_intensity_traces_continuous_time,
                                                      standard_deviation_of_intensity_noise=0)
    donor_intensity = simulated_traces.sel(channel='donor').values
    acceptor_intensity = simulated_traces.sel(channel='acceptor').values
    assert donor_intensity.min() >= 300 - 1e-9
    assert donor_intensity.max() <= 800 + 1e-9
    assert np.allclose(donor_intensity + acceptor_intensity, 1000)


@pytest.mark.parametrize('simulation_function', both_simulation_functions)
def test_fraction_of_time_in_each_state_matches_equilibrium(simulation_function):
    # with many molecules the fraction of frames in state 1 should be close to 2/3
    simulated_traces = simulate_with_default_settings(simulation_function, number_of_molecules=300,
                                                      number_of_frames=500, seed=2)
    fraction_of_time_in_state_1 = np.mean(simulated_traces.true_state.values == 1)
    assert fraction_of_time_in_state_1 == pytest.approx(2 / 3, abs=0.02)


def test_noise_has_the_requested_standard_deviation():
    # the difference between the noisy and noise-free simulation (same seed for the states) is the noise
    simulated_traces = simulate_with_default_settings(simulate_intensity_traces, number_of_molecules=200,
                                                      number_of_frames=200, standard_deviation_of_intensity_noise=50)
    intensity_of_each_state_per_channel = np.array(intensity_of_each_state_per_channel_two_states)
    intensity_without_noise = np.transpose(intensity_of_each_state_per_channel[simulated_traces.true_state.values], (0, 2, 1))
    noise = simulated_traces.values - intensity_without_noise
    assert np.mean(noise) == pytest.approx(0, abs=1)
    assert np.std(noise) == pytest.approx(50, rel=0.02)


# ---------------------------------------------------------------------------
# starting probabilities, blinking and bleaching
# ---------------------------------------------------------------------------

def test_starting_probabilities_default_is_equilibrium():
    assert check_starting_probabilities(None, transition_rate_matrix_two_states) == pytest.approx([1 / 3, 2 / 3])


def test_wrong_starting_probabilities_give_an_error():
    with pytest.raises(ValueError):
        check_starting_probabilities([0.5, 0.2], transition_rate_matrix_two_states)        # doesn't add up to 1
    with pytest.raises(ValueError):
        check_starting_probabilities([1.0, 0, 0], transition_rate_matrix_two_states)       # 3 values for 2 states
    with pytest.raises(ValueError):
        check_starting_probabilities([1.5, -0.5], transition_rate_matrix_two_states)       # negative value


@pytest.mark.parametrize('simulation_function', both_simulation_functions)
def test_given_starting_probabilities_are_used(simulation_function):
    # all molecules start in state 1
    simulated_traces = simulation_function(number_of_states=2,
                                           transition_rate_matrix=transition_rate_matrix_two_states,
                                           number_of_molecules=50, number_of_frames=5, time_per_frame_in_seconds=0.1,
                                           intensity_of_each_state_per_channel=intensity_of_each_state_per_channel_two_states,
                                           starting_probabilities=[0, 1], seed=1)
    assert np.all(simulated_traces.true_state.values[:, 0] == 1)


def test_bleaching_adds_a_state_that_cannot_be_left():
    transition_rate_matrix, intensity_of_each_state_per_channel, starting_probabilities = \
        add_blinking_and_bleaching_states(transition_rate_matrix_two_states,
                                          intensity_of_each_state_per_channel_two_states, bleaching_rate=0.1)
    assert transition_rate_matrix.shape == (3, 3)
    assert np.all(transition_rate_matrix[:2, 2] == 0.1)     # every state goes to bleached with the bleaching rate
    assert np.all(transition_rate_matrix[2] == 0)           # nothing comes out of the bleached state
    assert np.all(intensity_of_each_state_per_channel[2] == 0)
    assert starting_probabilities == pytest.approx([1 / 3, 2 / 3, 0])


def test_blinking_adds_a_dark_copy_of_every_state():
    transition_rate_matrix, intensity_of_each_state_per_channel, starting_probabilities = \
        add_blinking_and_bleaching_states(transition_rate_matrix_two_states,
                                          intensity_of_each_state_per_channel_two_states,
                                          blinking_rate=0.5, blinking_recovery_rate=5.0)
    assert transition_rate_matrix.shape == (4, 4)
    assert transition_rate_matrix[0, 2] == 0.5 and transition_rate_matrix[1, 3] == 0.5   # going dark
    assert transition_rate_matrix[2, 0] == 5.0 and transition_rate_matrix[3, 1] == 5.0   # coming back
    assert transition_rate_matrix[2, 3] == 2.0 and transition_rate_matrix[3, 2] == 1.0   # still switching while dark
    assert np.all(intensity_of_each_state_per_channel[2:] == 0)
    assert starting_probabilities == pytest.approx([1 / 3, 2 / 3, 0, 0])


@pytest.mark.parametrize('simulation_function', both_simulation_functions)
def test_molecules_stay_bleached(simulation_function):
    transition_rate_matrix, intensity_of_each_state_per_channel, starting_probabilities = \
        add_blinking_and_bleaching_states(transition_rate_matrix_two_states,
                                          intensity_of_each_state_per_channel_two_states, bleaching_rate=1.0)
    simulated_traces = simulation_function(number_of_states=3, transition_rate_matrix=transition_rate_matrix,
                                           number_of_molecules=50, number_of_frames=200, time_per_frame_in_seconds=0.1,
                                           intensity_of_each_state_per_channel=intensity_of_each_state_per_channel,
                                           starting_probabilities=starting_probabilities, seed=1)
    true_state = simulated_traces.true_state.values
    for state_of_each_frame in true_state:
        frames_bleached = np.flatnonzero(state_of_each_frame == 2)
        if len(frames_bleached) > 0:
            # once bleached, every later frame should be bleached too
            assert np.all(state_of_each_frame[frames_bleached[0]:] == 2)
    # with bleaching rate 1 per second, (almost) all molecules should be bleached after 20 s
    assert np.mean(true_state[:, -1] == 2) > 0.95


def test_file_simulate_traces_makes_a_dataset(tmp_path):
    from papylio.file import File
    file = File.simulate_traces(tmp_path, 'simulated_test',
                                transition_rate_matrix=transition_rate_matrix_two_states,
                                intensity_of_each_state_per_channel=intensity_of_each_state_per_channel_two_states,
                                number_of_molecules=10, number_of_frames=50,
                                standard_deviation_of_intensity_noise=50, bleaching_rate=0.1, seed=1)
    assert tmp_path.joinpath('simulated_test.nc').exists()
    intensity = file.intensity
    assert intensity.dims == ('molecule', 'channel', 'frame')
    assert intensity.shape == (10, 2, 50)
    assert list(intensity.channel.values) == [0, 1]
    assert 'true_state' in intensity.coords

    # making it again with the same name should not overwrite the dataset
    with pytest.raises(FileExistsError):
        File.simulate_traces(tmp_path, 'simulated_test',
                             transition_rate_matrix=transition_rate_matrix_two_states,
                             intensity_of_each_state_per_channel=intensity_of_each_state_per_channel_two_states)
