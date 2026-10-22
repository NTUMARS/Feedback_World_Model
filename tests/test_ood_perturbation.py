"""CPU unit tests for the robot-initial-state perturbation of the Robomimic OOD protocol (Eq. 45)."""
import numpy as np

from diffusion_policy.env_runner.robomimic_image_sequential_runner import PANDA_JOINT_LIMITS, perturb_robot_state

STATE_DIM = 1 + 2 * 9 + 20  # time, two robots (7 arm + 2 gripper joints each), objects / velocities


def _nominal_state():
    state = np.zeros(STATE_DIM)
    state[1:8] = PANDA_JOINT_LIMITS.mean(axis=1)      # robot 0 arm joints in the middle of their range
    state[10:17] = PANDA_JOINT_LIMITS.mean(axis=1)    # robot 1
    state[8:10] = 0.04                                # gripper fingers
    state[17:19] = 0.04
    state[19:] = np.arange(STATE_DIM - 19)            # object poses and velocities
    return state


def test_only_arm_joints_change_and_input_is_not_mutated():
    state = _nominal_state()
    original = state.copy()
    perturbed = perturb_robot_state(state, noise_scale=0.1, rng=np.random.RandomState(0))
    assert np.array_equal(state, original)
    changed = np.nonzero(perturbed != original)[0]
    assert set(changed) <= set(range(1, 8)) and len(changed) == 7
    deltas = perturbed[1:8] - original[1:8]
    assert 0.02 < np.abs(deltas).mean() < 0.3       # sigma = 0.1 rad


def test_dual_arm_perturbs_both_robots_independently():
    state = _nominal_state()
    perturbed = perturb_robot_state(state, noise_scale=0.1, rng=np.random.RandomState(1), n_robots=2)
    changed = set(np.nonzero(perturbed != state)[0])
    assert changed == set(range(1, 8)) | set(range(10, 17))
    assert not np.allclose(perturbed[1:8] - state[1:8], perturbed[10:17] - state[10:17])


def test_perturbed_joints_stay_within_limits():
    state = _nominal_state()
    perturbed = perturb_robot_state(state, noise_scale=10.0, rng=np.random.RandomState(2))
    assert np.all(perturbed[1:8] >= PANDA_JOINT_LIMITS[:, 0]) and np.all(perturbed[1:8] <= PANDA_JOINT_LIMITS[:, 1])


def test_same_seed_gives_same_initial_state_and_zero_scale_is_identity():
    state = _nominal_state()
    a = perturb_robot_state(state, 0.1, np.random.RandomState(100000))
    b = perturb_robot_state(state, 0.1, np.random.RandomState(100000))
    c = perturb_robot_state(state, 0.1, np.random.RandomState(100001))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert np.array_equal(perturb_robot_state(state, 0.0, np.random.RandomState(0)), state)
