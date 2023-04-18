import boat as boat_lib
import environment

import numpy as np
from geopy.distance import great_circle

def test_boat_update():
    # Test the Boat update method
    initial_position = np.array([0.0, 0.0])
    initial_heading = 0.0
    wind_velocity = np.array([2.0, 0.0])

    boat = boat_lib.Boat(initial_position, initial_heading, wind_velocity)

    # Test 1: Boat heading in the same direction as the wind
    action = np.array([0.0])
    dt = 1.0
    boat.update(action, dt)

    expected_velocity = np.array([2.0, 0.0])
    expected_position = initial_position + expected_velocity

    assert np.allclose(boat.velocity, expected_velocity), f"Expected {expected_velocity}, got {boat.velocity}"
    assert np.allclose(boat.position, expected_position), f"Expected {expected_position}, got {boat.position}"

    # Test 2: Boat heading perpendicular to the wind
    boat.position = initial_position
    action = np.array([np.pi / 2])
    boat.update(action, dt)

    expected_velocity = np.array([0.0, 3.0])
    expected_position = initial_position + expected_velocity

    assert np.allclose(boat.velocity, expected_velocity), f"Expected {expected_velocity}, got {boat.velocity}"
    assert np.allclose(boat.position, expected_position), f"Expected {expected_position}, got {boat.position}"

    # Test 3: Boat heading opposite to the wind
    boat.position = initial_position
    action = np.array([np.pi])
    boat.update(action, dt)

    expected_velocity = np.array([0.0, 0.0])
    expected_position = initial_position + expected_velocity

    assert np.allclose(boat.velocity, expected_velocity), f"Expected {expected_velocity}, got {boat.velocity}"
    assert np.allclose(boat.position, expected_position), f"Expected {expected_position}, got {boat.position}"

def test_environment_step():
    # Test the SailingNavigationEnv step method
    env = environment.SailingNavigationEnv()

    # Test 1: Boat moving in the same direction as the wind
    env.boat.wind_velocity = np.array([2.0, 0.0])
    action = np.array([0.0])
    dt = 1.0
    observation, reward, done, _ = env.step(action)

    expected_position = np.array([2.0, 0.0])
    expected_observation = np.array([2.0, 0.0, 0.0])

    assert np.allclose(observation, expected_observation), f"Expected {expected_observation}, got {observation}"

    # Test 2: Boat moving perpendicular to the wind
    env.reset()
    env.boat.wind_velocity = np.array([2.0, 0.0])
    action = np.array([np.pi / 2])
    observation, reward, done, _ = env.step(action)

    expected_position = np.array([0.0, 3.0])
    expected_observation = np.array([0.0, 3.0, np.pi / 2])

    assert np.allclose(observation, expected_observation), f"Expected {expected_observation}, got {observation}"
    # Test 3: Boat moving opposite to the wind
    env.reset()
    env.boat.wind_velocity = np.array([2.0, 0.0])
    action = np.array([np.pi])
    observation, reward, done, _ = env.step(action)

    expected_position = np.array([0.0, 0.0])
    expected_observation = np.array([0.0, 0.0, np.pi])

    assert np.allclose(observation, expected_observation), f"Expected {expected_observation}, got {observation}"

def test_environment_reset():
    # Test the SailingNavigationEnv reset method
    env = environment.SailingNavigationEnv()
    initial_observation = env.reset()

    expected_position = np.array([0.0, 0.0])
    expected_heading = np.array([0.0])
    expected_wind_velocity = np.array([0.0, 0.0])

    assert np.allclose(initial_observation[0], expected_position), f"Expected {expected_position}, got {initial_observation[0]}"
    assert np.allclose(initial_observation[1], expected_heading), f"Expected {expected_heading}, got {initial_observation[1]}"
    assert np.allclose(initial_observation[2], expected_wind_velocity), f"Expected {expected_wind_velocity}, got {initial_observation[2]}"

# Run the tests
test_boat_update()
test_environment_step()
test_environment_reset()

print("All tests passed!")
