import pytest

import weather as weather_lib
import boat as boat_lib
import environment

import numpy as np
from shapely.geometry import Point
import xarray as xr


def test_environment_step():
    # Test the SailingNavigationEnv step method
    env = environment.SailingNavigationEnv(weather_provider=weather_lib.ConstantWindWeatherProvider(2.0, 0.0))

    # Test 1: Boat moving in the same direction as the wind
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

    assert np.allclose(initial_observation[0], expected_position), f"Expected {expected_position}, got {initial_observation[0]}"
    assert np.allclose(initial_observation[1], expected_heading), f"Expected {expected_heading}, got {initial_observation[1]}"

def test_is_point_on_water():
    env = environment.SailingNavigationEnv("./ne_50m_coastline.geojson")

    # Points known to be on land and water
    land_point = Point(-98.5795, 39.8283)  # Geographic center of the contiguous United States
    water_point = Point(0, 0)  # Atlantic Ocean

    assert env._is_point_on_water(water_point), "Water point should be on water"
    assert not env._is_point_on_water(land_point), "Land point should not be on water"
