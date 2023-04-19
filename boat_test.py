import pytest

import weather as weather_lib
import boat as boat_lib

import numpy as np
from shapely.geometry import Point
import xarray as xr

def test_boat_update():
    # Test the Boat update method
    initial_position = np.array([0.0, 0.0])
    initial_heading = 0.0

    boat = boat_lib.Boat(initial_position, initial_heading, weather_lib.ConstantWindWeatherProvider(2.0, 0.0))

    # Test 1: Boat heading in the same direction as the wind
    action = np.array([0.0])
    dt = 1.0
    boat.update(action, dt)

    expected_velocity = np.array([2.0, 0.0])
    expected_position = initial_position + expected_velocity

    assert np.allclose(boat.position, expected_position), f"Expected {expected_position}, got {boat.position}"

    # Test 2: Boat heading perpendicular to the wind
    boat.position = initial_position
    action = np.array([np.pi / 2])
    boat.update(action, dt)

    expected_velocity = np.array([0.0, 3.0])
    expected_position = initial_position + expected_velocity

    assert np.allclose(boat.position, expected_position), f"Expected {expected_position}, got {boat.position}"

    # Test 3: Boat heading opposite to the wind
    boat.position = initial_position
    action = np.array([np.pi])
    boat.update(action, dt)

    expected_velocity = np.array([0.0, 0.0])
    expected_position = initial_position + expected_velocity

    assert np.allclose(boat.position, expected_position), f"Expected {expected_position}, got {boat.position}"
