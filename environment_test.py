import pytest

import weather as weather_lib
import boat as boat_lib
import environment

import numpy as np
from geopy.distance import great_circle
from shapely.geometry import Point
import xarray as xr

class ConstantWindWeatherProvider(weather_lib.WeatherProvider):
    def __init__(self, u_wind, v_wind, temperature=20):
        self.u_wind = u_wind
        self.v_wind = v_wind
        self.temperature = temperature

    def get_current_weather(self, point):
        return {
            'u_wind': self.u_wind,
            'v_wind': self.v_wind,
            'temperature': self.temperature
        }

    def get_weather_forecast(self, min_corner, max_corner, resolution=0.25):
        lat_range = np.arange(min_corner.y, max_corner.y, resolution)
        lon_range = np.arange(min_corner.x, max_corner.x, resolution)

        data = {
            'u_wind': (['latitude', 'longitude'], np.full((len(lat_range), len(lon_range)), self.u_wind)),
            'v_wind': (['latitude', 'longitude'], np.full((len(lat_range), len(lon_range)), self.v_wind)),
            'temperature': (['latitude', 'longitude'], np.full((len(lat_range), len(lon_range)), self.temperature)),
        }

        coords = {
            'latitude': lat_range,
            'longitude': lon_range,
        }

        return xr.Dataset(data, coords)


def test_boat_update():
    # Test the Boat update method
    initial_position = np.array([0.0, 0.0])
    initial_heading = 0.0

    boat = boat_lib.Boat(initial_position, initial_heading, ConstantWindWeatherProvider(2.0, 0.0))

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

def test_environment_step():
    # Test the SailingNavigationEnv step method
    env = environment.SailingNavigationEnv(weather_provider=ConstantWindWeatherProvider(2.0, 0.0))

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
