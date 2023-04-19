import boat as boat_lib
import weather as weather_lib
import gym
import matplotlib as plt
import numpy as np
import geopandas as gpd
from geopy.distance import great_circle
from shapely.geometry import Point


class SailingNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 geojson_filepath='./ne_110m_coastline.geojson',
                 weather_provider=weather_lib.MockWeatherProvider()):
        super(SailingNavigationEnv, self).__init__()

        # Load geographical data
        self.coastlines = gpd.read_file(geojson_filepath)

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float64)

        obs_low = np.array([-90.0, -180.0, -np.pi])
        obs_high = np.array([90.0, 180.0, np.pi])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

        # Define environment parameters
        self.destination = np.array([0.0, 100.0])  # Destination coordinates
        self.max_steps = 200  # Maximum steps per episode

        # Initialize the weather provider
        self.weather_provider = weather_provider

        self.reset()

    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        initial_position = np.array([0.0, 0.0])
        initial_heading = 0.0

        self.boat = boat_lib.Boat(initial_position, initial_heading, self.weather_provider)
        self.steps = 0

        return self._get_observation()

    def step(self, action):
        dt = 1.0

        # Update the boat state based on the action
        self.boat.update(action, dt)

        # Check for collision with land
        boat_point = Point(self.boat.position)
        if not self._is_point_on_water(boat_point):
            # Implement collision handling logic, e.g., end the episode
            pass

        # Calculate the reward based on the updated state
        reward = self.compute_reward()

        # Check if the episode should terminate
        done = self.is_terminal()

        # Increment step counter
        self.steps += 1

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return np.concatenate((self.boat.position, np.array([self.boat.heading])))

    def compute_reward(self):
        # Calculate the reward based on the current state
        current_distance = great_circle(self.boat.position, self.destination).m
        return -current_distance

    def is_terminal(self):
        # Check if the episode should end based on the current state
        current_distance = great_circle(self.boat.position, self.destination).m
        return (self.steps >= self.max_steps) or (current_distance < 1)

    def render(self, mode='human', close=False):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-180, 180)
            self.ax.set_ylim(-90, 90)
            self.ax.set_aspect('equal', 'box')
            self.points, = self.ax.plot([], [], 'bo', ms=5)

        x, y = self.boat.position
        xs, ys = self.points.get_data()
        xs = np.append(xs, x)
        ys = np.append(ys, y)
        self.points.set_data(xs, ys)
        plt.pause(0.01)

    def close(self):
        # Clean up resources if needed
        pass

    def _is_point_on_water(self, point):
        # Check if the given point is on water by looking for intersections with coastlines
        for _, row in self.coastlines.iterrows():
            if row['geometry'].contains(point):
                return False
        return True

gym.envs.registration.register(
    id='SailingNavigationEnv-v0',
    entry_point='environment:SailingNavigationEnv',
)