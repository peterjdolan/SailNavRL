import boat as boat_lib
import gym
import numpy as np
from geopy.distance import great_circle
import numpy as np

class SailingNavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SailingNavigationEnv, self).__init__()

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-np.pi]), high=np.array([np.pi]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 2), dtype=np.float32)

        # Define environment parameters
        self.destination = np.array([0.0, 100.0])  # Destination coordinates
        self.max_steps = 200  # Maximum steps per episode

        self.reset()

    def reset(self):
        # Reset the environment to the initial state
        initial_position = np.array([0.0, 0.0])
        initial_heading = 0.0
        wind_velocity = np.array([0.0, 0.0])
        self.boat = boat_lib.Boat(initial_position, initial_heading, wind_velocity)
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        dt = 1.0

        # Update the boat state based on the action
        self.boat.update(action, dt)

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
        # Implement rendering logic if needed
        pass

    def close(self):
        # Clean up resources if needed
        pass
