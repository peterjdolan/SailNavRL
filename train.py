import boat as boat_lib
import environment
import gym
import numpy as np
import weather
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Initialize the environment
sailing_env = gym.make('SailingNavigationEnv-v0')

# Create the WeatherProvider (you can use the ConstantWindWeatherProvider or the MockWeatherProvider)
u_wind = 2.0
v_wind = 1.0
constant_wind_provider = weather.ConstantWindWeatherProvider(u_wind, v_wind)

# Initialize the boat with the WeatherProvider
initial_position = np.array([0.0, 0.0])
initial_heading = 0.0
boat = boat_lib.Boat(initial_position, initial_heading, constant_wind_provider)

# Set the boat in the environment
sailing_env.boat = boat

# Wrap the environment in a DummyVecEnv, which is required by Stable Baselines
vec_env = DummyVecEnv([lambda: sailing_env])

# Create the PPO agent
agent = PPO('MlpPolicy', vec_env, verbose=1)

# Train the agent for a number of timesteps
training_timesteps = 100000
agent.learn(total_timesteps=training_timesteps)

# Save the trained agent
agent.save("ppo_sailing_navigation")
# Load the trained agent and test it on the environment
loaded_agent = PPO.load("ppo_sailing_navigation")
obs = sailing_env.reset()

total_reward = 0

for step in range(sailing_env.max_steps):
    action, _ = loaded_agent.predict(obs, deterministic=True)
    obs, reward, done, info = sailing_env.step(action)
    total_reward += reward
    sailing_env.render()

    print(f"Step: {step + 1}, Action: {action}, Position: {sailing_env.boat.position}, "
          f"Wind Velocity: {sailing_env.boat.wind_velocity}, Reward: {reward}")

    if done:
        break

print(f"Total Reward: {total_reward}")
