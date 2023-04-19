import numpy as np
from shapely.geometry import Point

class Boat:
    def __init__(self, initial_position, initial_heading, weather_provider):
        self.position = initial_position
        self.heading = initial_heading
        self.weather_provider = weather_provider

    def update(self, action, dt):
        # Update the boat state based on the action
        # Set the boat's heading
        self.heading = action[0]

        # Update the wind velocity based on the current position
        wind_data = self.weather_provider.get_current_weather(
            Point(self.position[0], self.position[1]))
        wind_velocity = np.array([wind_data['u_wind'], wind_data['v_wind']])

        # Compute the boat's velocity
        velocity = self.compute_velocity(wind_velocity)

        # Update the boat's position based on its velocity and time step
        displacement = velocity * dt
        self.position = self.position + displacement

    def compute_velocity(self, wind_velocity):
        # Compute the angle between the boat's heading and the wind's direction
        wind_speed = np.linalg.norm(wind_velocity)
        wind_heading = np.arctan2(wind_velocity[1], wind_velocity[0])

        angle_diff = self.heading - wind_heading
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # Compute the boat's speed based on the angle
        if -np.pi/2 <= angle_diff <= np.pi/2:
            speed = wind_speed * (1.0 + 0.5 * np.sin(angle_diff))
        else:
            speed = wind_speed * np.sin(angle_diff)

        # Compute the boat's velocity components
        velocity = np.array([speed * np.cos(self.heading), speed * np.sin(self.heading)])

        return velocity