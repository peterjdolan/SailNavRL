import numpy as np

class Boat:
    def __init__(self, initial_position, heading, wind_velocity):
        self.position = initial_position
        self.velocity = np.array([0.0, 0.0])
        self.heading = heading
        self.wind_velocity = wind_velocity

    def update(self, action, dt):
        # Update the boat state based on the action
        # Set the boat's heading
        self.heading = action[0]

        # Compute the boat's velocity
        self.velocity = self.compute_velocity()

        # Update the boat's position based on its velocity and time step
        displacement = self.velocity * dt
        self.position = self.position + displacement

    def compute_velocity(self):
        # Compute the angle between the boat's heading and the wind's direction
        wind_speed = np.linalg.norm(self.wind_velocity)
        wind_heading = np.arctan2(self.wind_velocity[1], self.wind_velocity[0])

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