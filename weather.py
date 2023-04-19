import xarray as xr
import numpy as np
from shapely.geometry import Point
from scipy.ndimage import gaussian_filter

class WeatherProvider:
    def __init__(self):
        pass

    def get_current_weather(self, location: Point):
        raise NotImplementedError

    def get_weather_forecast(self, min_corner: Point, max_corner: Point):
        raise NotImplementedError


class MockWeatherProvider(WeatherProvider):
    def __init__(self):
        super().__init__()

        # Generate a 0.25-degree spaced grid for the whole world
        latitudes = np.arange(-90, 90, 0.25)
        longitudes = np.arange(-180, 180, 0.25)

        # Create mock weather data for the grid
        temperature = np.random.uniform(0, 40, size=(len(latitudes), len(longitudes)))
        u_wind = np.random.uniform(-20, 20, size=(len(latitudes), len(longitudes)))
        v_wind = np.random.uniform(-20, 20, size=(len(latitudes), len(longitudes)))

        # Apply Gaussian filter to smooth the data
        sigma = 3  # Determines the degree of smoothing
        temperature_smoothed = gaussian_filter(temperature, sigma)
        u_wind_smoothed = gaussian_filter(u_wind, sigma)
        v_wind_smoothed = gaussian_filter(v_wind, sigma)

        # Create an xarray.Dataset with the smoothed weather data
        self.weather_forecast = xr.Dataset(
            {
                "temperature": (("latitude", "longitude"), temperature_smoothed),
                "u_wind": (("latitude", "longitude"), u_wind_smoothed),
                "v_wind": (("latitude", "longitude"), v_wind_smoothed),
            },
            coords={
                "latitude": latitudes,
                "longitude": longitudes,
            },
        )

    def get_current_weather(self, location: Point):
        # Interpolate weather data for the given location from the weather forecast
        current_weather = self.weather_forecast.interp(latitude=location.y, longitude=location.x)
        
        return {
            'temperature': current_weather['temperature'].item(),
            'u_wind': current_weather['u_wind'].item(),
            'v_wind': current_weather['v_wind'].item(),
        }

    def get_weather_forecast(self, min_corner: Point, max_corner: Point):
        # Select the region of interest from the global weather forecast
        region_forecast = self.weather_forecast.sel(
            latitude=slice(min_corner.y, max_corner.y),
            longitude=slice(min_corner.x, max_corner.x)
        )
        return region_forecast
