import requests


class WeatherService:

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def get_weather(
        self,
        location: str,
    ) -> str:

        #raise TimeoutError("Weather API timed out.") #It was meant for testing tool exception handling
    
        latitude, longitude, resolved_location = self._get_coordinates(location)

        weather = self._get_current_weather(latitude, longitude)

        return (
            f"Location: {resolved_location}\n"
            f"Temperature: {weather['temperature']}°C\n"
            f"Wind Speed: {weather['windspeed']} km/h\n"
            f"Wind Direction: {weather['winddirection']}°\n"
            f"Weather Code: {weather['weathercode']}"
        )

    def _get_coordinates(
        self,
        location: str,
    ) -> tuple[float, float, str]:

        response = requests.get(
            self.GEOCODING_URL,
            params={
                "name": location,
                "count": 1,
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        if not data.get("results"):
            raise ValueError(f"Location '{location}' not found.")

        result = data["results"][0]

        return (
            result["latitude"],
            result["longitude"],
            result["name"],
        )

    def _get_current_weather(
        self,
        latitude: float,
        longitude: float,
    ) -> dict:

        response = requests.get(
            self.WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": True,
            },
            timeout=10,
        )

        response.raise_for_status()

        return response.json()["current_weather"]