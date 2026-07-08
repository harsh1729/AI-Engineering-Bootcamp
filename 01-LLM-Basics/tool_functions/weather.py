from models.tools import LLMTool, LLMToolParam

from services import WeatherService

weather_service = WeatherService()


def get_weather(
    location: str,
) -> str:

    return weather_service.get_weather(location)


WEATHER_TOOL = LLMTool(
    name="get_weather",
    description="Returns the current weather for the specified location.",
    parameters=[
        LLMToolParam(
            name="location",
            type="string",
            description="Location such as Jaipur, Delhi, London or New York.",
        ),
    ],
)