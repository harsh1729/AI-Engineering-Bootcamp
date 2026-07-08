from typing import Callable

from .time import get_current_time
from .weather import get_weather
from .currency import convert_currency

class ToolRegistry:

    _tools: dict[str, Callable] = {
        "get_current_time": get_current_time,
        "get_weather": get_weather,
        "convert_currency":convert_currency,
        
    }

    @classmethod
    def get(cls, tool_name: str) -> Callable:
        return cls._tools[tool_name]