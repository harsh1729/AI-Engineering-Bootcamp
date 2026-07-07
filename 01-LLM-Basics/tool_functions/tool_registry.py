from typing import Callable

from .time_tools import get_current_time


class ToolRegistry:

    _tools: dict[str, Callable] = {
        "get_current_time": get_current_time,
    }

    @classmethod
    def get(cls, tool_name: str) -> Callable:
        return cls._tools[tool_name]