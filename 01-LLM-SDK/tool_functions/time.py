from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from models.tools import LLMTool, LLMToolParam

import time

def get_current_time(
    timezone: str,
    time_format: Literal["12h", "24h"] = "24h",
) -> str:
    """
    Return the current date and time for the given IANA timezone.
    """

    current_time = datetime.now(ZoneInfo(timezone))

    if time_format == "12h":
        return current_time.strftime("%Y-%m-%d %I:%M:%S %p")
    
    time.sleep(11)

    return current_time.strftime("%Y-%m-%d %H:%M:%S")


CURRENT_TIME_TOOL = LLMTool(
    name="get_current_time",
    description="Returns the current date and time for the specified timezone.",
    parameters=[
        LLMToolParam(
            name="timezone",
            type="string",
            description="IANA timezone such as Asia/Kolkata, Europe/London or America/New_York.",
        ),
        LLMToolParam(
            name="time_format",
            type="string",
            description="Time format. Allowed values are '12h' or '24h'.",
            required=False,
        ),
    ],
)