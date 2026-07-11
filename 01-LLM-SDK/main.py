
from factories import ProviderFactory
from models import LLMRequest,LLMMessage
from enums import MessageRole
from tool_functions import CURRENT_TIME_TOOL,WEATHER_TOOL,CONVERT_CURRENCY_TOOL

provider = ProviderFactory.create()

request = LLMRequest(
    messages=[
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="You are an assistent.",
        ),
        
        LLMMessage(
            role=MessageRole.USER,
            content="Tell time , weather and currency conversion of London?",
        ),
    ],
    temperature=0.2,
    max_tokens=400,
    tools=[
        CURRENT_TIME_TOOL,
        WEATHER_TOOL,
        CONVERT_CURRENCY_TOOL,
    ],
)


# Code for complete response as one
response = provider.generate_response(request)
if response is not None and response.text:
    print(response.text)


#Code for stream message
# for chunk in provider.generate_stream(request):
#     if chunk.text is not None:
#         print(chunk.text, end="")