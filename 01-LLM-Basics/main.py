
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
            content="What's the weather and current time in Delhi? Also whats doolar in their currency?",
        ),
    ],
    temperature=0.2,
    max_tokens=300,
    tools=[
        CURRENT_TIME_TOOL,
        WEATHER_TOOL,
        CONVERT_CURRENCY_TOOL,
    ],
)


# Code for complete response as one
# response = provider.generate(request)

# print(response.text)


# Code for stream message
for chunk in provider.generate_stream(request):
    print(chunk.text, end="", flush=True)