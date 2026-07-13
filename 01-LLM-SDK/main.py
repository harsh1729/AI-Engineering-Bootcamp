
from factories import ProviderFactory
from models import LLMRequest,LLMMessage
from enums import MessageRole
from tool_functions import CURRENT_TIME_TOOL,WEATHER_TOOL,CONVERT_CURRENCY_TOOL

provider = ProviderFactory.create()

request = LLMRequest(
    messages=[
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="You are a spritual teacher.",
        ),
        
        LLMMessage(
            role=MessageRole.USER,
            content="Explain what is atman in 100 words.",
        ),
    ],
    temperature=0.2,
    max_tokens=800,
    tools=[
        CURRENT_TIME_TOOL,
        WEATHER_TOOL,
        CONVERT_CURRENCY_TOOL,
    ],
)


# Code for complete response as one
# response = provider.generate_response(request)
# if response is not None and response.text:
#     print(response.text)

#     print("finish_reason:", response.finish_reason)
#     print("usage:", response.usage)
#     print("thoughts:", getattr(response.raw_response.usage_metadata, "thoughts_token_count", None))
#     print("words:", len((response.text or "").split()))


#Code for stream message
for chunk in provider.generate_stream(request):
    if chunk.text is not None:
        print(chunk.text, end="",flush=True)