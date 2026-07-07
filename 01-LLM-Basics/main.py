
from factories import ProviderFactory
from models import LLMRequest,LLMMessage
from enums import MessageRole
from tool_functions import CURRENT_TIME_TOOL,ToolRegistry

provider = ProviderFactory.create()

request = LLMRequest(
    messages=[
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="You are a time keeper in India.",
        ),
        
        LLMMessage(
            role=MessageRole.USER,
            content="What is current time in 12 hours format?",
        ),
    ],
    temperature=0.2,
    max_tokens=300,
    tools=[
        CURRENT_TIME_TOOL,
    ],
)

response = provider.generate(request)

if response.tool_calls:
    tool_call = response.tool_calls[0]

    tool = ToolRegistry.get(tool_call.name)

    result = tool(**tool_call.arguments)

    print(result)
else:
    print(response.text)


#Code for stream message
# for chunk in provider.generate_stream(request):
#     print(chunk.text, end="", flush=True)