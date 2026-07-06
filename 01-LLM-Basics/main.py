
from factories import ProviderFactory
from models import LLMRequest,LLMMessage
from enums import MessageRole

provider = ProviderFactory.create()

request = LLMRequest(
    messages=[
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="You are a spiritual expert.",
        ),
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="Keep the explanation under 100 words.",
        ),
        LLMMessage(
            role=MessageRole.USER,
            content="What is self realization.",
        ),
    ],
    temperature=0.2,
    max_tokens=300,
)

# response = provider.generate(request)

# print(response.text)
# print(f"Finish Reason : {response.finish_reason}")
# print(f"LLMUsage  : {response.usage}")

for chunk in provider.generate_stream(request):
    print(chunk.text, end="", flush=True)