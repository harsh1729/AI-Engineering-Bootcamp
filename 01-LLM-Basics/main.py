
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
            content="Answer from the teachings of Ramana Maharishi in Hindi.",
        ),
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="Keep the explanation under 100 words.",
        ),
        LLMMessage(
            role=MessageRole.SYSTEM,
            content="Use simple language suitable for a 12-year-old.",
        ),
        LLMMessage(
            role=MessageRole.USER,
            content="Explain Brahman in Advaita Vedanta.",
        ),
    ],
    temperature=0.2,
    max_tokens=300,
)

response = provider.generate(request)

print(response.text)
print()

print(f"Finish Reason : {response.finish_reason}")
print(f"Input Tokens  : {response.input_tokens}")
print(f"Output Tokens : {response.output_tokens}")
print(f"Total Tokens  : {response.total_tokens}")