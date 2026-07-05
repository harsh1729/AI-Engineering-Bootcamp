from config import LLM_PROVIDER
from enums import ProviderType
from providers import LLMProvider,OpenAIProvider,ClaudeProvider



class ProviderFactory:

    PROVIDERS = {
    ProviderType.OPENAI: OpenAIProvider,
    ProviderType.CLAUDE:ClaudeProvider,
    }

    @staticmethod
    def create() -> LLMProvider:

        provider_class = ProviderFactory.PROVIDERS.get(LLM_PROVIDER)

        if provider_class is None:
            raise ValueError(
                f"Unsupported provider: {LLM_PROVIDER}"
            )

        return provider_class()