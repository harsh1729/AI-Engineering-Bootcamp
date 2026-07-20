from llm_sdk.config import LLM_PROVIDER
from llm_sdk.enums import ProviderType
from llm_sdk.providers import LLMProvider, OpenAIProvider, ClaudeProvider, GeminiProvider



class ProviderFactory:

    PROVIDERS = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.CLAUDE: ClaudeProvider,
        ProviderType.GEMINI: GeminiProvider,
    }

    @staticmethod
    def create(provider: ProviderType | None = None) -> LLMProvider:

        selected_provider = provider or LLM_PROVIDER

        provider_class = ProviderFactory.PROVIDERS.get(selected_provider)

        if provider_class is None:
            raise ValueError(
                f"Unsupported provider: {selected_provider}"
            )

        return provider_class()