// Hardcoded providers and models for the first frontend milestone.
// `value` must match the backend's ProviderType values (llm_sdk.enums.ProviderType).
export const PROVIDERS = [
  {
    label: "ChatGPT",
    value: "openai",
    models: ["gpt-4.1-mini","gpt-5-mini","gpt-5"],
  },
  {
    label: "Gemini",
    value: "gemini",
    models: ["gemini-3.5-flash", "gemini-3.5-pro"],
  },
  {
    label: "Claude",
    value: "claude",
    models: ["claude-opus-4-8"],
  },
];

export const DEFAULT_PROVIDER = PROVIDERS[0];
export const DEFAULT_MODEL = DEFAULT_PROVIDER.models[0];

export function getProvider(providerValue) {
  return PROVIDERS.find((provider) => provider.value === providerValue);
}
