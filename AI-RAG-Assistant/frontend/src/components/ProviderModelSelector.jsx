import { PROVIDERS, getProvider } from "../constants/providers";

export default function ProviderModelSelector({
  providerValue,
  modelValue,
  onProviderChange,
  onModelChange,
}) {
  const models = getProvider(providerValue)?.models ?? [];

  return (
    <div className="selector-row">
      <div className="selector-field">
        <label htmlFor="provider-select">LLM Provider</label>
        <select
          id="provider-select"
          value={providerValue}
          onChange={(event) => onProviderChange(event.target.value)}
        >
          {PROVIDERS.map((provider) => (
            <option key={provider.value} value={provider.value}>
              {provider.label}
            </option>
          ))}
        </select>
      </div>

      <div className="selector-field">
        <label htmlFor="model-select">Model</label>
        <select
          id="model-select"
          value={modelValue}
          onChange={(event) => onModelChange(event.target.value)}
        >
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
