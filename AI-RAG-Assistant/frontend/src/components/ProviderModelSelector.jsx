import { PROVIDERS, getProvider } from "../constants/providers";
import { RAG_PIPELINES } from "../constants/ragPipeline";

export default function ProviderModelSelector({
  providerValue,
  modelValue,
  ragPipelineValue,
  disabled = false,
  onProviderChange,
  onModelChange,
  onRagPipelineChange,
}) {
  const models = getProvider(providerValue)?.models ?? [];

  return (
    <div className="selector-row">
      <div className="selector-field">
        <label htmlFor="provider-select">LLM Provider</label>
        <select
          id="provider-select"
          value={providerValue}
          disabled={disabled}
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
        <label htmlFor="model-select">LLM Model</label>
        <select
          id="model-select"
          value={modelValue}
          disabled={disabled}
          onChange={(event) => onModelChange(event.target.value)}
        >
          {models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      </div>

      <div className="selector-field">
        <label htmlFor="rag-pipeline-select">RAG Pipeline</label>
        <select
          id="rag-pipeline-select"
          value={ragPipelineValue}
          disabled={disabled}
          onChange={(event) => onRagPipelineChange(event.target.value)}
        >
          {RAG_PIPELINES.map((pipeline) => (
            <option key={pipeline.value} value={pipeline.value} disabled={!pipeline.enabled}>
              {pipeline.label}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
