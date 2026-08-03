import {
  CHUNKING_STRATEGIES,
  EMBEDDING_PROVIDERS,
  VECTOR_STORES,
} from "../constants/ragOptions";

export default function RagOptionsPanel({
  enabled,
  onEnabledChange,
  ragOptions,
  onRagOptionsChange,
  disabled = false,
}) {
  const updateOption = (field, value) => {
    onRagOptionsChange({ ...ragOptions, [field]: value });
  };

  return (
    <div className="rag-options-panel">
      <label className="rag-options-toggle">
        <input
          type="checkbox"
          checked={enabled}
          disabled={disabled}
          onChange={(event) => onEnabledChange(event.target.checked)}
        />
        <span>Advanced RAG options</span>
      </label>

      {enabled && (
        <div className="selector-row rag-options-row">
          <div className="selector-field">
            <label htmlFor="chunking-strategy-select">Chunking Strategy</label>
            <select
              id="chunking-strategy-select"
              value={ragOptions.chunking_strategy}
              disabled={disabled}
              onChange={(event) => updateOption("chunking_strategy", event.target.value)}
            >
              {CHUNKING_STRATEGIES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="selector-field">
            <label htmlFor="embedding-provider-select">Embedding Provider</label>
            <select
              id="embedding-provider-select"
              value={ragOptions.embedding_provider}
              disabled={disabled}
              onChange={(event) => updateOption("embedding_provider", event.target.value)}
            >
              {EMBEDDING_PROVIDERS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          <div className="selector-field">
            <label htmlFor="vector-store-select">Vector Store</label>
            <select
              id="vector-store-select"
              value={ragOptions.vector_store}
              disabled={disabled}
              onChange={(event) => updateOption("vector_store", event.target.value)}
            >
              {VECTOR_STORES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
