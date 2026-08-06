import {
  CHUNKING_STRATEGIES,
  EMBEDDING_PROVIDERS,
  VECTOR_STORES,
} from "../constants/ragOptions";

export default function RagOptionsPanel({
  ragOptions,
  onRagOptionsChange,
  disabled = false,
  chunkingStrategies = CHUNKING_STRATEGIES,
  embeddingProviders = EMBEDDING_PROVIDERS,
  vectorStores = VECTOR_STORES,
  embeddingModels = {},
}) {
  const updateOption = (field, value) => {
    onRagOptionsChange({ ...ragOptions, [field]: value });
  };

  const selectedEmbeddingModel = embeddingModels[ragOptions.embedding_provider];

  return (
    <div className="rag-options-panel">
      <div className="selector-row rag-options-row">
        <div className="selector-field">
          <label htmlFor="chunking-strategy-select">Chunking Strategy</label>
          <select
            id="chunking-strategy-select"
            value={ragOptions.chunking_strategy}
            disabled={disabled}
            onChange={(event) => updateOption("chunking_strategy", event.target.value)}
          >
            {chunkingStrategies.map((option) => (
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
            {embeddingProviders.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          {selectedEmbeddingModel ? (
            <span className="rag-option-model-hint">Model: {selectedEmbeddingModel}</span>
          ) : null}
        </div>

        <div className="selector-field">
          <label htmlFor="vector-store-select">Vector Store</label>
          <select
            id="vector-store-select"
            value={ragOptions.vector_store}
            disabled={disabled}
            onChange={(event) => updateOption("vector_store", event.target.value)}
          >
            {vectorStores.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}
