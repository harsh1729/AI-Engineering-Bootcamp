export const DEFAULT_RAG_OPTIONS = {
  chunking_strategy: "recursive",
  embedding_provider: "openai",
  vector_store: "chroma",
};

export const CHUNKING_STRATEGIES = [
  { value: "recursive", label: "Recursive" },
  { value: "character", label: "Character" },
];

export const EMBEDDING_PROVIDERS = [
  { value: "openai", label: "OpenAI" },
];

export const VECTOR_STORES = [{ value: "chroma", label: "Chroma" }];
