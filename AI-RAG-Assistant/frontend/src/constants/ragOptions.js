export const DEFAULT_RAG_OPTIONS = {
  chunking_strategy: "recursive",
  embedding_provider: "openai",
  vector_store: "chroma",
};

export const CHUNKING_STRATEGIES = [
  { value: "recursive", label: "Recursive" },
  { value: "character", label: "Character" },
  { value: "sentence", label: "Sentence" },
  { value: "header_aware", label: "Header Aware" },
];

export const EMBEDDING_PROVIDERS = [
  { value: "openai", label: "OpenAI" },
  { value: "voyage", label: "Voyage AI" },
  { value: "cohere", label: "Cohere" },
];

export const VECTOR_STORES = [{ value: "chroma", label: "Chroma" }];
