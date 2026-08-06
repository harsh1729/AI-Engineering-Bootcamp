export const RAG_PIPELINE_CUSTOM = "custom";
export const RAG_PIPELINE_LANGCHAIN = "langchain";

export const DEFAULT_RAG_PIPELINE = RAG_PIPELINE_CUSTOM;

export const RAG_PIPELINES = [
  { value: RAG_PIPELINE_CUSTOM, label: "Custom RAG", enabled: true },
  { value: RAG_PIPELINE_LANGCHAIN, label: "LangChain RAG (Coming soon)", enabled: false },
];
