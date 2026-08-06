const RAG_OPTIONS_ENDPOINT = "http://127.0.0.1:8000/rag/options";

export async function fetchRagOptionsCatalog() {
  const response = await fetch(RAG_OPTIONS_ENDPOINT);

  if (!response.ok) {
    throw new Error("Could not load RAG options.");
  }

  return response.json();
}
