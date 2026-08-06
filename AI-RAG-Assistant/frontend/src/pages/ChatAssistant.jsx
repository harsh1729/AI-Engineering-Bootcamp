import { useEffect, useState } from "react";
import ProviderModelSelector from "../components/ProviderModelSelector";
import RagOptionsPanel from "../components/RagOptionsPanel";
import ChatHistory from "../components/ChatHistory";
import MessageInput from "../components/MessageInput";
import { fetchRagOptionsCatalog } from "../api/ragOptionsApi";
import { DEFAULT_PROVIDER, DEFAULT_MODEL, getProvider } from "../constants/providers";
import { DEFAULT_RAG_PIPELINE } from "../constants/ragPipeline";
import { DEFAULT_RAG_OPTIONS } from "../constants/ragOptions";
import { sendChatMessage, sendRagChatMessage } from "../api/chatApi";
import "./ChatAssistant.css";

function ChatAssistant() {
  const [provider, setProvider] = useState(DEFAULT_PROVIDER.value);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [ragPipeline, setRagPipeline] = useState(DEFAULT_RAG_PIPELINE);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState("Thinking...");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [ragOptions, setRagOptions] = useState(DEFAULT_RAG_OPTIONS);
  const [ragCatalog, setRagCatalog] = useState(null);

  useEffect(() => {
    fetchRagOptionsCatalog()
      .then((catalog) => {
        setRagCatalog(catalog);
        setRagOptions(catalog.defaults ?? DEFAULT_RAG_OPTIONS);
      })
      .catch(() => {
        setRagCatalog(null);
      });
  }, []);

  const handleProviderChange = (nextProvider) => {
    if (nextProvider === provider) {
      return;
    }

    if (messages.length > 0) {
      const nextProviderLabel = getProvider(nextProvider)?.label ?? nextProvider;
      const confirmed = window.confirm(
        `Switching to ${nextProviderLabel} starts a new conversation. ` +
          "Your current chat will be cleared. Continue?"
      );
      if (!confirmed) {
        return;
      }
      setMessages([]);
    }

    setProvider(nextProvider);
    setModel(getProvider(nextProvider).models[0]);
  };

  const appendToLastMessage = (chunk) => {
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, content: last.content + chunk };
      return next;
    });
  };

  const toConversationHistory = (uiMessages) =>
    uiMessages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .filter((message) => message.content)
      .map((message) => ({ role: message.role, content: message.content }));

  const handleSend = async (text, documentIds, attachmentRagOptions) => {
    const hasDocuments = documentIds?.length > 0;
    const activeRagOptions = showAdvancedOptions
      ? attachmentRagOptions ?? ragOptions
      : null;
    const conversation = [...toConversationHistory(messages), { role: "user", content: text }];

    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);
    setIsLoading(true);
    setLoadingLabel(hasDocuments ? "Answering..." : "Thinking...");

    try {
      if (hasDocuments) {
        const result = await sendRagChatMessage({
          provider,
          model,
          messages: conversation,
          documentIds,
          ragOptions: activeRagOptions,
        });
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: result.response,
            sources: result.sources ?? [],
            warning: result.warning ?? null,
          };
          return next;
        });
        return;
      }

      await sendChatMessage({
        provider,
        model,
        messages: conversation,
        documentIds,
        onChunk: appendToLastMessage,
      });
    } catch (error) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = last.content
          ? { ...last, warning: error.message }
          : { role: "error", content: error.message };
        return next;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="app-config">
        <h2 className="app-config-heading">Custom Configuration</h2>

        <ProviderModelSelector
          providerValue={provider}
          modelValue={model}
          ragPipelineValue={ragPipeline}
          disabled={isLoading}
          onProviderChange={handleProviderChange}
          onModelChange={setModel}
          onRagPipelineChange={setRagPipeline}
        />

        <div className="advanced-options-bar">
          <button
            type="button"
            className="advanced-options-toggle"
            onClick={() => setShowAdvancedOptions((open) => !open)}
            disabled={isLoading}
            aria-expanded={showAdvancedOptions}
          >
            {showAdvancedOptions ? (
            <>
              <span className="advanced-options-toggle-label advanced-options-toggle-label-desktop">
                Hide Advanced RAG Options
              </span>
              <span className="advanced-options-toggle-label advanced-options-toggle-label-mobile">
                Hide RAG options
              </span>
            </>
          ) : (
            <>
              <span className="advanced-options-toggle-label advanced-options-toggle-label-desktop">
                Show Advanced RAG Options
              </span>
              <span className="advanced-options-toggle-label advanced-options-toggle-label-mobile">
                Show RAG options
              </span>
            </>
          )}
          </button>
          <span className="rag-options-hint rag-options-hint-desktop">
            Configure before uploading documents.
          </span>
          <span className="rag-options-hint rag-options-hint-mobile">(configure before upload)</span>
        </div>

        {showAdvancedOptions && (
          <RagOptionsPanel
            ragOptions={ragOptions}
            onRagOptionsChange={setRagOptions}
            disabled={isLoading}
            chunkingStrategies={ragCatalog?.chunking_strategies}
            embeddingProviders={ragCatalog?.embedding_providers}
            vectorStores={ragCatalog?.vector_stores}
            embeddingModels={ragCatalog?.embedding_models ?? {}}
          />
        )}
      </div>

      <div className="app-conversation">
        <ChatHistory messages={messages} isLoading={isLoading} loadingLabel={loadingLabel} />

        <MessageInput
          onSend={handleSend}
          disabled={isLoading}
          loadingLabel={loadingLabel}
          ragOptions={showAdvancedOptions ? ragOptions : null}
          onAttachmentsChange={() => {}}
        />
      </div>
    </div>
  );
}

export default ChatAssistant;
