import { useCallback, useEffect, useState } from "react";
import ProviderModelSelector from "../components/ProviderModelSelector";
import RagOptionsPanel from "../components/RagOptionsPanel";
import ChatHistory from "../components/ChatHistory";
import MessageInput from "../components/MessageInput";
import { sendChatMessage } from "../api/chatApi";
import { useChatSession } from "../context/ChatSessionContext";
import { fetchRagOptionsCatalog } from "../api/ragOptionsApi";
import { DEFAULT_PROVIDER, DEFAULT_MODEL, getProvider } from "../constants/providers";
import { DEFAULT_RAG_PIPELINE } from "../constants/ragPipeline";
import { DEFAULT_RAG_OPTIONS } from "../constants/ragOptions";
import "./ChatAssistant.css";

function ChatAssistant() {
  const [provider, setProvider] = useState(DEFAULT_PROVIDER.value);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [ragPipeline, setRagPipeline] = useState(DEFAULT_RAG_PIPELINE);
  const [loadingLabel, setLoadingLabel] = useState("Thinking...");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [ragOptions, setRagOptions] = useState(DEFAULT_RAG_OPTIONS);
  const [ragCatalog, setRagCatalog] = useState(null);
  const [ragOptionsLocked, setRagOptionsLocked] = useState(false);
  const [inputResetKey, setInputResetKey] = useState(0);

  const {
    activeChatId,
    messages,
    setMessages,
    isLoadingSessions,
    isLoadingMessages,
    isSending,
    setIsSending,
    sessionBusy,
    setChatProvider,
    refreshChats,
    startNewChat,
    setActiveChatId,
  } = useChatSession();

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

  useEffect(() => {
    setChatProvider(provider);
  }, [provider, setChatProvider]);

  const handleProviderChange = (nextProvider) => {
    if (nextProvider === provider) {
      return;
    }

    const nextProviderLabel = getProvider(nextProvider)?.label ?? nextProvider;
    const confirmed = window.confirm(
      `Switch to ${nextProviderLabel}? Each LLM provider keeps its own chat history.`,
    );
    if (!confirmed) {
      return;
    }

    setProvider(nextProvider);
    setModel(getProvider(nextProvider).models[0]);
  };

  const appendToLastMessage = useCallback(
    (chunk) => {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, content: last.content + chunk };
        return next;
      });
    },
    [setMessages],
  );

  const applySourcesToLastMessage = useCallback(
    (sources) => {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, sources };
        return next;
      });
    },
    [setMessages],
  );

  const toConversationHistory = (uiMessages) =>
    uiMessages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .filter((message) => message.content)
      .map((message) => ({ role: message.role, content: message.content }));

  const handleAttachmentsChange = useCallback((hasAttachments, attachments) => {
    setRagOptionsLocked(hasAttachments);
    const lockedOptions = attachments[0]?.ragOptions;
    if (hasAttachments && lockedOptions) {
      setRagOptions(lockedOptions);
    }
  }, []);

  const handleStartNewChat = useCallback(() => {
    startNewChat();
    setRagOptionsLocked(false);
    setInputResetKey((key) => key + 1);
  }, [startNewChat]);

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
    setIsSending(true);
    setLoadingLabel(hasDocuments ? "Answering..." : "Thinking...");

    try {
      const result = await sendChatMessage({
        provider,
        model,
        messages: conversation,
        documentIds: documentIds ?? [],
        ragOptions: activeRagOptions,
        chatId: activeChatId,
        onChunk: appendToLastMessage,
        onSources: applySourcesToLastMessage,
      });

      if (result.chatId) {
        setActiveChatId(result.chatId);
      }

      await refreshChats(provider);
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
      setIsSending(false);
    }
  };

  const inputDisabled = sessionBusy;

  return (
    <div className="app">
      <div className="app-config">
        <h2 className="app-config-heading">Custom Configuration</h2>

        <ProviderModelSelector
          providerValue={provider}
          modelValue={model}
          ragPipelineValue={ragPipeline}
          disabled={isSending}
          ragPipelineDisabled={ragOptionsLocked}
          onProviderChange={handleProviderChange}
          onModelChange={setModel}
          onRagPipelineChange={setRagPipeline}
        />

        <div className="advanced-options-bar">
          <button
            type="button"
            className="advanced-options-toggle"
            onClick={() => setShowAdvancedOptions((open) => !open)}
            disabled={isSending}
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
          <span
            className={`rag-options-hint rag-options-hint-desktop${
              ragOptionsLocked ? " rag-options-hint-locked" : ""
            }`}
          >
            {ragOptionsLocked
              ? "Locked while documents are attached. Start a new chat or remove all attachments to change."
              : "Configure before uploading documents."}
          </span>
          <span
            className={`rag-options-hint rag-options-hint-mobile${
              ragOptionsLocked ? " rag-options-hint-locked" : ""
            }`}
          >
            {ragOptionsLocked ? "(locked — remove docs or new chat)" : "(configure before upload)"}
          </span>
        </div>

        {showAdvancedOptions && (
          <RagOptionsPanel
            ragOptions={ragOptions}
            onRagOptionsChange={setRagOptions}
            disabled={isSending || ragOptionsLocked}
            chunkingStrategies={ragCatalog?.chunking_strategies}
            embeddingProviders={ragCatalog?.embedding_providers}
            vectorStores={ragCatalog?.vector_stores}
            embeddingModels={ragCatalog?.embedding_models ?? {}}
          />
        )}
      </div>

      <div className="app-conversation">
        <ChatHistory
          messages={messages}
          isLoading={isSending || isLoadingMessages || isLoadingSessions}
          loadingLabel={
            isLoadingSessions || isLoadingMessages ? "Loading chat..." : loadingLabel
          }
        />

        <MessageInput
          key={inputResetKey}
          onSend={handleSend}
          onStartNewChat={handleStartNewChat}
          disabled={inputDisabled}
          loadingLabel={loadingLabel}
          ragOptions={showAdvancedOptions ? ragOptions : null}
          onAttachmentsChange={handleAttachmentsChange}
        />
      </div>
    </div>
  );
}

export default ChatAssistant;
