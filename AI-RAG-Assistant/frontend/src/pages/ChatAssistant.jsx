import { useState } from "react";
import ProviderModelSelector from "../components/ProviderModelSelector";
import ChatHistory from "../components/ChatHistory";
import MessageInput from "../components/MessageInput";
import { DEFAULT_PROVIDER, DEFAULT_MODEL, getProvider } from "../constants/providers";
import { sendChatMessage } from "../api/chatApi";
import "./ChatAssistant.css";

function ChatAssistant() {
  const [provider, setProvider] = useState(DEFAULT_PROVIDER.value);
  const [model, setModel] = useState(DEFAULT_MODEL);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleProviderChange = (nextProvider) => {
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

  // The LLM only needs the actual conversation turns - UI-only error
  // bubbles and the not-yet-filled streaming placeholder are dropped.
  const toConversationHistory = (uiMessages) =>
    uiMessages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .filter((message) => message.content)
      .map((message) => ({ role: message.role, content: message.content }));

  const handleSend = async (text, documentIds) => {
    const conversation = [...toConversationHistory(messages), { role: "user", content: text }];

    // The assistant bubble is added up front, empty, and filled in place as
    // stream chunks arrive.
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);
    setIsLoading(true);

    try {
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
        // Keep any partial text already streamed in - only replace the
        // bubble outright if nothing arrived before the failure.
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
      <h1 className="app-title">AI RAG Assistant</h1>

      <ProviderModelSelector
        providerValue={provider}
        modelValue={model}
        onProviderChange={handleProviderChange}
        onModelChange={setModel}
      />

      <ChatHistory messages={messages} isLoading={isLoading} />

      <MessageInput onSend={handleSend} disabled={isLoading} />
    </div>
  );
}

export default ChatAssistant;
