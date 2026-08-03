import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";

export default function ChatHistory({ messages, isLoading, loadingLabel = "Thinking..." }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="chat-history">
      <div className="chat-history-messages">
        {messages.length === 0 && !isLoading && (
          <p className="chat-history-empty">Send a message to start the conversation.</p>
        )}

        {messages.map((message, index) => (
          <MessageBubble
            key={index}
            role={message.role}
            content={message.content}
            warning={message.warning}
            sources={message.sources}
            pending={isLoading && index === messages.length - 1 && message.role === "assistant"}
            pendingLabel={loadingLabel}
          />
        ))}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
