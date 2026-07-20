const ROLE_LABELS = {
  user: "User",
  assistant: "Assistant",
  error: "Assistant",
};

export default function MessageBubble({ role, content, warning, pending }) {
  const hasContent = content && content.trim().length > 0;

  return (
    <div className={`message message-${role}`}>
      <div className="message-label">{ROLE_LABELS[role]}:</div>

      {hasContent && <div className="message-content">{content}</div>}

      {!hasContent && pending && (
        <div className="message-content message-content-empty">Thinking...</div>
      )}

      {!hasContent && !pending && warning && (
        <div className="message-content message-content-empty">No answer was generated.</div>
      )}

      {warning && <div className="message-warning">⚠️ {warning}</div>}
    </div>
  );
}
