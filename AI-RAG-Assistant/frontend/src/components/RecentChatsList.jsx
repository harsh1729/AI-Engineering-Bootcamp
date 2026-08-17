import { useLocation } from "react-router-dom";
import NewChatButton from "./NewChatButton";
import { useChatSession } from "../context/ChatSessionContext";

export default function RecentChatsList({ onSelect, compact = false }) {
  const location = useLocation();
  const {
    chats,
    activeChatId,
    chatProvider,
    sessionBusy,
    selectChat,
    startNewChat,
  } = useChatSession();

  const isAssistantRoute = location.pathname.startsWith("/assistant");
  if (!isAssistantRoute || !chatProvider) {
    return null;
  }

  const handleSelect = (chatId) => {
    selectChat(chatId, chatProvider);
    onSelect?.();
  };

  const handleStartNewChat = () => {
    startNewChat();
    onSelect?.();
  };

  return (
    <section
      className={`sidebar-recent-chats${compact ? " sidebar-recent-chats-compact" : ""}`}
      aria-label="Recent chats"
    >
      <div className="sidebar-recent-chats-header">
        <h2 className="sidebar-recent-chats-heading">Recent Chats</h2>
        <NewChatButton
          onStartNewChat={handleStartNewChat}
          disabled={sessionBusy}
          menuPlacement="below"
        />
      </div>

      {chats.length === 0 ? (
        <p className="sidebar-recent-chats-empty">No chats yet for this provider.</p>
      ) : (
        <ul className="sidebar-recent-chats-list">
          {chats.map((chat) => {
            const isActive = chat.id === activeChatId;

            return (
              <li key={chat.id}>
                <button
                  type="button"
                  className={`sidebar-recent-chat-item${
                    isActive ? " sidebar-recent-chat-item-active" : ""
                  }`}
                  onClick={() => handleSelect(chat.id)}
                  disabled={sessionBusy}
                  aria-current={isActive ? "true" : undefined}
                >
                  <span className="sidebar-recent-chat-item-title">{chat.title}</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
