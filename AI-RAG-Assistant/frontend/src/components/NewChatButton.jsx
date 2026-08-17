import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

const MENU_WIDTH_PX = 148;
const MENU_PORTAL_ID = "new-chat-menu-portal";

function PenIcon() {
  return (
    <svg
      className="new-chat-trigger-icon"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M12 20H21"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M16.5 3.5C17.3284 2.67157 18.6716 2.67157 19.5 3.5C20.3284 4.32843 20.3284 5.67157 19.5 6.5L7 19L3 20L4 16L16.5 3.5Z"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function computeMenuPosition(trigger, menuPlacement) {
  const rect = trigger.getBoundingClientRect();
  const left = Math.min(
    Math.max(8, rect.right - MENU_WIDTH_PX),
    window.innerWidth - MENU_WIDTH_PX - 8,
  );

  if (menuPlacement === "below") {
    return {
      top: rect.bottom + 8,
      left,
    };
  }

  return {
    top: rect.top - 8,
    left,
    transform: "translateY(-100%)",
  };
}

export default function NewChatButton({
  onStartNewChat,
  disabled = false,
  menuPlacement = "above",
}) {
  const containerRef = useRef(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [menuPosition, setMenuPosition] = useState(null);

  const updateMenuPosition = () => {
    const trigger = containerRef.current?.querySelector(".new-chat-trigger");
    if (!trigger) {
      return;
    }

    setMenuPosition(computeMenuPosition(trigger, menuPlacement));
  };

  useEffect(() => {
    if (!isMenuOpen) {
      return;
    }

    updateMenuPosition();

    const handleOutsideClick = (event) => {
      const menu = document.getElementById(MENU_PORTAL_ID);
      if (
        containerRef.current?.contains(event.target) ||
        menu?.contains(event.target)
      ) {
        return;
      }
      setIsMenuOpen(false);
    };

    const handleReposition = () => updateMenuPosition();

    const timeoutId = window.setTimeout(() => {
      document.addEventListener("mousedown", handleOutsideClick);
      document.addEventListener("touchstart", handleOutsideClick, { passive: true });
    }, 0);

    window.addEventListener("resize", handleReposition);
    window.addEventListener("scroll", handleReposition, true);

    return () => {
      window.clearTimeout(timeoutId);
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("touchstart", handleOutsideClick);
      window.removeEventListener("resize", handleReposition);
      window.removeEventListener("scroll", handleReposition, true);
    };
  }, [isMenuOpen, menuPlacement]);

  const handleTriggerClick = () => {
    if (disabled) {
      return;
    }

    if (!isMenuOpen) {
      updateMenuPosition();
    }
    setIsMenuOpen((open) => !open);
  };

  const handleStartNewChat = () => {
    setIsMenuOpen(false);
    onStartNewChat();
  };

  const menuPortal =
    isMenuOpen &&
    menuPosition &&
    createPortal(
      <div
        id={MENU_PORTAL_ID}
        className="new-chat-menu new-chat-menu-portal"
        style={{
          ...menuPosition,
          width: "max-content",
        }}
        role="menu"
      >
        <button type="button" onClick={handleStartNewChat} role="menuitem">
          Start new chat
        </button>
      </div>,
      document.body,
    );

  return (
    <div className="new-chat-button" ref={containerRef}>
      <button
        type="button"
        className="message-input-action new-chat-trigger"
        onClick={handleTriggerClick}
        disabled={disabled}
        aria-label="Start new chat"
        aria-expanded={isMenuOpen}
        aria-haspopup="menu"
        title="Start new chat"
      >
        <PenIcon />
      </button>
      {menuPortal}
    </div>
  );
}
