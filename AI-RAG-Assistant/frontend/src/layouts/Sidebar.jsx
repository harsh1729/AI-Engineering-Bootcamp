import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import RecentChatsList from "../components/RecentChatsList";
import { useAuth } from "../context/AuthContext";
import { useChatSession } from "../context/ChatSessionContext";
import "./Sidebar.css";

const NAV_ITEMS = [
  { to: "/", label: "Home", end: true },
  { to: "/assistant", label: "Chat Assistant" },
  { to: "/agents", label: "Agents" },
  { to: "/settings", label: "Settings" },
];

const ADMIN_NAV_ITEM = { to: "/approve-users", label: "Approve Users" };

function getNavItems(isAdmin) {
  if (!isAdmin) {
    return NAV_ITEMS;
  }

  return [
    NAV_ITEMS[0],
    NAV_ITEMS[1],
    NAV_ITEMS[2],
    ADMIN_NAV_ITEM,
    NAV_ITEMS[3],
  ];
}

function getPageTitle(pathname, isAdmin) {
  const match = getNavItems(isAdmin).find((item) =>
    item.end ? pathname === item.to : pathname.startsWith(item.to),
  );
  return match?.label ?? "AI RAG Assistant";
}

function MenuIcon() {
  return (
    <svg
      className="sidebar-menu-icon"
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M4 7H20M4 12H20M4 17H20"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg
      className="sidebar-menu-icon"
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M6 6L18 18M18 6L6 18"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

function getFirstName(fullName) {
  const trimmed = fullName?.trim();
  if (!trimmed) {
    return "there";
  }
  return trimmed.split(/\s+/)[0];
}

export default function Sidebar() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const { activeChatTitle } = useChatSession();
  const { user, logout, isLoading: isAuthLoading } = useAuth();
  const pageTitle = getPageTitle(location.pathname, user?.is_admin);
  const isAssistantRoute = location.pathname.startsWith("/assistant");
  const navItems = getNavItems(Boolean(user?.is_admin));

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (!menuOpen) return;

    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [menuOpen]);

  const closeMenu = () => setMenuOpen(false);

  const handleLogout = () => {
    const confirmed = window.confirm("Log out of your account?");
    if (!confirmed) {
      return;
    }

    logout();
    closeMenu();
  };

  const authSection = (
    <div className="sidebar-auth">
      {isAuthLoading ? (
        <p className="sidebar-auth-status">Checking session...</p>
      ) : user ? (
        <>
          <p className="sidebar-auth-welcome">Welcome {getFirstName(user.name)}</p>
          <button type="button" className="sidebar-auth-logout" onClick={handleLogout}>
            Log out
          </button>
        </>
      ) : (
        <p className="sidebar-auth-links">
          <NavLink to="/login" className="sidebar-auth-link" onClick={closeMenu}>
            Log-in
          </NavLink>
          <span className="sidebar-auth-separator" aria-hidden="true">
            {" / "}
          </span>
          <NavLink to="/register" className="sidebar-auth-link" onClick={closeMenu}>
            Register
          </NavLink>
        </p>
      )}
    </div>
  );

  return (
    <>
      <nav className="sidebar" aria-label="Main navigation">
        <div className="sidebar-desktop">
          <p className="sidebar-title">AI RAG Assistant</p>

          <ul className="sidebar-nav">
            {navItems.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) =>
                    `sidebar-link${isActive ? " sidebar-link-active" : ""}`
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>

          <RecentChatsList />
          {authSection}
        </div>

        <div className="sidebar-mobile-header">
          <button
            type="button"
            className="sidebar-menu-toggle"
            onClick={() => setMenuOpen((open) => !open)}
            aria-expanded={menuOpen}
            aria-controls="sidebar-mobile-drawer"
            aria-label={menuOpen ? "Close navigation menu" : "Open navigation menu"}
          >
            {menuOpen ? <CloseIcon /> : <MenuIcon />}
          </button>
          <div className="sidebar-mobile-title-wrap">
            <h1 className="sidebar-mobile-title">{pageTitle}</h1>
            {isAssistantRoute && activeChatTitle && (
              <p className="sidebar-mobile-subtitle">{activeChatTitle}</p>
            )}
          </div>
        </div>
      </nav>

      {menuOpen && (
        <button
          type="button"
          className="sidebar-overlay"
          aria-label="Close navigation menu"
          onClick={closeMenu}
        />
      )}

      <aside
        id="sidebar-mobile-drawer"
        className={`sidebar-drawer${menuOpen ? " sidebar-drawer-open" : ""}`}
        aria-hidden={!menuOpen}
      >
        <div className="sidebar-drawer-header">
          <p className="sidebar-drawer-app-name">AI RAG Assistant</p>
          <button
            type="button"
            className="sidebar-drawer-close"
            onClick={closeMenu}
            aria-label="Close navigation menu"
          >
            <CloseIcon />
          </button>
        </div>

        <ul className="sidebar-nav sidebar-drawer-nav">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `sidebar-link${isActive ? " sidebar-link-active" : ""}`
                }
                onClick={closeMenu}
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <RecentChatsList compact onSelect={closeMenu} />
        {authSection}
      </aside>
    </>
  );
}
