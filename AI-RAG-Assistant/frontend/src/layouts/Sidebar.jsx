import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import "./Sidebar.css";

const NAV_ITEMS = [
  { to: "/", label: "Home", end: true },
  { to: "/assistant", label: "Chat Assistant" },
  { to: "/agents", label: "Agents" },
  { to: "/settings", label: "Settings" },
];

function getPageTitle(pathname) {
  const match = NAV_ITEMS.find((item) =>
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

export default function Sidebar() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const pageTitle = getPageTitle(location.pathname);

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

  return (
    <>
      <nav className="sidebar" aria-label="Main navigation">
        <div className="sidebar-desktop">
          <p className="sidebar-title">AI RAG Assistant</p>

          <ul className="sidebar-nav">
            {NAV_ITEMS.map((item) => (
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
          <h1 className="sidebar-mobile-title">{pageTitle}</h1>
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
          {NAV_ITEMS.map((item) => (
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
      </aside>
    </>
  );
}
