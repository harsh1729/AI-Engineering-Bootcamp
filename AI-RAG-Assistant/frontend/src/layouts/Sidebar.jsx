import { NavLink } from "react-router-dom";
import "./Sidebar.css";

const NAV_ITEMS = [
  { to: "/", label: "Home", end: true },
  { to: "/assistant", label: "Chat Assistant" },
  { to: "/agents", label: "Agents" },
  { to: "/settings", label: "Settings" },
];

export default function Sidebar() {
  return (
    <nav className="sidebar">
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
    </nav>
  );
}
