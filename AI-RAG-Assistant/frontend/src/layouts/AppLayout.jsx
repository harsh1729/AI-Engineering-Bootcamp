import { AuthProvider } from "../context/AuthContext";
import { ChatSessionProvider } from "../context/ChatSessionContext";
import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";
import "./AppLayout.css";

export default function AppLayout() {
  return (
    <AuthProvider>
      <ChatSessionProvider>
        <div className="app-layout">
          <Sidebar />
          <main className="app-main">
            <Outlet />
          </main>
        </div>
      </ChatSessionProvider>
    </AuthProvider>
  );
}
