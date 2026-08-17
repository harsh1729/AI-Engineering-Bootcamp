import { useCallback, useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import {
  approveUser,
  disableUser,
  enableUser,
  fetchAdminUsers,
} from "../api/adminApi";
import { useAuth } from "../context/AuthContext";
import "../pages/Page.css";
import "./ApproveUsers.css";

function formatDate(value) {
  return new Date(value).toLocaleString();
}

function UserTable({ title, users, emptyMessage, renderAction }) {
  return (
    <section className="approve-users-panel">
      <h2 className="approve-users-panel-title">{title}</h2>
      <div className="approve-users-table-wrap">
        {users.length === 0 ? (
          <p className="approve-users-empty">{emptyMessage}</p>
        ) : (
          <table className="approve-users-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Registered</th>
                <th aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.id}>
                  <td>{user.name}</td>
                  <td>{user.email}</td>
                  <td>{formatDate(user.created_at)}</td>
                  <td className="approve-users-action-cell">{renderAction(user)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

export default function ApproveUsers() {
  const { user, isLoading } = useAuth();
  const [pendingUsers, setPendingUsers] = useState([]);
  const [approvedUsers, setApprovedUsers] = useState([]);
  const [loadingUsers, setLoadingUsers] = useState(true);
  const [error, setError] = useState("");
  const [busyUserId, setBusyUserId] = useState(null);

  const loadUsers = useCallback(async () => {
    setLoadingUsers(true);
    setError("");

    try {
      const data = await fetchAdminUsers();
      setPendingUsers(data.pending ?? []);
      setApprovedUsers(data.approved ?? []);
    } catch (loadError) {
      setError(loadError.message || "Could not load users.");
    } finally {
      setLoadingUsers(false);
    }
  }, []);

  useEffect(() => {
    if (user?.is_admin) {
      loadUsers();
    }
  }, [user?.is_admin, loadUsers]);

  const runAction = async (userId, action) => {
    setBusyUserId(userId);
    setError("");

    try {
      await action();
      await loadUsers();
    } catch (actionError) {
      setError(actionError.message || "Action failed. Please try again.");
    } finally {
      setBusyUserId(null);
    }
  };

  if (isLoading) {
    return <div className="page approve-users-page">Loading...</div>;
  }

  if (!user?.is_admin) {
    return <Navigate to="/" replace />;
  }

  return (
    <div className="page approve-users-page">
      <h1 className="page-title approve-users-title">Approve Users</h1>

      {error && <p className="approve-users-error">{error}</p>}
      {loadingUsers && <p className="approve-users-status">Loading users...</p>}

      <div className="approve-users-panels">
        <UserTable
          title="Pending Users"
          users={pendingUsers}
          emptyMessage="No users waiting for approval."
          renderAction={(row) => (
            <button
              type="button"
              className="approve-users-button approve-users-button-primary"
              disabled={busyUserId === row.id}
              onClick={() => runAction(row.id, () => approveUser(row.id))}
            >
              Approve
            </button>
          )}
        />

        <UserTable
          title="Approved Users"
          users={approvedUsers}
          emptyMessage="No approved users yet."
          renderAction={(row) =>
            row.is_enabled ? (
              <button
                type="button"
                className="approve-users-button approve-users-button-danger"
                disabled={busyUserId === row.id}
                onClick={() => runAction(row.id, () => disableUser(row.id))}
              >
                Disable
              </button>
            ) : (
              <button
                type="button"
                className="approve-users-button approve-users-button-primary"
                disabled={busyUserId === row.id}
                onClick={() => runAction(row.id, () => enableUser(row.id))}
              >
                Enable
              </button>
            )
          }
        />
      </div>
    </div>
  );
}
