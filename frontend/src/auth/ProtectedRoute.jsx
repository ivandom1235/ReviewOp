import { Navigate } from "react-router-dom";
import { useAuth } from "./AuthContext";
import { RouteLoading } from "../components/RouteLoading";

export function AdminRoute({ children }) {
  const { user, loading, isAuthenticated } = useAuth();
  if (loading) return <RouteLoading label="Checking admin access..." />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (user.role !== "admin") return <Navigate to="/" replace />;
  return children;
}

export function UserRoute({ children }) {
  const { user, loading, isAuthenticated } = useAuth();
  if (loading) return <RouteLoading label="Checking account..." />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (user.role !== "user") return <Navigate to="/admin" replace />;
  return children;
}
