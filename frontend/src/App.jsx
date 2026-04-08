import { Navigate, Route, Routes } from "react-router-dom";
import AdminPortal from "./admin/pages/AdminPortal";
import LoginPage from "./pages/user/LoginPage";
import RegisterPage from "./pages/user/RegisterPage";
import UserHomePage from "./pages/user/UserHomePage";
import SearchResultsPage from "./pages/user/SearchResultsPage";
import ProductPage from "./pages/user/ProductPage";
import SubmitReviewPage from "./pages/user/SubmitReviewPage";
import MyReviewsPage from "./pages/user/MyReviewsPage";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AdminRoute, UserRoute } from "./auth/ProtectedRoute";

function AppRoutes() {
  const { user } = useAuth();
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route
        path="/admin"
        element={
          <AdminRoute>
            <AdminPortal />
          </AdminRoute>
        }
      />

      <Route
        path="/"
        element={
          <UserRoute>
            <UserHomePage />
          </UserRoute>
        }
      />
      <Route
        path="/search"
        element={
          <UserRoute>
            <SearchResultsPage />
          </UserRoute>
        }
      />
      <Route
        path="/products/:productId"
        element={
          <UserRoute>
            <ProductPage />
          </UserRoute>
        }
      />
      <Route
        path="/create-review"
        element={
          <UserRoute>
            <SubmitReviewPage />
          </UserRoute>
        }
      />
      <Route
        path="/products/:productId/review"
        element={
          <UserRoute>
            <SubmitReviewPage />
          </UserRoute>
        }
      />
      <Route
        path="/my-reviews"
        element={
          <UserRoute>
            <MyReviewsPage />
          </UserRoute>
        }
      />

      <Route path="*" element={<Navigate to={user?.role === "admin" ? "/admin" : "/"} replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
