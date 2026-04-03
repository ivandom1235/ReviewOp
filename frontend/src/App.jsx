import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth/AuthContext";
import { AdminRoute, UserRoute } from "./auth/ProtectedRoute";
import { RouteLoading } from "./components/RouteLoading";

const AdminPortal = lazy(() => import("./admin/pages/AdminPortal"));
const LoginPage = lazy(() => import("./pages/user/LoginPage"));
const RegisterPage = lazy(() => import("./pages/user/RegisterPage"));
const UserHomePage = lazy(() => import("./pages/user/UserHomePage"));
const SearchResultsPage = lazy(() => import("./pages/user/SearchResultsPage"));
const ProductPage = lazy(() => import("./pages/user/ProductPage"));
const SubmitReviewPage = lazy(() => import("./pages/user/SubmitReviewPage"));
const MyReviewsPage = lazy(() => import("./pages/user/MyReviewsPage"));

function AppRoutes() {
  const { user } = useAuth();
  return (
    <Suspense fallback={<RouteLoading label="Loading module..." />}>
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
    </Suspense>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
