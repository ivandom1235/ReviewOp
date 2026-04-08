import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { loginUser } from "../../api/client";
import { useAuth } from "../../auth/AuthContext";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const nav = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (!username.trim() || !password.trim()) {
      setError("Username and password are required.");
      return;
    }
    setLoading(true);
    try {
      const payload = await loginUser(username.trim(), password);
      login(payload);
      nav(payload.user.role === "admin" ? "/admin" : "/");
    } catch (ex) {
      setError(ex.message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-100 via-emerald-50 to-slate-100 p-4 dark:from-[#060b18] dark:via-[#0b1b2b] dark:to-[#060b18]">
      <div className="w-full max-w-md">
        <div className="mb-5 text-center">
          <p className="text-4xl font-extrabold tracking-[0.08em] text-emerald-700 dark:text-emerald-300">REVIEWOPS</p>
          <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">Product Intelligence + Review Platform</p>
        </div>
        <form onSubmit={handleSubmit} className="w-full rounded-2xl border border-slate-200 bg-white/95 p-6 shadow-lg dark:border-slate-700 dark:bg-slate-900/95">
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Login</h1>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">Use your username and password.</p>
          {error ? <p className="mt-3 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</p> : null}
          <div className="mt-4 space-y-3">
            <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
            <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Password" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          </div>
          <button disabled={loading} className="mt-4 w-full rounded-lg bg-emerald-600 px-4 py-2 font-medium text-white disabled:opacity-60">
            {loading ? "Signing in..." : "Login"}
          </button>
          <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">
            User account? <Link className="text-emerald-700 underline" to="/register">Register here</Link>
          </p>
        </form>
      </div>
    </div>
  );
}
