import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { registerUser } from "../../api/client";

export default function RegisterPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    if (username.trim().length < 3) {
      setError("Username must be at least 3 characters.");
      return;
    }
    if (password.length < 5) {
      setError("Password must be at least 5 characters.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setLoading(true);
    try {
      await registerUser(username.trim(), password);
      nav("/login");
    } catch (ex) {
      setError(ex.message || "Registration failed");
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
          <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100">Create User Account</h1>
          {error ? <p className="mt-3 rounded-lg bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{error}</p> : null}
          <div className="mt-4 space-y-3">
            <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
            <input value={password} onChange={(e) => setPassword(e.target.value)} type="password" placeholder="Password" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
            <input value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} type="password" placeholder="Confirm Password" className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-slate-900 placeholder:text-slate-400 focus:border-emerald-500 focus:outline-none dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100" />
          </div>
          <button disabled={loading} className="mt-4 w-full rounded-lg bg-emerald-600 px-4 py-2 font-medium text-white disabled:opacity-60">
            {loading ? "Creating account..." : "Register"}
          </button>
          <p className="mt-4 text-sm text-slate-600 dark:text-slate-300">
            Already have an account? <Link className="text-emerald-700 underline" to="/login">Login</Link>
          </p>
        </form>
      </div>
    </div>
  );
}
