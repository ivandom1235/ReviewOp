import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { useEffect, useState } from "react";

export default function UserShell({ children, title = "ReviewOps User Portal" }) {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const location = useLocation();
  const [theme, setTheme] = useState(() => localStorage.getItem("reviewop-user-theme") || "light");
  const isDark = theme === "dark";

  useEffect(() => {
    localStorage.setItem("reviewop-user-theme", theme);
    document.documentElement.classList.toggle("dark", isDark);
  }, [theme, isDark]);

  return (
    <div className={`min-h-screen ${isDark ? "bg-[#060b18] text-slate-100" : "bg-slate-50 text-slate-900"}`}>
      <header className={`sticky top-0 z-10 border-b backdrop-blur ${isDark ? "border-slate-800 bg-[#050a16]" : "border-slate-200 bg-white/95"}`}>
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-5">
            <Link to="/" className="text-xl font-bold tracking-wide text-emerald-500">
              REVIEWOP USER
            </Link>
            <nav className="flex items-center gap-2 text-sm">
              <Link to="/" className={`rounded-lg px-3 py-1.5 ${location.pathname === "/" ? "bg-emerald-500 text-slate-950" : isDark ? "text-slate-300" : "text-slate-600"}`}>
                Home
              </Link>
              <Link to="/create-review" className={`rounded-lg px-3 py-1.5 ${location.pathname === "/create-review" ? "bg-emerald-500 text-slate-950" : isDark ? "text-slate-300" : "text-slate-600"}`}>
                Create Review
              </Link>
              <Link to="/my-reviews" className={`rounded-lg px-3 py-1.5 ${location.pathname === "/my-reviews" ? "bg-emerald-500 text-slate-950" : isDark ? "text-slate-300" : "text-slate-600"}`}>
                My Reviews
              </Link>
            </nav>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <label className="inline-flex items-center gap-2">
              <span className={isDark ? "text-slate-300" : "text-slate-600"}>Day</span>
              <span className="relative inline-flex items-center">
                <input
                  type="checkbox"
                  className="peer sr-only"
                  checked={isDark}
                  onChange={(e) => setTheme(e.target.checked ? "dark" : "light")}
                  aria-label="Theme toggle"
                />
                <span className="h-7 w-12 rounded-full bg-slate-300 transition-all duration-300 peer-checked:bg-indigo-600" />
                <span className="absolute left-1 top-1 h-5 w-5 rounded-full bg-white shadow transition-all duration-300 peer-checked:left-6" />
              </span>
              <span className={isDark ? "text-slate-300" : "text-slate-600"}>Night</span>
            </label>
            <span className={isDark ? "text-slate-300" : "text-slate-600"}>{user?.username}</span>
            <button
              type="button"
              onClick={() => {
                logout();
                nav("/login");
              }}
              className={`rounded-lg border px-3 py-1.5 ${isDark ? "border-slate-700 hover:bg-slate-800" : "border-slate-300 hover:bg-slate-100"}`}
            >
              Logout
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 py-6">
        <h1 className="text-2xl font-semibold">{title}</h1>
        {children}
      </main>
    </div>
  );
}
