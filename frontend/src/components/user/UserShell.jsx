import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../../auth/AuthContext";
import { useTheme } from "../../theme/ThemeContext";

export default function UserShell({ children, title = "ReviewOps User Portal" }) {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const location = useLocation();
  const { isDark, setTheme } = useTheme();

  return (
    <div className="min-h-screen bg-app text-[hsl(var(--text-main))]">
      <a href="#main-content" className="skip-to-content">
        Skip to content
      </a>
      <header className="sticky top-0 z-10 border-b border-border-subtle bg-[hsla(var(--bg-surface)/0.82)] backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-5">
            <Link to="/" className="text-xl font-bold tracking-wide text-brand-primary" id="nav-logo">
              REVIEWOP USER
            </Link>
            <nav className="flex items-center gap-2 text-sm" aria-label="Main navigation">
              <Link to="/" id="nav-home" className={`rounded-lg px-3 py-1.5 transition-colors ${location.pathname === "/" ? "premium-gradient text-white" : "text-muted-main hover:text-brand-primary"}`}>
                Home
              </Link>
              <Link to="/create-review" id="nav-create-review" className={`rounded-lg px-3 py-1.5 transition-colors ${location.pathname === "/create-review" ? "premium-gradient text-white" : "text-muted-main hover:text-brand-primary"}`}>
                Create Review
              </Link>
              <Link to="/my-reviews" id="nav-my-reviews" className={`rounded-lg px-3 py-1.5 transition-colors ${location.pathname === "/my-reviews" ? "premium-gradient text-white" : "text-muted-main hover:text-brand-primary"}`}>
                My Reviews
              </Link>
            </nav>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <label className="inline-flex items-center gap-2" id="theme-toggle-label">
              <span className="text-muted-main">Day</span>
              <span className="relative inline-flex items-center">
                <input
                  type="checkbox"
                  className="peer sr-only"
                  checked={isDark}
                  onChange={(e) => setTheme(e.target.checked ? "dark" : "light")}
                  aria-label="Toggle dark mode"
                  id="theme-toggle"
                />
                <span className="h-7 w-12 rounded-full bg-slate-300 transition-all duration-300 peer-checked:bg-blue-600" />
                <span className="absolute left-1 top-1 h-5 w-5 rounded-full bg-white shadow transition-all duration-300 peer-checked:left-6" />
              </span>
              <span className="text-muted-main">Night</span>
            </label>
            <span className="text-muted-main">{user?.username}</span>
            <button
              type="button"
              id="btn-logout"
              onClick={() => {
                logout();
                nav("/login");
              }}
              className="rounded-lg border border-border-subtle bg-app/30 px-3 py-1.5 text-muted-main transition-colors hover:bg-app/50"
            >
              Logout
            </button>
          </div>
        </div>
      </header>
      <main id="main-content" className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 py-6">
        <h1 className="text-2xl font-semibold">{title}</h1>
        {children}
      </main>
    </div>
  );
}
