export function RouteLoading({ label = "Loading...", isDark }) {
  const dark =
    typeof isDark === "boolean"
      ? isDark
      : typeof document !== "undefined"
        ? document.documentElement.classList.contains("dark")
        : false;

  return (
    <div
      className={`route-loading-shell ${dark ? "route-loading-shell--dark" : "route-loading-shell--light"}`}
      role="status"
      aria-live="polite"
      aria-label={label}
    >
      <div className="route-loading-grid" aria-hidden="true" />
      <div className="route-loading-glow" aria-hidden="true" />
      <div className="route-loading-card">
        <div className="route-loading-kicker">ReviewOp</div>
        <p>{label}</p>
        <div className="route-loading-bar" />
      </div>
    </div>
  );
}
