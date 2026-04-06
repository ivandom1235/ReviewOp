import { useTheme } from "../theme/ThemeContext";

export function RouteLoading({ label = "Loading...", isDark }) {
  const theme = useTheme();
  const dark =
    typeof isDark === "boolean"
      ? isDark
      : theme.isDark;

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
