/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Outfit", "system-ui", "sans-serif"],
        heading: ["Sora", "Outfit", "sans-serif"],
      },
      colors: {
        slatebg: "#0f172a",
        panel: "#111827",
        accent: "#14b8a6",
        warning: "#f59e0b",
        danger: "#ef4444",
        success: "#22c55e",
        brand: {
          primary: "hsl(var(--brand-primary))",
          secondary: "hsl(var(--brand-secondary))",
        },
      },
      borderRadius: {
        DEFAULT: "0.75rem",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(20,184,166,0.35), 0 8px 24px rgba(15,23,42,0.4)",
        "soft": "0 2px 8px rgba(0,0,0,0.06), 0 0 1px rgba(0,0,0,0.1)",
        "soft-lg": "0 4px 16px rgba(0,0,0,0.08), 0 0 1px rgba(0,0,0,0.1)",
      },
      keyframes: {
        reveal: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        "fade-in": {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      animation: {
        reveal: "reveal 550ms ease-out forwards",
        shimmer: "shimmer 1.5s ease-in-out infinite",
        "fade-in": "fade-in 300ms ease-out forwards",
      },
      transitionDuration: {
        DEFAULT: "150ms",
      },
    },
  },
  plugins: [],
};
