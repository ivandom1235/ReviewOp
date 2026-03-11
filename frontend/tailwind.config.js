/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        slatebg: "#0f172a",
        panel: "#111827",
        accent: "#14b8a6",
        warning: "#f59e0b",
        danger: "#ef4444",
        success: "#22c55e",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(20,184,166,0.35), 0 8px 24px rgba(15,23,42,0.4)",
      },
      keyframes: {
        reveal: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        reveal: "reveal 550ms ease-out forwards",
      },
    },
  },
  plugins: [],
};
