import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./hooks/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-geist)", "Inter", "system-ui", "sans-serif"],
      },
      colors: {
        ink: {
          950: "#020617",
          900: "#0f172a",
          800: "#1e293b",
        },
      },
      boxShadow: {
        glow: "0 0 40px -12px rgba(56, 189, 248, 0.45)",
        "glow-amber": "0 0 40px -12px rgba(251, 191, 36, 0.4)",
        "glow-violet": "0 0 40px -12px rgba(167, 139, 250, 0.45)",
        "glow-emerald": "0 0 40px -12px rgba(52, 211, 153, 0.4)",
      },
      backgroundImage: {
        grain:
          "radial-gradient(ellipse at top, rgba(56,189,248,0.08), transparent 50%), radial-gradient(ellipse at bottom right, rgba(167,139,250,0.08), transparent 45%)",
      },
    },
  },
  plugins: [],
};

export default config;
