import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const backendTarget = env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: {
      proxy: {
        "/health": { target: backendTarget, changeOrigin: true },
        "/infer": { target: backendTarget, changeOrigin: true },
        "/jobs": { target: backendTarget, changeOrigin: true },
        "/analytics": { target: backendTarget, changeOrigin: true },
      },
    },
  };
});
