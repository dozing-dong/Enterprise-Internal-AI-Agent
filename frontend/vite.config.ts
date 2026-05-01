import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Vite proxies same-origin paths to the FastAPI backend running on :8000.
// SSE works fine through Vite's HTTP proxy as long as we don't touch the body.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/chat": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/sessions": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/history": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
