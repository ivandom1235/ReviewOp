import { build } from "vite";
import react from "@vitejs/plugin-react";

try {
  await build({
    configFile: false,
    esbuild: {
      include: /\.[jt]sx?$/,
    },
    resolve: {
      preserveSymlinks: true,
    },
    build: {
      minify: false,
    },
    plugins: [react()],
  });
} catch (error) {
  console.error(error);
  process.exit(1);
}
