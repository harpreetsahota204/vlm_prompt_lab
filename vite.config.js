const { defineConfig } = require("vite");
const react = require("@vitejs/plugin-react").default;
const path = require("path");

module.exports = defineConfig({
  mode: "development",
  plugins: [
    react({ jsxRuntime: "classic" }),
  ],
  build: {
    minify: true,
    lib: {
      entry: path.resolve(__dirname, "src/index.ts"),
      name: "harpreetsahota_vlm_prompt_lab",
      fileName: (format) => `index.${format}.js`,
      formats: ["umd"],
    },
    rollupOptions: {
      external: [
        "react",
        "react-dom",
        "@fiftyone/operators",
        "@fiftyone/plugins",
        "@fiftyone/state",
        "recoil",
      ],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
          "@fiftyone/operators": "__foo__",
          "@fiftyone/plugins": "__fop__",
          "@fiftyone/state": "__fos__",
          recoil: "recoil",
        },
      },
    },
    sourcemap: true,
  },
  define: {
    "process.env.NODE_ENV": '"development"',
  },
  optimizeDeps: {
    exclude: ["react", "react-dom"],
  },
});
