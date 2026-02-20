import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Webpack config for react-pdf (PDF.js worker)
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    return config;
  },
  // Turbopack equivalent (Next.js 16 default bundler)
  turbopack: {
    resolveAlias: {
      canvas: { browser: "" },
    },
  },
};

export default nextConfig;
